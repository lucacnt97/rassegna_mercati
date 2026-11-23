import os
import re
import json
import html
import time
import asyncio
from pathlib import Path
from datetime import datetime, timezone, timedelta
from email.utils import parsedate_to_datetime
import feedparser
import edge_tts
import requests
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

# ==========================================
# CONFIGURAZIONE
# ==========================================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
SITE_URL = os.getenv("SITE_URL", "")

# Voce neurale unica naturale
VOICE_NAME = "it-IT-GiuseppeNeural"
VOICE_FALLBACK = "it-IT-DiegoNeural"
VOICE_RATE = "+0%"
MAX_HISTORY_DAYS = 45

DATA_DIR = Path("data")
DOCS_DIR = Path("docs")
AUDIO_DIR = DOCS_DIR / "audio"
HISTORY_FILE = DATA_DIR / "history.json"
HTML_OUTPUT = DOCS_DIR / "index.html"

# Feed RSS finanziari (IT + Internazionali + Settoriali/Commodities/Emergenti)
RSS_FEEDS = {
    "Il Sole 24 Ore - Finanza": "https://www.ilsole24ore.com/rss/finanza.xml",
    "Il Sole 24 Ore - Economia": "https://www.ilsole24ore.com/rss/economia.xml",
    "Milano Finanza": "https://www.milanofinanza.it/rss",
    "ANSA Economia": "https://www.ansa.it/sito/notizie/economia/economia_rss.xml",
    "ANSA Mondo & Geopolitica": "https://www.ansa.it/sito/notizie/mondo/mondo_rss.xml",
    "Reuters Business": "https://news.google.com/rss/search?q=site:reuters.com+markets+OR+finance&hl=en-US&gl=US&ceid=US:en",
    "Bloomberg Markets": "https://feeds.bloomberg.com/markets/news.rss",
    "Financial Times": "https://www.ft.com/rss/home",
    "Morningstar": "https://news.google.com/rss/search?q=site:morningstar.it+OR+site:morningstar.com+ETF+OR+markets&hl=it&gl=IT&ceid=IT:it",
    "CNBC Top News": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10000664",
    "MarketWatch Top Stories": "https://feeds.content.dowjones.io/public/rss/mw_topstories",
    "Oro & Commodities": "https://news.google.com/rss/search?q=gold+price+OR+bullion+OR+central+bank+gold+markets&hl=en-US&gl=US&ceid=US:en",
    "Mercati Emergenti": "https://news.google.com/rss/search?q=emerging+markets+OR+China+economy+OR+BRICS+markets&hl=en-US&gl=US&ceid=US:en",
    "Investing.com Italia": "https://it.investing.com/rss/news.rss",
}

# ==========================================
# MODELLI DATI (STRUCTURED OUTPUT GEMINI)
# ==========================================
class TopicSlide(BaseModel):
    category: str = Field(
        description=(
            "Usa esattamente una di queste 7 categorie nell'ordine indicato: "
            "1. Tema del Giorno, "
            "2. Macro, Tassi & Geopolitica, "
            "3. Azionario Globale, "
            "4. Rotazione & Focus Settori, "
            "5. Oro & Materie Prime, "
            "6. Mercati Emergenti, "
            "7. Focus Italia"
        )
    )
    headline: str = Field(description="Titolo sintetico e d'impatto (max 12 parole)")
    key_takeaway: str = Field(
        description="Impatto concreto sul portafoglio (azionario globale, settori, oro, emergenti o cambio EUR/USD) in 1 frase"
    )
    bullets: list[str] = Field(description="2-4 punti elenco con dati numerici, livelli o fatti concreti")
    sources: list[str] = Field(description="Nomi delle testate da cui proviene la notizia")

class DailyReport(BaseModel):
    day_headline: str = Field(
        description="Titolo generale della giornata finanziaria centrato sulla macro-evidenza principale (max 10 parole)"
    )
    market_sentiment: str = Field(description="Una parola: Risk-On, Risk-Off, Cauto, o Misto")
    slides: list[TopicSlide] = Field(
        description=(
            "Esattamente 7 schede in questo ordine fisso: "
            "1) Tema del Giorno (la macro-evidenza principale che muove i mercati), "
            "2) Macro, Tassi & Geopolitica (banche centrali, tassi d'interesse, inflazione, politica estera e cambio EUR/USD), "
            "3) Azionario Globale (USA, Europa e listini sviluppati), "
            "4) Rotazione & Focus Settori (settori migliori e peggiori della seduta: Tech/AI, Banche, Energia, Difesa, Healthcare, Lusso/Consumi), "
            "5) Oro & Materie Prime (andamento oro, tassi reali e beni rifugio), "
            "6) Mercati Emergenti (Cina, Asia, America Latina, dazi e impatto del dollaro), "
            "7) Focus Italia (Piazza Affari, spread/BTP ed economia italiana)."
        )
    )
    podcast_script: str = Field(
        description=(
            "Monologo fluido in italiano per un briefing audio mattutino di circa 4-5 minuti da ascoltare in auto, "
            "letto da un unico conduttore. "
            "Struttura narrativa obbligatoria: "
            "1) Saluto iniziale e citazione delle principali fonti consultate oggi; "
            "2) La macro-evidenza del giorno (il fatto principale che domina i mercati); "
            "3) Quadro macroeconomico, tassi d'interesse, geopolitica, politica estera e cambio euro-dollaro; "
            "4) Analisi del portafoglio globale: listini mondiali sviluppati e focus sui settori che guidano o frenano il mercato; "
            "5) Oro e mercati emergenti; "
            "6) Chiusura rapida dedicata all'Italia. "
            "NON inserire nomi di speaker (niente 'Marco:', 'Anna:', ecc.), NON usare elenchi puntati, asterischi o simboli. "
            "Scrivi sigle e numeri come si pronunciano (es. 'S and P cinquecento', 'Banca Centrale Europea', 'euro dollaro')."
        )
    )

# ==========================================
# 1. RACCOLTA NOTIZIE RSS
# ==========================================
def clean_html(raw_html: str) -> str:
    return re.sub(r"<.*?>", "", raw_html).strip()

def fetch_recent_news(hours_lookback: int = 20) -> tuple[str, list[str]]:
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours_lookback)
    articles = []
    seen = set()
    active_sources = []

    headers = {"User-Agent": "Mozilla/5.0 (compatible; MorningMarketBot/1.0)"}

    for source, url in RSS_FEEDS.items():
        try:
            resp = requests.get(url, headers=headers, timeout=12)
            feed = feedparser.parse(resp.content)
            count = 0
            for entry in feed.entries:
                if count >= 8:
                    break
                pub_str = entry.get("published") or entry.get("updated")
                if pub_str:
                    try:
                        pub_dt = parsedate_to_datetime(pub_str)
                        if pub_dt.tzinfo is None:
                            pub_dt = pub_dt.replace(tzinfo=timezone.utc)
                        if pub_dt < cutoff:
                            continue
                    except Exception:
                        pass

                title = entry.get("title", "").strip()
                key = title.lower()[:55]
                if not title or key in seen:
                    continue
                seen.add(key)

                summary = clean_html(entry.get("summary", entry.get("description", "")))[:260]
                articles.append(f"[{source}] {title}: {summary}")
                count += 1

            if count > 0:
                active_sources.append(source)
                print(f"   ✓ {source}: {count} notizie")
            else:
                print(f"   - {source}: nessuna notizia recente")
        except Exception as e:
            print(f"   ✗ Errore feed {source}: {e}")

    return "\n".join(articles), active_sources

# ==========================================
# 2. ELABORAZIONE CON GEMINI FLASH (SOLO FREE TIER)
# ==========================================
def generate_report(news_text: str, date_label: str, active_sources: list[str]) -> DailyReport:
    client = genai.Client(api_key=GEMINI_API_KEY)
    sources_str = ", ".join(active_sources)

    prompt = f"""
    Sei un analista finanziario senior e conduttore di un briefing mattutino pensato per un investitore europeo
    con un portafoglio core diversificato su Azionario Globale (Developed Markets), Oro e Mercati Emergenti.
    Data di oggi: {date_label}.
    Fonti consultate oggi: {sources_str}.

    Analizza le notizie raccolte, elimina il rumor speculativo di breve periodo su titoli minori, e costruisci:
    1. Le 7 schede tematiche nell'ordine esatto richiesto:
       - Tema del Giorno (la macro-evidenza principale)
       - Macro, Tassi & Geopolitica (banche centrali, tassi, politica estera, geopolitica e cambio EUR/USD)
       - Azionario Globale (andamento listini sviluppati USA/Europa)
       - Rotazione & Focus Settori (quali settori stanno sovraperformando o soffrendo: Tech/AI, Finanziari, Energia, Difesa, Healthcare, Consumi)
       - Oro & Materie Prime (driver su oro, tassi reali e beni rifugio)
       - Mercati Emergenti (Cina, Asia, emergenti, dazi e impatto del dollaro)
       - Focus Italia (Piazza Affari, BTP/spread ed economia italiana)
    2. Lo script audio monologo per un singolo conduttore seguendo esattamente lo stesso filo logico, aprendo con la macro-evidenza del giorno e citando le fonti consultate ({sources_str}), passando poi a macro/tassi/geopolitica, azionario globale e focus sui settori, oro ed emergenti, e chiudendo con l'Italia.

    NOTIZIE:
    {news_text}
    """

    candidates = [
        "gemini-3.8-flash",
        "gemini-3.5-flash",
        "gemini-3.5-flash-lite",
    ]

    last_err = None
    for attempt in range(3):
        for model_name in candidates:
            try:
                print(f"   Tentativo {attempt + 1} con modello: {model_name}")
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=DailyReport,
                        temperature=0.35,
                    ),
                )
                return DailyReport.model_validate_json(response.text)
            except Exception as e:
                print(f"   Modello {model_name} non disponibile ({e})")
                last_err = e
                time.sleep(4)
        time.sleep(10)

    raise RuntimeError(f"Nessun modello Gemini Flash disponibile dopo i retry: {last_err}")

# ==========================================
# 3. GENERAZIONE AUDIO MP3 (VOCE UNICA NATURALE)
# ==========================================
async def generate_audio(script_text: str, output_path: Path):
    clean_text = re.sub(r"[*_#`]+", "", script_text).strip()

    for voice in (VOICE_NAME, VOICE_FALLBACK):
        try:
            print(f"   Sintesi vocale con: {voice}")
            communicate = edge_tts.Communicate(clean_text, voice, rate=VOICE_RATE)
            await communicate.save(str(output_path))
            if output_path.exists() and output_path.stat().st_size > 0:
                return
        except Exception as e:
            print(f"   Avviso TTS con {voice} ({e}), provo fallback...")
            await asyncio.sleep(2)

    raise RuntimeError("Impossibile generare il file audio con nessuna delle voci configurate.")

# ==========================================
# 4. AGGIORNAMENTO STORICO & GENERAZIONE HTML
# ==========================================
def update_history_and_html(
    date_iso: str,
    date_label: str,
    report: DailyReport,
    audio_rel_path: str,
    active_sources: list[str],
):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    history = []
    if HISTORY_FILE.exists():
        history = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))

    history = [item for item in history if item["date"] != date_iso]

    new_entry = {
        "date": date_iso,
        "date_label": date_label,
        "audio_path": audio_rel_path,
        "day_headline": report.day_headline,
        "market_sentiment": report.market_sentiment,
        "active_sources": active_sources,
        "slides": [s.model_dump() for s in report.slides],
        "podcast_script": report.podcast_script,
    }
    history.insert(0, new_entry)

    if len(history) > MAX_HISTORY_DAYS:
        for old in history[MAX_HISTORY_DAYS:]:
            old_audio = DOCS_DIR / old["audio_path"]
            if old_audio.exists():
                old_audio.unlink()
        history = history[:MAX_HISTORY_DAYS]

    HISTORY_FILE.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    history_json_safe = json.dumps(history, ensure_ascii=False).replace("</", "<\\/")

    html_template = f"""<!DOCTYPE html>
<html lang="it">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Morning Market Briefing &amp; Podcast</title>
  <meta name="mobile-web-app-capable" content="yes" />
  <meta name="apple-mobile-web-app-capable" content="yes" />
  <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent" />
  <meta name="theme-color" content="#0d1117" />
  <link rel="icon" type="image/svg+xml" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 512 512'%3E%3Crect width='512' height='512' rx='112' fill='%230d1117'/%3E%3Cpath d='M96 352l112-112 80 80 128-144' fill='none' stroke='%2358a6ff' stroke-width='44' stroke-linecap='round' stroke-linejoin='round'/%3E%3Cpath d='M336 176h80v80' fill='none' stroke='%233fb950' stroke-width='44' stroke-linecap='round' stroke-linejoin='round'/%3E%3C/svg%3E" />
  <link rel="apple-touch-icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 512 512'%3E%3Crect width='512' height='512' rx='112' fill='%230d1117'/%3E%3Cpath d='M96 352l112-112 80 80 128-144' fill='none' stroke='%2358a6ff' stroke-width='44' stroke-linecap='round' stroke-linejoin='round'/%3E%3Cpath d='M336 176h80v80' fill='none' stroke='%233fb950' stroke-width='44' stroke-linecap='round' stroke-linejoin='round'/%3E%3C/svg%3E" />
  <style>
    :root {{
      --bg: #0d1117;
      --card: #161b22;
      --border: #30363d;
      --text: #e6edf3;
      --muted: #8b949e;
      --accent: #58a6ff;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.6;
      padding: 16px;
      max-width: 920px;
      margin: 0 auto;
    }}
    header {{
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      padding-bottom: 16px;
      border-bottom: 1px solid var(--border);
      margin-bottom: 20px;
    }}
    h1 {{ font-size: 1.35rem; font-weight: 700; }}
    .controls {{ display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }}
    select, button {{
      background: var(--card);
      color: var(--text);
      border: 1px solid var(--border);
      padding: 8px 12px;
      border-radius: 8px;
      font-size: 0.9rem;
      cursor: pointer;
    }}
    button:hover, select:hover {{ border-color: var(--accent); }}
    button.active {{ background: var(--accent); color: #0d1117; font-weight: 600; border-color: var(--accent); }}
    .player-box {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 16px;
      margin-bottom: 16px;
    }}
    .player-top {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 10px;
      flex-wrap: wrap;
      gap: 8px;
    }}
    audio {{ width: 100%; height: 42px; margin-top: 6px; }}
    .speed-btns button {{ padding: 4px 10px; font-size: 0.8rem; }}
    .sources-banner {{
      background: rgba(88, 166, 255, 0.08);
      border: 1px solid rgba(88, 166, 255, 0.28);
      border-radius: 10px;
      padding: 12px 16px;
      margin-bottom: 18px;
      font-size: 0.88rem;
      color: #c9d1d9;
    }}
    .sources-banner strong {{ color: var(--accent); }}
    .source-pills {{
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-top: 8px;
    }}
    .source-pill {{
      background: var(--card);
      border: 1px solid var(--border);
      padding: 2px 9px;
      border-radius: 999px;
      font-size: 0.78rem;
      color: var(--text);
    }}
    .hero {{
      margin-bottom: 20px;
      padding: 16px;
      background: linear-gradient(135deg, #161b22 0%, #1c2432 100%);
      border: 1px solid var(--border);
      border-radius: 12px;
    }}
    .badge {{
      display: inline-block;
      padding: 3px 10px;
      border-radius: 999px;
      font-size: 0.78rem;
      font-weight: 600;
      background: rgba(88, 166, 255, 0.15);
      color: var(--accent);
      margin-bottom: 8px;
    }}
    .hero h2 {{ font-size: 1.4rem; margin-bottom: 4px; }}
    .slides-grid {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 16px;
    }}
    .slide-card {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 18px;
    }}
    .slide-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 6px;
      margin-bottom: 6px;
    }}
    .slide-cat {{
      font-size: 0.78rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--accent);
      font-weight: 700;
    }}
    .slide-src {{
      font-size: 0.75rem;
      color: var(--muted);
    }}
    .slide-card h3 {{ font-size: 1.15rem; margin-bottom: 8px; }}
    .takeaway {{
      background: rgba(255,255,255,0.04);
      border-left: 3px solid var(--accent);
      padding: 8px 12px;
      margin-bottom: 12px;
      font-size: 0.95rem;
      color: #c9d1d9;
    }}
    ul {{ padding-left: 20px; }}
    li {{ margin-bottom: 6px; font-size: 0.95rem; }}
    .archive-list {{ display: flex; flex-direction: column; gap: 14px; }}
    .hidden {{ display: none; }}
  </style>
</head>
<body>
  <header>
    <h1>📈 Rassegna Mercati</h1>
    <div class="controls">
      <select id="dateSelect" aria-label="Seleziona data"></select>
      <button id="btnDayView" class="active" onclick="setView('day')">Giorno</button>
      <button id="btnMonthView" onclick="setView('month')">Storico Ultimo Mese</button>
    </div>
  </header>

  <div id="dayView">
    <div class="player-box">
      <div class="player-top">
        <strong>🎧 Briefing Audio Mattutino (Modalità Auto)</strong>
        <div class="speed-btns">
          <button onclick="setSpeed(1)">1x</button>
          <button onclick="setSpeed(1.25)">1.25x</button>
          <button onclick="setSpeed(1.5)">1.5x</button>
        </div>
      </div>
      <audio id="audioPlayer" controls preload="metadata"></audio>
    </div>

    <div class="sources-banner">
      <strong>📰 Fonti consultate per questa rassegna:</strong>
      <div class="source-pills" id="sourcesContainer"></div>
    </div>

    <div class="hero">
      <span class="badge" id="sentimentBadge"></span>
      <h2 id="dayHeadline"></h2>
      <div style="color: var(--muted); font-size: 0.9rem;" id="dayDateLabel"></div>
    </div>

    <div class="slides-grid" id="slidesContainer"></div>
  </div>

  <div id="monthView" class="hidden">
    <h2 style="margin-bottom: 14px; font-size: 1.2rem;">Archivio ultimi 30+ giorni</h2>
    <div class="archive-list" id="archiveContainer"></div>
  </div>

  <script>
    const HISTORY = {history_json_safe};
    const dateSelect = document.getElementById("dateSelect");
    const audioPlayer = document.getElementById("audioPlayer");

    HISTORY.forEach((item, idx) => {{
      const opt = document.createElement("option");
      opt.value = idx;
      opt.textContent = `${{item.date_label}} — ${{item.day_headline}}`;
      dateSelect.appendChild(opt);
    }});

    function renderDay(index) {{
      const item = HISTORY[index];
      if (!item) return;
      document.getElementById("sentimentBadge").textContent = `Sentiment: ${{item.market_sentiment}}`;
      document.getElementById("dayHeadline").textContent = item.day_headline;
      document.getElementById("dayDateLabel").textContent = item.date_label;
      audioPlayer.src = item.audio_path;

      const sourcesContainer = document.getElementById("sourcesContainer");
      const sourcesList = item.active_sources || [
        "Il Sole 24 Ore", "Milano Finanza", "ANSA Economia", "Reuters", "Bloomberg", "Financial Times", "Morningstar"
      ];
      sourcesContainer.innerHTML = sourcesList.map(s => `<span class="source-pill">${{s}}</span>`).join("");

      const container = document.getElementById("slidesContainer");
      container.innerHTML = "";
      item.slides.forEach((slide, i) => {{
        const card = document.createElement("div");
        card.className = "slide-card";
        const bulletsHtml = slide.bullets.map(b => `<li>${{b}}</li>`).join("");
        const srcTag = (slide.sources && slide.sources.length) ? `Fonti: ${{slide.sources.join(", ")}}` : "";
        card.innerHTML = `
          <div class="slide-header">
            <span class="slide-cat">Slide ${{i + 1}}/${{item.slides.length}} • ${{slide.category}}</span>
            <span class="slide-src">${{srcTag}}</span>
          </div>
          <h3>${{slide.headline}}</h3>
          <div class="takeaway"><strong>Impatto:</strong> ${{slide.key_takeaway}}</div>
          <ul>${{bulletsHtml}}</ul>
        `;
        container.appendChild(card);
      }});
    }}

    function renderArchive() {{
      const container = document.getElementById("archiveContainer");
      container.innerHTML = "";
      HISTORY.forEach((item, idx) => {{
        const card = document.createElement("div");
        card.className = "slide-card";
        const topics = item.slides.map(s => `<li><strong>${{s.category}}:</strong> ${{s.headline}} — <em>${{s.key_takeaway}}</em></li>`).join("");
        card.innerHTML = `
          <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
            <span class="slide-cat">${{item.date_label}} • ${{item.market_sentiment}}</span>
            <button onclick="openDayFromArchive(${{idx}})">Apri briefing &amp; audio →</button>
          </div>
          <h3 style="margin-bottom:10px;">${{item.day_headline}}</h3>
          <ul>${{topics}}</ul>
        `;
        container.appendChild(card);
      }});
    }}

    function openDayFromArchive(idx) {{
      dateSelect.value = idx;
      renderDay(idx);
      setView("day");
      window.scrollTo({{ top: 0, behavior: "smooth" }});
    }}

    function setView(mode) {{
      const isDay = mode === "day";
      document.getElementById("dayView").classList.toggle("hidden", !isDay);
      document.getElementById("monthView").classList.toggle("hidden", isDay);
      document.getElementById("btnDayView").classList.toggle("active", isDay);
      document.getElementById("btnMonthView").classList.toggle("active", !isDay);
      if (!isDay) renderArchive();
    }}

    function setSpeed(rate) {{
      audioPlayer.playbackRate = rate;
    }}

    dateSelect.addEventListener("change", (e) => renderDay(e.target.value));
    renderDay(0);
  </script>
</body>
</html>"""

    HTML_OUTPUT.write_text(html_template, encoding="utf-8")

# ==========================================
# 5. NOTIFICA TELEGRAM (AUDIO + RIASSUNTO TESTUALE)
# ==========================================
def notify_telegram(date_label: str, report: DailyReport, audio_path: Path, active_sources: list[str]):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return

    base_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"
    sources_preview = ", ".join(active_sources[:6]) + ("..." if len(active_sources) > 6 else "")
    link_line = f'\n\n📊 <a href="{SITE_URL}">Apri presentazione HTML e storico</a>' if SITE_URL else ""

    caption = (
        f"🎧 <b>Briefing Mercati — {date_label}</b>\n"
        f"Sentiment: <b>{html.escape(report.market_sentiment)}</b>\n"
        f"📰 Fonti: {html.escape(sources_preview)}"
    )
    with open(audio_path, "rb") as f:
        requests.post(
            f"{base_url}/sendAudio",
            data={
                "chat_id": TELEGRAM_CHAT_ID,
                "caption": caption,
                "parse_mode": "HTML",
                "title": f"Mercati {date_label}",
                "performer": "Rassegna Mercati",
            },
            files={"audio": (audio_path.name, f, "audio/mpeg")},
        )

    lines = [f"📈 <b>{html.escape(report.day_headline)}</b>\n"]
    for s in report.slides:
        lines.append(f"🔹 <b>{html.escape(s.category.upper())}</b> — {html.escape(s.headline)}")
        lines.append(f"<i>{html.escape(s.key_takeaway)}</i>")
        for b in s.bullets[:2]:
            lines.append(f"• {html.escape(b)}")
        lines.append("")

    full_text = "\n".join(lines).strip() + link_line
    requests.post(
        f"{base_url}/sendMessage",
        json={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": full_text[:4096],
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        },
    )

# ==========================================
# MAIN
# ==========================================
def main():
    now = datetime.now()
    date_iso = now.strftime("%Y-%m-%d")
    date_label = now.strftime("%d/%m/%Y")

    print("1. Scaricamento feed RSS...")
    news_text, active_sources = fetch_recent_news(hours_lookback=20)
    if not news_text:
        print("Nessuna notizia recente trovata.")
        return

    print("2. Generazione schede e script con Gemini Flash...")
    report = generate_report(news_text, date_label, active_sources)

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    audio_filename = f"{date_iso}.mp3"
    audio_full_path = AUDIO_DIR / audio_filename

    print("3. Sintesi vocale briefing...")
    asyncio.run(generate_audio(report.podcast_script, audio_full_path))

    print("4. Aggiornamento storico e presentazione HTML...")
    update_history_and_html(date_iso, date_label, report, f"audio/{audio_filename}", active_sources)

    print("5. Invio su Telegram (se configurato)...")
    notify_telegram(date_label, report, audio_full_path, active_sources)
    print("Fatto!")

if __name__ == "__main__":
    main()
