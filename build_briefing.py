import os
import re
import json
import html
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
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")  # Opzionale
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")      # Opzionale
SITE_URL = os.getenv("SITE_URL", "")                      # Es. https://tuonome.github.io/rassegna-mercati/

VOICE_NAME = "it-IT-DiegoNeural"
VOICE_RATE = "+10%"
MAX_HISTORY_DAYS = 45  # Conserva oltre un mese di storico

DATA_DIR = Path("data")
DOCS_DIR = Path("docs")
AUDIO_DIR = DOCS_DIR / "audio"
HISTORY_FILE = DATA_DIR / "history.json"
HTML_OUTPUT = DOCS_DIR / "index.html"

RSS_FEEDS = {
    "Il Sole 24 Ore - Finanza": "https://www.ilsole24ore.com/rss/finanza.xml",
    "Il Sole 24 Ore - Economia": "https://www.ilsole24ore.com/rss/economia.xml",
    "CNBC Top News": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10000664",
    "MarketWatch Top Stories": "https://feeds.content.dowjones.io/public/rss/mw_topstories",
    "Yahoo Finance": "https://finance.yahoo.com/news/rssindex",
    "Investing.com Italia": "https://it.investing.com/rss/news.rss",
}

# ==========================================
# MODELLI DATI (STRUCTURED OUTPUT GEMINI)
# ==========================================
class TopicSlide(BaseModel):
    category: str = Field(description="Categoria: Macro & Banche Centrali, Azionario, Materie Prime & Valute, oppure Italia & Europa")
    headline: str = Field(description="Titolo sintetico e d'impatto della notizia/tema (max 12 parole)")
    key_takeaway: str = Field(description="Impatto pratico sui mercati in 1 frase chiara")
    bullets: list[str] = Field(description="2-4 punti elenco concreti con dati, livelli o dettagli rilevanti")

class DailyReport(BaseModel):
    day_headline: str = Field(description="Titolo generale della giornata finanziaria (max 10 parole)")
    market_sentiment: str = Field(description="Una parola: Risk-On, Risk-Off, Cauto, o Misto")
    slides: list[TopicSlide] = Field(description="Da 4 a 6 schede tematiche ordinate per importanza")
    podcast_script: str = Field(
        description=(
            "Script discorsivo in italiano per un podcast mattutino di 3-4 minuti da ascoltare in auto. "
            "VIETATI elenchi puntati, asterischi, simboli o tag. Scrivi sigle e numeri come si pronunciano "
            "(es. 'S and P cinquecento', 'Banca Centrale Europea', 'più zero virgola cinque percento'). "
            "Usa transizioni naturali tra un tema e l'altro."
        )
    )

# ==========================================
# 1. RACCOLTA NOTIZIE RSS
# ==========================================
def clean_html(raw_html: str) -> str:
    return re.sub(r"<.*?>", "", raw_html).strip()

def fetch_recent_news(hours_lookback: int = 20) -> str:
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours_lookback)
    articles = []
    seen = set()

    for source, url in RSS_FEEDS.items():
        try:
            feed = feedparser.parse(url)
            count = 0
            for entry in feed.entries:
                if count >= 12:
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

                summary = clean_html(entry.get("summary", entry.get("description", "")))[:300]
                articles.append(f"[{source}] {title}: {summary}")
                count += 1
        except Exception as e:
            print(f"Errore feed {source}: {e}")

    return "\n".join(articles)

# ==========================================
# 2. ELABORAZIONE CON GEMINI 3.8 FLASH (FREE TIER)
# ==========================================
def generate_report(news_text: str, date_label: str) -> DailyReport:
    client = genai.Client(api_key=GEMINI_API_KEY)
    prompt = f"""
    Sei un analista finanziario senior e conduttore di un briefing mattutino per investitori.
    Data di oggi: {date_label}.
    Analizza le notizie raccolte, elimina duplicati e rumore di fondo, e crea:
    1. Una presentazione strutturata in schede tematiche (chiare, ricche di numeri e livelli chiave).
    2. Uno script fluido per il podcast audio da ascoltare in auto.

    NOTIZIE:
    {news_text}
    """
    response = client.models.generate_content(
        model="gemini-flash-latest",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=DailyReport,
            temperature=0.3,
        ),
    )
    return DailyReport.model_validate_json(response.text)

# ==========================================
# 3. GENERAZIONE AUDIO MP3
# ==========================================
async def generate_audio(script_text: str, output_path: Path):
    communicate = edge_tts.Communicate(script_text, VOICE_NAME, rate=VOICE_RATE)
    await communicate.save(str(output_path))

# ==========================================
# 4. AGGIORNAMENTO STORICO & GENERAZIONE HTML
# ==========================================
def update_history_and_html(date_iso: str, date_label: str, report: DailyReport, audio_rel_path: str):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    history = []
    if HISTORY_FILE.exists():
        history = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))

    # Rimuovi eventuale entry dello stesso giorno se rieseguito
    history = [item for item in history if item["date"] != date_iso]

    new_entry = {
        "date": date_iso,
        "date_label": date_label,
        "audio_path": audio_rel_path,
        "day_headline": report.day_headline,
        "market_sentiment": report.market_sentiment,
        "slides": [s.model_dump() for s in report.slides],
        "podcast_script": report.podcast_script,
    }
    history.insert(0, new_entry)

    # Pulizia file audio più vecchi di MAX_HISTORY_DAYS
    if len(history) > MAX_HISTORY_DAYS:
        for old in history[MAX_HISTORY_DAYS:]:
            old_audio = DOCS_DIR / old["audio_path"]
            if old_audio.exists():
                old_audio.unlink()
        history = history[:MAX_HISTORY_DAYS]

    HISTORY_FILE.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")

    # Incorpora lo storico direttamente nell'HTML statico per consultazione istantanea e offline
    history_json_safe = json.dumps(history, ensure_ascii=False).replace("</", "<\\/")

    html_template = f"""<!DOCTYPE html>
<html lang="it">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Morning Market Briefing &amp; Podcast</title>
  <style>
    :root {{
      --bg: #0d1117;
      --card: #161b22;
      --border: #30363d;
      --text: #e6edf3;
      --muted: #8b949e;
      --accent: #58a6ff;
      --green: #3fb950;
      --red: #f85149;
      --amber: #d29922;
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
      margin-bottom: 20px;
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
    .slide-cat {{
      font-size: 0.78rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--accent);
      font-weight: 700;
      margin-bottom: 6px;
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
        <strong>🎧 Podcast Mattutino (Modalità Auto)</strong>
        <div class="speed-btns">
          <button onclick="setSpeed(1)">1x</button>
          <button onclick="setSpeed(1.25)">1.25x</button>
          <button onclick="setSpeed(1.5)">1.5x</button>
        </div>
      </div>
      <audio id="audioPlayer" controls preload="metadata"></audio>
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

      const container = document.getElementById("slidesContainer");
      container.innerHTML = "";
      item.slides.forEach((slide, i) => {{
        const card = document.createElement("div");
        card.className = "slide-card";
        const bulletsHtml = slide.bullets.map(b => `<li>${{b}}</li>`).join("");
        card.innerHTML = `
          <div class="slide-cat">Slide ${{i + 1}}/${{item.slides.length}} • ${{slide.category}}</div>
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
# 5. NOTIFICA TELEGRAM OPZIONALE
# ==========================================
def notify_telegram(date_label: str, report: DailyReport, audio_path: Path):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return

    base_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"
    link_line = f'\n\n📊 <a href="{SITE_URL}">Apri presentazione HTML e storico</a>' if SITE_URL else ""
    caption = f"📈 <b>{html.escape(report.day_headline)}</b> ({date_label})\nSentiment: {html.escape(report.market_sentiment)}{link_line}"

    with open(audio_path, "rb") as f:
        requests.post(
            f"{base_url}/sendAudio",
            data={"chat_id": TELEGRAM_CHAT_ID, "caption": caption, "parse_mode": "HTML", "title": f"Mercati {date_label}"},
            files={"audio": (audio_path.name, f, "audio/mpeg")},
        )

# ==========================================
# MAIN
# ==========================================
def main():
    now = datetime.now()
    date_iso = now.strftime("%Y-%m-%d")
    date_label = now.strftime("%d/%m/%Y")

    print("1. Scaricamento feed RSS...")
    news_text = fetch_recent_news(hours_lookback=20)
    if not news_text:
        print("Nessuna notizia recente trovata.")
        return

    print("2. Generazione schede e script con Gemini 3.8 Flash...")
    report = generate_report(news_text, date_label)

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    audio_filename = f"{date_iso}.mp3"
    audio_full_path = AUDIO_DIR / audio_filename

    print("3. Sintesi vocale podcast...")
    asyncio.run(generate_audio(report.podcast_script, audio_full_path))

    print("4. Aggiornamento storico e presentazione HTML...")
    update_history_and_html(date_iso, date_label, report, f"audio/{audio_filename}")

    print("5. Invio su Telegram (se configurato)...")
    notify_telegram(date_label, report, audio_full_path)
    print("Fatto!")

if __name__ == "__main__":
    main()
