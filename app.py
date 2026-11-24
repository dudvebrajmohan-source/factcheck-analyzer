import json
import os
import re
from pathlib import Path

import requests
import streamlit as st
from deep_translator import GoogleTranslator
from dotenv import load_dotenv
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
load_dotenv()
NEWSDATA_API_KEY = os.getenv("NEWSDATA_API_KEY") or "pub_15a58d084651412fa1b95adb5b6a8975"
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "data" / "fact_database.json"

MATCH_THRESHOLD = 0.35      # local DB match strictness
EVIDENCE_THRESHOLD = 0.15   # how closely an article must match the claim

st.set_page_config(
    page_title="FactCheck Analyzer — AI Global News & Sentiment",
    page_icon="🔎",
    layout="centered",
)

st.markdown(
    """
    <style>
    .stButton > button {
        background: linear-gradient(135deg, #4F46E5 0%, #6366F1 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        box-shadow: 0 4px 14px rgba(79, 70, 229, 0.3);
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(79, 70, 229, 0.5);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

analyzer = SentimentIntensityAnalyzer()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def translate_to_english(text):
    """Translate non-English input. Plain ASCII text skips the network call."""
    text = (text or "").strip()
    if not text or text.isascii():
        return text
    try:
        translated = GoogleTranslator(source="auto", target="en").translate(text)
        return translated.strip() if translated else text
    except Exception:
        return text


def extract_keywords(text, max_terms=5):
    """Return up to max_terms unique, meaningful words from the text."""
    words = re.findall(r"[a-z0-9]+", (text or "").lower())
    seen, keywords = set(), []
    for word in words:
        if len(word) > 2 and word not in ENGLISH_STOP_WORDS and word not in seen:
            seen.add(word)
            keywords.append(word)
    # NewsData.io free 'q' works best with concise query terms
    return " ".join(keywords[:max_terms]) or (text or "").strip()


@st.cache_resource(show_spinner=False)
def load_fact_index():
    """Load the local fact database (list or {'claims': [...]}) and build TF-IDF index."""
    if not DB_PATH.exists():
        return None, None, None, "Local database file not found."
    try:
        with open(DB_PATH, encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, dict):
            claims = raw.get("claims", [])
        elif isinstance(raw, list):
            claims = raw
        else:
            claims = []
    except (OSError, json.JSONDecodeError) as e:
        return None, None, None, f"Error reading local database: {e}"

    if not claims:
        return None, None, None, "Local database is empty."

    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
    matrix = vectorizer.fit_transform([str(c.get("claim", "")) for c in claims])
    return claims, vectorizer, matrix, None


def check_local_database(query):
    claims, vectorizer, matrix, error = load_fact_index()
    if error:
        return None, error
    sims = cosine_similarity(vectorizer.transform([query]), matrix)[0]
    best = int(sims.argmax())
    if sims[best] >= MATCH_THRESHOLD:
        return claims[best], None
    return None, "No matching record found in local database."


def _normalize_newsdata_items(items):
    articles = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        articles.append(
            {
                "title": item.get("title") or "No title",
                "description": (
                    item.get("description")
                    or item.get("content")
                    or "No description available."
                ),
                "url": item.get("link"),
                "urlToImage": item.get("image_url"),
                "publishedAt": item.get("pubDate") or "N/A",
                "source": {
                    "name": item.get("source_name")
                    or item.get("source_id")
                    or "NewsData.io"
                },
            }
        )
    return articles


@st.cache_data(ttl=600, show_spinner=False)
def search_newsdata(query, language="en"):
    """Fetch latest articles from NewsData.io. Returns (articles, error_message)."""
    if not NEWSDATA_API_KEY:
        return [], "NewsData API key missing. Add NEWSDATA_API_KEY to your .env file."

    # Keep query short for NewsData.io compatibility
    short_query = " ".join((query or "").split()[:5]).strip()
    if not short_query:
        return [], "Empty search query."

    params = {
        "apikey": NEWSDATA_API_KEY,
        "q": short_query,
        "language": language,
    }
    try:
        response = requests.get(
            "https://newsdata.io/api/1/latest", params=params, timeout=10
        )
    except requests.exceptions.RequestException as e:
        return [], f"Network error during NewsData fetch: {e}"

    if response.status_code != 200:
        return [], f"NewsData.io returned HTTP {response.status_code}."

    try:
        data = response.json()
    except ValueError:
        return [], "Invalid JSON response from NewsData.io."

    if data.get("status") != "success":
        results_obj = data.get("results")
        if isinstance(results_obj, dict):
            err_msg = results_obj.get("message") or data.get("message") or "unknown error"
        else:
            err_msg = data.get("message") or "unknown error"
        return [], f"NewsData API error: {err_msg}"

    return _normalize_newsdata_items(data.get("results")), None


@st.cache_data(ttl=600, show_spinner=False)
def search_newsapi_fallback(query):
    """Optional fallback to NewsAPI if NewsData returns no articles or fails."""
    if not NEWS_API_KEY:
        return [], None

    params = {
        "q": query,
        "sortBy": "publishedAt",
        "pageSize": 6,
        "language": "en",
        "apiKey": NEWS_API_KEY,
    }
    try:
        response = requests.get(
            "https://newsapi.org/v2/everything", params=params, timeout=8
        )
        if response.status_code != 200:
            return [], None
        data = response.json()
        if data.get("status") != "ok":
            return [], None
        return data.get("articles", []), None
    except requests.exceptions.RequestException:
        return [], None


def fetch_live_news(query):
    """Primary NewsData.io lookup with automatic NewsAPI fallback."""
    articles, newsdata_err = search_newsdata(query)
    if articles:
        return articles, "NewsData.io", None

    fallback_articles, _ = search_newsapi_fallback(query)
    if fallback_articles:
        return fallback_articles, "NewsAPI (Fallback)", None

    return [], "NewsData.io", newsdata_err


def count_supporting_articles(claim, articles):
    """Count articles whose title/description is similar to the claim."""
    texts = [
        f"{a.get('title') or ''} {a.get('description') or ''}".strip()
        for a in articles
    ]
    texts = [t for t in texts if t]
    if not texts:
        return 0
    try:
        vectorizer = TfidfVectorizer(stop_words="english")
        matrix = vectorizer.fit_transform(texts + [claim])
        sims = cosine_similarity(matrix[-1], matrix[:-1])[0]
    except ValueError:
        return 0
    return int((sims >= EVIDENCE_THRESHOLD).sum())


def tone_label(compound):
    if compound >= 0.05:
        return "Positive 😊"
    if compound <= -0.05:
        return "Negative ⚠️"
    return "Neutral 📄"


def show_verdict(verdict):
    v = str(verdict).lower()
    if v in ("verified", "true", "supported"):
        st.success("✅ VERIFIED / SUPPORTED")
    elif v in ("false", "contradicted"):
        st.error("❌ FALSE / CONTRADICTED")
    elif v in ("misleading", "mixed"):
        st.warning("⚠️ MISLEADING / MIXED")
    else:
        st.info("❓ UNVERIFIED")


def set_sample(text):
    st.session_state.user_input = text


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.subheader("AI Assistant Overview")
    st.markdown("**FactCheck Analyzer**")
    st.write(
        "Accepts English, Hindi, or typo-heavy input. It translates, extracts "
        "search terms, checks a local fact database, and fetches live news."
    )
    if NEWSDATA_API_KEY:
        st.success("NewsData.io API key active")
    else:
        st.warning("NewsData.io API key missing")
    if NEWS_API_KEY:
        st.caption("NewsAPI fallback: Enabled")
    st.info("Built for AI & Data Science College Micro-Project.")


# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------
st.title("🔎 FactCheck Analyzer")
st.write(
    "Enter a news claim, headline, or social media post in English or Hindi. "
    "The app checks a local fact database, live news coverage, and text tone."
)

if "user_input" not in st.session_state:
    st.session_state.user_input = ""

st.markdown("**💡 Try an example:**")
col1, col2, col3 = st.columns(3)
col1.button(
    "🌍 World Tech News",
    on_click=set_sample,
    args=("Artificial intelligence latest global breakthroughs",),
)
col2.button(
    "🇮🇳 Hindi News Query",
    on_click=set_sample,
    args=("दुनिया भर में नई तकनीक और अंतरिक्ष खोज",),
)
col3.button(
    "⚠️ Sentiment Sample",
    on_click=set_sample,
    args=("This new tool is amazing and super helpful!",),
)

user_input = st.text_area(
    "Ask your AI Assistant or enter a news claim:",
    key="user_input",
    placeholder="e.g. This is a terrible scam, or type something in Hindi...",
    height=100,
)

if st.button("🔍 Analyze", type="primary"):
    query_text = (st.session_state.get("user_input") or "").strip()
    if not query_text:
        st.error("Please enter a question or news claim to analyze.")
        st.stop()

    with st.spinner("Analyzing..."):
        normalized = translate_to_english(query_text)
        local_match, local_error = check_local_database(normalized)
        search_terms = extract_keywords(normalized)
        articles, source_used, news_error = fetch_live_news(search_terms)
        scores = analyzer.polarity_scores(normalized)
        compound = scores["compound"]

    if normalized != query_text:
        st.caption(f"🌐 Interpreted as: **{normalized}**")

    # 1. Local fact database
    st.subheader("🗂️ Local Fact Check")
    if local_match:
        show_verdict(
            local_match.get("verdict") or local_match.get("status") or "Unverified"
        )
        st.write(f"**Matched claim:** {local_match.get('claim', 'N/A')}")
        st.write(f"**Explanation:** {local_match.get('explanation', 'N/A')}")
    else:
        st.info(local_error or "No local match found.")

    # 2. Live news coverage
    st.subheader("🌐 Live News Coverage")
    st.caption(f"Search terms: {search_terms} · Provider: {source_used}")
    if news_error and not articles:
        st.warning(news_error)
    elif not articles:
        st.info("No recent articles found for these terms.")
    else:
        supporting = count_supporting_articles(normalized, articles)
        st.write(
            f"Found **{len(articles)}** articles. "
            f"**{supporting}** appear to discuss this claim."
        )
        for i, article in enumerate(articles, 1):
            title = article.get("title") or "No title"
            with st.expander(f"{i}. {title}"):
                image_url = article.get("urlToImage")
                if image_url:
                    st.image(image_url)
                source = (article.get("source") or {}).get("name") or "Unknown"
                st.caption(f"{source} · {article.get('publishedAt') or 'N/A'}")
                st.write(article.get("description") or "No description available.")
                if article.get("url"):
                    st.markdown(f"[Read full article]({article['url']})")

    # 3. Tone (not truth)
    st.subheader("📊 Tone of the Text")
    st.write(f"**Overall tone:** {tone_label(compound)}")
    st.write(f"**Compound score:** {compound:.3f}")
    st.caption(
        f"Scores — Pos: {scores['pos']:.3f}, Neu: {scores['neu']:.3f}, Neg: {scores['neg']:.3f} | "
        "This measures emotional tone only. It does not indicate whether the claim is true."
    )