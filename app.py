#!/usr/bin/env python3
# Streamlit YouTube Comments Sentiment Analysis

import random
import os
import re
import time
import requests
from urllib.parse import urlparse, parse_qs
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from io import BytesIO

# For PDF report
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import inch
import datetime

# Ensure NLTK data (VADER) is available
try:
    from nltk.sentiment import SentimentIntensityAnalyzer
except LookupError:
    nltk.download('vader_lexicon')
    from nltk.sentiment import SentimentIntensityAnalyzer

# Optional TextBlob support
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except Exception:
    TEXTBLOB_AVAILABLE = False

# ---------------------- Custom CSS ----------------------
custom_css = """
<style>
.stApp {
    background: linear-gradient(to bottom right, #ffffff, #f5f5f5);
    font-family: 'Segoe UI', Tahoma, sans-serif;
    color: #000000 !important;
}

/* Main title in bright orange */
h1 {
    color: #080000 !important;
    text-align: center;
    font-size: 2.8rem !important;
    margin-top: 0.5rem !important;
    margin-bottom: 1rem !important;
    font-weight: 800;
}

/* Subheaders in bright blue */
h2, h3 {
    color: #1a73e8 !important;
    margin-top: 1rem !important;
    font-weight: 700;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #673ab7, #3f51b5);
    color: white;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p {
    color: #ffffff !important;
    font-weight: 500;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: #ffffff;
    padding: 1rem;
    border-radius: 14px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.15);
    margin-bottom: 1rem;
    border-left: 6px solid #1a73e8;
    color: #000000 !important;
}

/* Buttons with gradient */
.stButton>button {
    background: linear-gradient(to right, #ff5722, #ff7043);
    color: white;
    border-radius: 12px;
    border: none;
    padding: 0.7rem 1.3rem;
    font-size: 1rem;
    font-weight: bold;
    cursor: pointer;
    transition: all 0.3s ease-in-out;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.2);
}
.stButton>button:hover {
    transform: scale(1.05);
    background: linear-gradient(to right, #ff7043, #ff5722);
}

/* Hide Streamlit Deploy button and 3-dot menu */
#MainMenu {visibility: hidden;}
header {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)



# ---------------------- Helpers ----------------------
YOUTUBE_API_URL = "https://www.googleapis.com/youtube/v3/commentThreads"

def get_api_key():
    key = None
    try:
        key = st.secrets.get("YOUTUBE_API_KEY", None)
    except Exception:
        pass
    if not key:
        key = os.getenv("YOUTUBE_API_KEY")
    return key

def extract_video_id(url_or_id: str) -> str:
    url_or_id = url_or_id.strip()
    if re.fullmatch(r"[A-Za-z0-9_\-]{6,}", url_or_id):
        return url_or_id
    parsed = urlparse(url_or_id)
    if 'youtube' in parsed.netloc or 'youtu.be' in parsed.netloc:
        if parsed.netloc.endswith("youtu.be"):
            return parsed.path.strip("/")
        qs = parse_qs(parsed.query)
        if "v" in qs:
            return qs["v"][0]
    return ""

def fetch_comments(video_id: str, api_key: str, max_comments: int = 500, order: str = "relevance") -> pd.DataFrame:
    comments = []
    params = {
        "part": "snippet",
        "videoId": video_id,
        "maxResults": 100,
        "key": api_key,
        "textFormat": "plainText",
        "order": order
    }
    next_token = None
    fetched = 0

    with st.spinner("Fetching comments from YouTube..."):
        while True:
            if next_token:
                params["pageToken"] = next_token
            resp = requests.get(YOUTUBE_API_URL, params=params, timeout=30)
            if resp.status_code != 200:
                raise RuntimeError(f"YouTube API error {resp.status_code}: {resp.text}")
            data = resp.json()
            for item in data.get("items", []):
                s = item["snippet"]["topLevelComment"]["snippet"]
                comments.append({
                    "author": s.get("authorDisplayName", ""),
                    "text": s.get("textOriginal", ""),
                    "likeCount": s.get("likeCount", 0),
                    "publishedAt": s.get("publishedAt", ""),
                })
                fetched += 1
                if fetched >= max_comments:
                    break
            if fetched >= max_comments:
                break
            next_token = data.get("nextPageToken")
            if not next_token:
                break
            time.sleep(0.1)
    df = pd.DataFrame(comments)
    if not df.empty:
        df["publishedAt"] = pd.to_datetime(df["publishedAt"], errors="coerce")
    return df

def basic_clean(text: str) -> str:
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"[@#]\w+", "", text)
    text = re.sub(r"[\r\n]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def add_vader_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    sia = SentimentIntensityAnalyzer()
    scores = df["clean_text"].apply(sia.polarity_scores).apply(pd.Series)
    df = pd.concat([df, scores], axis=1)
    df["sentiment"] = pd.cut(df["compound"], bins=[-1.01, -0.05, 0.05, 1.01],
                             labels=["Negative", "Neutral", "Positive"])
    return df

def add_textblob_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    if not TEXTBLOB_AVAILABLE:
        st.warning("TextBlob not installed; skipping TextBlob sentiments.")
        return df
    def tb_polarity(t):
        try:
            return TextBlob(t).sentiment.polarity
        except Exception:
            return np.nan
    df["tb_polarity"] = df["clean_text"].astype(str).apply(tb_polarity)
    df["tb_sentiment"] = pd.cut(df["tb_polarity"], bins=[-1.01, -0.05, 0.05, 1.01],
                                labels=["Negative", "Neutral", "Positive"])
    return df

def make_wordcloud(texts, title):
    txt = " ".join(texts)
    if not txt.strip():
        st.info(f"No text for {title} wordcloud.")
        return
    wc = WordCloud(width=800, height=400, background_color="white").generate(txt)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)

def generate_sentiment_report(df, video_url, filename="sentiment_report.pdf"):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4,
                            topMargin=40, bottomMargin=40,
                            leftMargin=40, rightMargin=40)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="CenterTitle", alignment=TA_CENTER,
                              fontSize=18, textColor=colors.HexColor("#ff6600"),
                              spaceAfter=20))
    styles.add(ParagraphStyle(name="SectionTitle", fontSize=14,
                              textColor=colors.HexColor("#1a73e8"),
                              spaceAfter=8))
    styles.add(ParagraphStyle(name="NormalText", fontSize=10, leading=14))
    elements = []

    elements.append(Paragraph("🎥 YouTube Sentiment Analyzer Report", styles["CenterTitle"]))
    elements.append(Spacer(1, 10))
    elements.append(Paragraph(f"Video URL: {video_url}", styles["NormalText"]))
    elements.append(Spacer(1, 15))

    positive = int((df["sentiment"] == "Positive").sum())
    negative = int((df["sentiment"] == "Negative").sum())
    neutral = int((df["sentiment"] == "Neutral").sum())
    total = max(1, positive + negative + neutral)

    summary = f"""Based on <b>{len(df)}</b> comments:<br/>
• Positive: <b>{(positive/total)*100:.1f}%</b><br/>
• Neutral: <b>{(neutral/total)*100:.1f}%</b><br/>
• Negative: <b>{(negative/total)*100:.1f}%</b><br/>"""
    elements.append(Paragraph("📌 Executive Summary", styles["SectionTitle"]))
    elements.append(Paragraph(summary, styles["NormalText"]))
    elements.append(Spacer(1, 10))

    data = [
        ["Metric", "Value"],
        ["Total Comments", str(len(df))],
        ["Positive", str(positive)],
        ["Neutral", str(neutral)],
        ["Negative", str(negative)],
        ["Average Likes", f"{df['likeCount'].mean():.2f}"],
    ]
    table = Table(data, colWidths=[2.5*inch, 2.5*inch])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a73e8")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
    ]))
    elements.append(Paragraph("📊 Overview Metrics", styles["SectionTitle"]))
    elements.append(table)
    elements.append(Spacer(1, 10))

    # Pie chart only in PDF report
    counts = [positive, neutral, negative]
    labels = ["Positive", "Neutral", "Negative"]
    colors_chart = ["#34a853", "#9aa0a6", "#ea4335"]
    fig, ax = plt.subplots()
    ax.pie(counts, labels=labels, autopct="%1.1f%%",
           colors=colors_chart, startangle=140)
    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    elements.append(Paragraph("📊 Sentiment Distribution", styles["SectionTitle"]))
    elements.append(Image(buf, width=4.5*inch, height=3.5*inch))
    elements.append(Spacer(1, 10))

    rec = "💡 Recommendations: "
    if positive > negative:
        rec += "Majority of feedback is positive. Continue with similar content to maintain engagement."
    else:
        rec += "Notable negative sentiment detected. Consider addressing recurring issues."
    elements.append(Paragraph("💡 Recommendations", styles["SectionTitle"]))
    elements.append(Paragraph(rec, styles["NormalText"]))

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#1a1a1a"))
        canvas.setLineWidth(3)
        canvas.rect(20, 20, A4[0]-40, A4[1]-40)
        footer_text = f"Generated on {datetime.datetime.now().strftime('%B %d, %Y %H:%M:%S')}"
        canvas.setFont("Helvetica", 8)
        canvas.drawCentredString(A4[0]/2.0, 30, footer_text)
        canvas.restoreState()

    doc.build(elements, onFirstPage=footer, onLaterPages=footer)
    buffer.seek(0)
    return buffer

# ---------------------- UI ----------------------
st.title("🎥 YouTube Comments Sentiment Analyzer")
st.write("Paste a YouTube video URL or ID, fetch comments with the YouTube Data API v3, and analyze sentiment with NLTK VADER (and optionally TextBlob).")

# ---------------------- Sidebar ----------------------
with st.sidebar:
    st.header("Settings")
    url_or_id = st.text_input("YouTube URL or Video ID", placeholder="https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    max_comments = st.slider("Max comments to fetch", min_value=50, max_value=2000, value=500, step=50)
    order = st.selectbox("Fetch order", options=["relevance", "time"], index=0)
    engine = st.selectbox("Sentiment engine", options=["VADER (NLTK)", "TextBlob", "Both"], index=0)
    run = st.button("Analyze")

# ---------------------- Main Logic ----------------------
api_key = get_api_key()
if not api_key:
    st.warning("No API key found. Set environment variable 'YOUTUBE_API_KEY' or add it to Streamlit secrets.")

if run:
    vid = extract_video_id(url_or_id)
    if not vid:
        st.error("Could not extract a valid video ID. Please check the URL/ID and try again.")
        st.stop()
    try:
        df = fetch_comments(vid, api_key, max_comments=max_comments, order=order)
    except Exception as e:
        st.error(f"Failed to fetch comments: {e}")
        st.stop()

    if df.empty:
        st.warning("No comments found for this video (or comments are disabled).")
        st.stop()

    df["clean_text"] = df["text"].astype(str).apply(basic_clean)
    if engine in ("VADER (NLTK)", "Both"):
        df = add_vader_sentiment(df)
    if engine in ("TextBlob", "Both"):
        df = add_textblob_sentiment(df)

    st.subheader("Overview")
    left, right = st.columns([2,1])
    with left:
        st.dataframe(df)
    with right:
        st.metric("Comments analyzed", len(df))
        if "compound" in df:
            counts = df["sentiment"].value_counts().reindex(["Positive","Neutral","Negative"]).fillna(0).astype(int)
            total = counts.sum() if counts.sum() > 0 else 1
            st.metric("Positive %", f"{(counts['Positive']/total)*100:.1f}%")
            st.metric("Negative %", f"{(counts['Negative']/total)*100:.1f}%")

    # Homepage bar chart only (no pie chart here)
    st.subheader("Sentiment Distribution")
    if "compound" in df:
        counts = df["sentiment"].value_counts().reindex(["Positive", "Neutral", "Negative"]).fillna(0).astype(int)
        fig1, ax1 = plt.subplots()
        ax1.bar(counts.index, counts.values, color=["green", "gray", "red"])
        ax1.set_title("VADER Sentiment Label Counts")
        st.pyplot(fig1)

    st.subheader("Wordclouds")
    if "sentiment" in df:
        st.markdown("### Positive Wordcloud")
        make_wordcloud(df.loc[df["sentiment"]=="Positive", "clean_text"].tolist(), "Positive")

        st.markdown("### Negative Wordcloud")
        make_wordcloud(df.loc[df["sentiment"]=="Negative", "clean_text"].tolist(), "Negative")

    st.subheader("Download Results")
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download CSV", csv, file_name="youtube_comments_sentiment.csv", mime="text/csv")

    if "compound" in df:
        pdf_buffer = generate_sentiment_report(df, url_or_id, "sentiment_report.pdf")
        st.download_button("📑 Download Professional PDF Report", data=pdf_buffer,
                           file_name="sentiment_report.pdf", mime="application/pdf")

    st.success("✅ Analysis complete!")
