# 🎥 YouTube Comments Sentiment Analyzer (Streamlit)

Analyze YouTube video comments with **NLTK VADER** (and optional **TextBlob**) and visualize trends in a simple **Streamlit** web app.

---

## ✅ Features

* 🔑 Fetch comments via **YouTube Data API v3**
* 🧹 Clean text, compute sentiment (Positive / Neutral / Negative)
* 📊 Charts: label counts, score histogram, time trend
* ☁️ Wordclouds for positive/negative comments
* 💬 Table of most positive/negative comments
* 📂 Export results to CSV
* 🖥️ One-file app (`app.py`) - easy to run

---

## 🛠️ 1) Setup (Local)

### 📋 Prerequisites

* 🐍 Python 3.9+
* ☁️ A Google Cloud project with a **YouTube Data API v3** key

### ⚙️ Create a virtual env & install

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### 🔑 Provide your API key

You can pass your key in **either** of two ways:

* 🌍 Environment variable:

```bash
export YOUTUBE_API_KEY="YOUR_YOUTUBE_API_KEY"
# Windows PowerShell: setx YOUTUBE_API_KEY "YOUR_YOUTUBE_API_KEY"
```

* 📂 Streamlit secrets file (useful for local consistency):
  Create `.streamlit/secrets.toml` with

```toml
YOUTUBE_API_KEY = "YOUR_YOUTUBE_API_KEY"
```

### ▶️ Run

```bash
streamlit run app.py
```

Paste a YouTube video URL (or ID), choose number of comments to fetch, and click **Analyze**.

> ⚡ If you see an error about `vader_lexicon`, the app will auto-download it the first time.

---

## 🔑 2) Getting a YouTube Data API Key (Quick)

1. 🌐 Go to Google Cloud Console → **APIs & Services** → **Enabled APIs & services** → **+ ENABLE APIS AND SERVICES**.
2. 🔍 Search **YouTube Data API v3**, enable it.
3. 🔑 Go to **Credentials** → **+ CREATE CREDENTIALS** → **API key**.
4. 📋 Copy the key. (Optional: restrict to `youtube.googleapis.com`.)

---

## 📂 3) Project Structure

```
.
├── app.py
├── requirements.txt
└── README.md
```

---

## ⚠️ 4) Common Troubleshooting

* ❌ **`YouTube API error 403`**: Quota exceeded or API not enabled. Check Cloud Console and usage limits.
* 🚫 **No comments found**: Comments disabled or turned off for that video.
* ⚠️ **TextBlob not installed**: The app still runs with VADER. Install TextBlob if you want both.
* 🔤 **Wordcloud missing fonts on some hosts**: Most environments work out of the box; if not, install an extra font or ensure `pillow` is installed.

---

🚀 You’re ready to analyze YouTube comments! Have fun building! 🎉
