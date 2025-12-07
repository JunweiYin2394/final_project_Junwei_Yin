import os
import json
import requests
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from dotenv import load_dotenv


os.chdir(os.path.dirname(__file__))
load_dotenv()
API_KEY = os.getenv("DEEPSEEK_API_KEY")
API_URL = "https://api.deepseek.com/v1/chat/completions"


INPUT_FILE = "data/reddit_ai_clean.csv"
OUTPUT_FILE = "data/reddit_ai_labeled.csv"


# Prompt Builder
def build_prompt(title):
    return f"""
You are a senior financial analyst specializing in the AI semiconductor sector.

Your task is to evaluate whether the following news headline could affect NVIDIA (NVDA) stock in a positive, negative, or neutral way. 
Focus specifically on NVIDIA’s business drivers such as demand for GPUs, competition, AI model compute requirements, cloud spending, supply chain, regulation, and AI market sentiment.

Important rules:
- Consider effects on NVIDIA ONLY, not the entire AI industry.
- Competitive threats (AMD/Google/DeepSeek reducing compute cost) may be negative.
- Increased GPU demand (OpenAI/GPT releases, training demand, model upgrades) may be positive.
- Ignore irrelevant Reddit posts, memes, personal experiences, and unrelated content.
- Base your judgment only on the headline.

Headline:
"{title}"

Return ONLY a JSON object in this format:
{{
  "sentiment": "positive" | "negative" | "neutral",
  "confidence": 0.0 - 1.0,
}}
"""


# DeepSeek Classification
def classify_news(title):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
        "Accept": "application/json"
    }

    payload = {
        "model": "deepseek-chat",
        "messages": [
            {
                "role": "system",
                "content": "You are a financial sentiment analysis assistant. Return ONLY valid JSON."
            },
            {
                "role": "user",
                "content": build_prompt(title)
            }
        ],
        "temperature": 0.0
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status()

        raw_text = response.json()["choices"][0]["message"]["content"].strip()

        start = raw_text.find("{")
        end = raw_text.rfind("}") + 1

        if start == -1 or end == -1:
            return {"sentiment": "unknown", "confidence": 0}

        json_text = raw_text[start:end]
        result = json.loads(json_text)

        return {
            "sentiment": result.get("sentiment", "unknown"),
            "confidence": float(result.get("confidence", 0))
        }

    except Exception as e:
        return {
            "sentiment": "unknown",
            "confidence": 0,
            "reason": str(e)
        }



# Sentiment Labeling

def label_news(
    input_file="data/reddit_ai_clean.csv",
    output_file="data/reddit_ai_labeled.csv"
):
    df = pd.read_csv(input_file)

    sentiments = []
    confidences = []

    print(f"Total {len(df)} news to process.")

    for i, row in df.iterrows():

        if i >= 300:
            break

        title = row["title"]
        result = classify_news(title)

        sentiments.append(result.get("sentiment", "unknown"))
        confidences.append(result.get("confidence", 0))

    df = df.iloc[:len(sentiments)].copy()
    df["sentiment"] = sentiments
    df["confidence"] = confidences

    df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"Saved labeled Reddit data → {output_file}")

# Reddit Sentiment Heatmap
def reddit_sentiment_heatmap(
    reddit_path="data/reddit_ai_labeled.csv",
    stock_path="data/nvidia_stock_data.csv",
    output_path="results/reddit_sentiment_heatmap.png"
):
    print("Loading data...")

    reddit = pd.read_csv(reddit_path, on_bad_lines="skip")

    reddit["date"] = pd.to_datetime(
        reddit["date"], utc=True, errors="coerce"
    ).dt.tz_localize(None)

    reddit["sentiment_index"] = reddit["sentiment"].map({
        "positive":  1,
        "negative": -1,
        "neutral":   0
    })

    reddit = reddit.dropna(subset=["sentiment_index"])

    df_daily = (
        reddit.groupby("date")["sentiment_index"]
        .mean()
        .reset_index()
        .rename(columns={"sentiment_index": "reddit_sent"})
    )

    stock = pd.read_csv(stock_path)
    stock["Date"] = pd.to_datetime(
        stock["Date"], utc=True, errors="coerce"
    ).dt.tz_localize(None)

    stock = stock[["Date", "Close"]].rename(columns={"Date": "date"})

    df = pd.merge(stock, df_daily, on="date", how="inner")

    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df["reddit_sent"] = pd.to_numeric(df["reddit_sent"], errors="coerce")

    df["nvda_pct"] = df["Close"].pct_change() * 100
    df["reddit_sent_pct"] = df["reddit_sent"].pct_change() * 100

    df = df.dropna()

    corr = df[["nvda_pct", "reddit_sent_pct"]].corr()

    plt.figure(figsize=(6, 5))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)
    plt.title("Correlation: Reddit Sentiment vs NVIDIA Stock Returns")
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.show()

    print(f"Saved heatmap → {output_path}")

    return corr

if __name__ == "__main__":
    label_news(
        input_file="data/reddit_ai_raw.csv",
        output_file="data/reddit_ai_labeled.csv"
    )

