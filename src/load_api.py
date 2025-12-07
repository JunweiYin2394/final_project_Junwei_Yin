import os
import time
import pandas as pd
import yfinance as yf
import requests
from pytrends.request import TrendReq

# Create data directory at runtime
os.makedirs("data", exist_ok=True)

def get_nvidia_stock(
    start="2023-03-01",
    end="2025-11-24",
    output_path = "data/nvidia_stock_data.csv"
):
    """Download NVIDIA stock data from Yahoo Finance API"""

    if os.path.exists(output_path):
        return pd.read_csv(output_path)

    df = yf.download("NVDA", start=start, end=end)
    df.reset_index(inplace=True)
    df.to_csv(output_path, index=False)

    return df


def get_google_trends_data(
    keywords=["AI", "ChatGPT", "NVIDIA", "DeepSeek"],
    start="2023-01-01",
    end="2025-10-16",
    country="US",
    output_path="data/google_trends_ai.csv"
):
    """Fetch daily Google Trends search data for specific keywords."""

    if os.path.exists(output_path):
        return pd.read_csv(output_path)

    pytrend = TrendReq(hl='en-US', tz=360)

    # Google Trends requires multiple short time windows
    timeframes = [
        "2023-01-01 2023-09-30",
        "2023-10-01 2024-06-30",
        "2024-07-01 2025-03-31",
        "2025-04-01 2025-10-16"
    ]

    all_trends = []
    for tf in timeframes:
        pytrend.build_payload(
            kw_list=keywords,
            timeframe=tf,
            geo=country
        )
        time.sleep(10)
        part = pytrend.interest_over_time()
        if not part.empty:
            all_trends.append(part)

    if not all_trends:
        return pd.DataFrame()

    df_trend = pd.concat(all_trends)
    df_trend.reset_index(inplace=True)
    df_trend.to_csv(output_path, index=False, encoding="utf-8-sig")

    return df_trend


def load_gdelt_news(
    keywords=["artificial intelligence", "machine learning", "chatgpt", "deepseek"],
    output_dir="data"
):
    """Fetch GDELT news volume time series for multiple keywords"""

    os.makedirs(output_dir, exist_ok=True)
    results = {}

    for kw in keywords:
        print(f"\n=== Fetching news COUNT for: {kw} ===")

        clean_kw = kw.lower().replace(" ", "_")
        output_path = os.path.join(output_dir, f"{clean_kw}_news.csv")

        # 1. Load from cache if file already exists
        if os.path.exists(output_path):
            print(f"Using cached file → {output_path}")
            df = pd.read_csv(output_path)
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            results[kw] = df
            continue

        # 2. Fetch from GDELT API
        url = f"https://api.gdeltproject.org/api/v2/doc/doc?query={kw}&mode=TimelineVolRaw&format=json"
        r = requests.get(url, timeout=30)

        if r.status_code != 200:
            print(f"Error fetching {kw}")
            continue

        js = r.json()
        timeline = js.get("timeline", [])
        if not timeline:
            print(f"No timeline for {kw}")
            continue

        data = timeline[0].get("data", [])
        if not data:
            print(f"No data for {kw}")
            continue

        df = pd.DataFrame(data)

        date_col = None
        for c in df.columns:
            if c.lower() == "date":
                date_col = c
                break

        if date_col is None:
            print(f"No date column for {kw}, skip")
            continue

        if "count" in df.columns:
            news_col = "count"
        elif "value" in df.columns:
            news_col = "value"
        elif "norm" in df.columns:
            news_col = "norm"
        else:
            print(f"No usable news field for {kw}, skip")
            continue

        df = df[[date_col, news_col]].copy()
        df.rename(columns={
            date_col: "Date",
            news_col: f"{clean_kw}_news"
        }, inplace=True)

        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

        df.to_csv(output_path, index=False)
        print(f"Saved {len(df)} rows → {output_path}")

        results[kw] = df

    return results