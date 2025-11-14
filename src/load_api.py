import os
import time
import pandas as pd
import yfinance as yf
import requests
from pytrends.request import TrendReq

os.makedirs("data", exist_ok=True)


def get_nvidia_stock(start="2023-01-01", end="2025-10-01"):
    """Download NVIDIA stock data from Yahoo Finance API"""
    output_path = "data/nvidia_stock_data.csv"
    if os.path.exists(output_path):
        return pd.read_csv(output_path)

    df = yf.download("NVDA", start=start, end=end)
    df.reset_index(inplace=True)  # ← 关键：把日期索引展开成普通列


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

    timeframes = [
        "2023-01-01 2023-09-30",
        "2023-10-01 2024-06-30",
        "2024-07-01 2025-03-31",
        "2025-04-01 2025-10-16"
    ]

    all_trends = []
    for tf in timeframes:
        pytrend.build_payload(kw_list=keywords, timeframe=tf, geo=country)
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


def load_gdelt_news(keywords = ["artificial intelligence", "machine learning", "chatgpt", "deepseek"], output_dir="data"):
    os.makedirs(output_dir, exist_ok=True)
    results = {}


    for kw in keywords:

        output_path = os.path.join(output_dir, f"{kw.lower()}_news.csv")
        if os.path.exists(output_path):
            results[kw] = pd.read_csv(output_path)
            continue

        url = f"https://api.gdeltproject.org/api/v2/doc/doc?query={kw}&mode=TimelineVolRaw&format=json"
        res = requests.get(url, timeout=30)

        if res.status_code != 200 or not res.text.strip().startswith("{"):
            print(f"Warning: invalid response for {kw}")
            continue

        try:
            json_data = res.json()
        except ValueError:
            print(f"Warning: failed to parse JSON for {kw}")
            continue

        timeline_list = json_data.get("timeline", [])
        if not timeline_list:
            print(f"No timeline data for {kw}")
            continue

        data = timeline_list[0].get("data", [])
        if not data:
            print(f"No 'data' field for {kw}")
            continue

        df = pd.DataFrame(data)
        if "date" not in df.columns:
            print(f"Warning: missing 'date' field for {kw}")
            continue

        df["date"] = pd.to_datetime(df["date"], errors="coerce")

        # rename columns
        df.rename(columns={
            "value": f"{kw.lower()}_news",
            "norm":  f"{kw.lower()}_norm"
        }, inplace=True)

        # save individual file
        df.to_csv(output_path, index=False)

        print(f"Saved → {output_path} ({len(df)} rows)")
        results[kw] = df

    return results