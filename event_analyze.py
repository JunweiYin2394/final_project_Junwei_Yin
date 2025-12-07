import os
import time
import pandas as pd
from pytrends.request import TrendReq
import matplotlib.pyplot as plt

def fetch_daily_trends(keywords, start_date, end_date, output_path, data_dir="data_event"):
    """
    Fetch daily Google Trends data for keywords within a <= 90 day window.
    Save to output_path.
    """

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    pytrend = TrendReq(hl="en-US", tz=360)

    timeframe = f"{start_date} {end_date}"

    pytrend.build_payload(
        kw_list=keywords,
        timeframe=timeframe,
        geo="US"
    )


    time.sleep(3)

    df = pytrend.interest_over_time()

    if df.empty:
        print(f"No data returned for {timeframe}")
        return None

    df = df.reset_index()
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Saved to {output_path}")

    return df

def get_event_gpt4(data_dir="data_event"):
    keywords = ["AI", "ChatGPT", "NVIDIA", "DeepSeek"]
    output = os.path.join(data_dir, "gpt4_release.csv")

    return fetch_daily_trends(
        keywords,
        start_date="2023-03-01",
        end_date="2023-04-15",
        output_path=output,
        data_dir=data_dir
    )


def get_event_gpt_store(data_dir="data_event"):
    keywords = ["AI", "ChatGPT", "NVIDIA", "DeepSeek"]
    output = os.path.join(data_dir, "gpt_store.csv")

    return fetch_daily_trends(
        keywords,
        start_date="2024-01-01",
        end_date="2024-02-15",
        output_path=output,
        data_dir=data_dir
    )

def get_event_deepseek(data_dir="data_event"):
    keywords = ["AI", "ChatGPT", "NVIDIA", "DeepSeek"]
    output = os.path.join(data_dir, "deepseek.csv")

    return fetch_daily_trends(
        keywords,
        start_date="2025-01-01",
        end_date="2025-02-15",
        output_path=output,
        data_dir=data_dir
    )

def load_nvda(stock_path="data/nvidia_stock_data.csv"):
    stock = pd.read_csv(stock_path)
    stock["Date"] = pd.to_datetime(stock["Date"])
    stock = stock.sort_values("Date")
    return stock

def plot_event_window(
    stock_df,
    trend_df,
    event_date,
    title,
    keywords=["AI", "ChatGPT", "NVIDIA", "DeepSeek"],
    result_dir="results"
):

    if "Date" not in trend_df.columns and "date" in trend_df.columns:
        trend_df = trend_df.rename(columns={"date": "Date"})

    trend_df["Date"] = pd.to_datetime(trend_df["Date"])
    event_date = pd.to_datetime(event_date)

    stock_df["Close"] = pd.to_numeric(stock_df["Close"], errors="coerce")
    stock_df["Close"] = stock_df["Close"].round(2)  # Round to 2 decimals

    stock_df = stock_df.sort_values("Date")
    trend_df = trend_df.sort_values("Date")

    df = pd.merge(stock_df, trend_df, on="Date", how="inner")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    ax1.plot(df["Date"], df["Close"], label="NVDA Close Price", color="blue")
    ax1.axvline(event_date, color="red", linestyle="--", label="Event Day")
    ax1.set_title(f"{title} — NVDA Price Reaction")
    ax1.set_ylabel("NVDA Price (USD)")
    ax1.grid(True)
    ax1.legend()

    for k in keywords:
        if k in df.columns:
            ax2.plot(df["Date"], df[k], label=k)

    ax2.axvline(event_date, color="red", linestyle="--")
    ax2.set_title(f"{title} — Google Trends (Daily)")
    ax2.set_ylabel("Search Interest (0–100)")
    ax2.grid(True)
    ax2.legend()

    plt.xlabel("Date")
    plt.tight_layout()

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    safe_title = title.replace(" ", "_").replace("—", "-")
    output_path = os.path.join(result_dir, f"{safe_title}.png")

    plt.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_gpt4_event(data_dir="data_event", result_dir="results"):
    print("\n=== Plotting GPT-4 Event ===")
    stock = load_nvda()
    trend = pd.read_csv(os.path.join(data_dir, "gpt4_release.csv"))

    plot_event_window(
        stock_df=stock,
        trend_df=trend,
        event_date="2023-03-14",
        title="GPT-4 Release (2023-03-14)",
        result_dir=result_dir
    )

def plot_gpt_store_event(data_dir="data_event", result_dir="results"):
    print("\n=== Plotting GPT Store Event ===")
    stock = load_nvda()
    trend = pd.read_csv(os.path.join(data_dir, "gpt_store.csv"))

    plot_event_window(
        stock_df=stock,
        trend_df=trend,
        event_date="2024-01-10",
        title="GPT Store Launch (2024-01-10)",
        result_dir=result_dir
    )

def plot_deepseek_event(data_dir="data_event", result_dir="results"):
    print("\n=== Plotting DeepSeek Event ===")
    stock = load_nvda()
    trend = pd.read_csv(os.path.join(data_dir, "deepseek.csv"))

    plot_event_window(
        stock_df=stock,
        trend_df=trend,
        event_date="2025-01-20",
        title="DeepSeek Release (2025-01-20)",
        result_dir=result_dir
    )
