import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn as sns

def save_fig(name, result_dir="results"):
    os.makedirs(result_dir, exist_ok=True)
    plt.savefig(os.path.join(result_dir, f"{name}.png"), dpi=300)

def plot_all_trends_split(
    trends_path="data/google_trends_ai.csv",
    result_dir="results"
):
    """
    Plot Google Trends for AI, ChatGPT, DeepSeek
    """

    tr = pd.read_csv(trends_path)
    tr["date"] = pd.to_datetime(tr["date"], errors="coerce")
    tr = tr.sort_values("date")

    keywords = ["AI", "ChatGPT", "DeepSeek"]

    # Validate column names
    for k in keywords:
        if k not in tr.columns:
            raise ValueError(f"Column '{k}' not found in google_trends_ai.csv")

    # Color map
    color_map = {
        "AI": "steelblue",
        "ChatGPT": "darkred",
        "DeepSeek": "purple"
    }

    # Split point
    split_date = pd.to_datetime("2025-04-01")
    tr_before = tr[tr["date"] < split_date]

    # Plot figure
    plt.figure(figsize=(16, 7))
    for k in keywords:
        plt.plot(tr_before["date"], tr_before[k], label=k, linewidth=2, color=color_map[k])

    plt.title("Google Trends (Before 2025-04-01)\nWeekly Search Interest Across AI Topics", fontsize=18)
    plt.xlabel("Date")
    plt.ylabel("Search Interest (0–100)")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(title="Keyword", fontsize=12)
    plt.tight_layout()
    save_fig("trends_before_20250401", result_dir)
    plt.show()


def plot_nvidia_trend_before(
    trends_path="data/google_trends_ai.csv",
    result_dir="results"
):
    """
    Plot Google Trends for NVIDIA keyword before 2025-04-01.
    """

    # Convert date
    tr = pd.read_csv(trends_path)
    tr["date"] = pd.to_datetime(tr["date"], errors="coerce")
    tr = tr.sort_values("date")

    # Check column exists
    if "NVIDIA" not in tr.columns:
        raise ValueError("google_trends_ai.csv must contain a 'NVIDIA' column.")

    # Filter before April 1, 2025
    cutoff = pd.to_datetime("2025-04-01")
    before = tr[tr["date"] < cutoff]

    # Plot
    plt.figure(figsize=(16, 7))
    plt.plot(before["date"], before["NVIDIA"], color="darkgreen", linewidth=2, label="NVIDIA")

    plt.title("Google Trends - NVIDIA (Before 2025-04-01)\nWeekly Public Interest in NVIDIA", fontsize=18)
    plt.xlabel("Date")
    plt.ylabel("Search Interest (0–100)")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(title="Keyword", fontsize=12)

    plt.tight_layout()
    save_fig("nvidia_before_20250401", result_dir)
    plt.show()

def plot_smooth_ai_vs_nvidia(
    trends_path="data/google_trends_ai.csv",
    result_dir="results"
):
    """
    Plot smoothed Google Trends for AI and NVIDIA before 2025-04-01.
    """
    tr = pd.read_csv(trends_path)
    tr["date"] = pd.to_datetime(tr["date"], errors="coerce")
    tr = tr.sort_values("date")

    # Filter before 2025-04-01
    cutoff = pd.to_datetime("2025-04-01")
    df = tr[tr["date"] < cutoff].copy()

    # Check required columns
    required = ["AI", "NVIDIA"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"google_trends_ai.csv missing column: {col}")

    # Apply smoothing (4-week rolling average)
    df["AI_smooth"] = df["AI"].rolling(window=4, center=True).mean()
    df["NVIDIA_smooth"] = df["NVIDIA"].rolling(window=4, center=True).mean()

    # Plot
    plt.figure(figsize=(16, 7))
    plt.plot(df["date"], df["AI_smooth"], label="AI (Smoothed)", color="steelblue", linewidth=3)
    plt.plot(df["date"], df["NVIDIA_smooth"], label="NVIDIA (Smoothed)", color="darkgreen", linewidth=3)

    plt.title("Smoothed Google Trends (Before 2025-04-01)\nAI vs NVIDIA Public Interest Trends", fontsize=18)
    plt.xlabel("Date")
    plt.ylabel("Search Interest (0–100)")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(fontsize=12)

    plt.tight_layout()
    save_fig("smoothed_ai_nvidia", result_dir)
    plt.show()


def model_correlation_analysis(
    stock_path="data/nvidia_stock_data.csv",
    trends_path="data/google_trends_ai.csv",
    result_dir="results"
):
    """
    Model 1: Pearson Correlation Analysis
    Analyze how AI news, trend data correlate with NVIDIA stock price.
    """

    # Load stock
    stock = pd.read_csv(stock_path)
    stock["Date"] = pd.to_datetime(stock["Date"], errors="coerce")
    stock = stock[["Date", "Close"]]

    # Load news data
    ai = pd.read_csv("data/artificial_intelligence_news.csv")
    cg = pd.read_csv("data/chatgpt_news.csv")
    ds = pd.read_csv("data/deepseek_news.csv")
    ml = pd.read_csv("data/machine_learning_news.csv")

    for df in [ai, cg, ds, ml]:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    ai = ai.rename(columns={"Date": "Date", "artificial_intelligence_news": "AI_news"})
    cg = cg.rename(columns={"Date": "Date", "chatgpt_news": "ChatGPT_news"})
    ds = ds.rename(columns={"Date": "Date", "deepseek_news": "DeepSeek_news"})
    ml = ml.rename(columns={"Date": "Date", "machine_learning_news": "ML_news"})

    # Load Google Trends
    tr = pd.read_csv("data/google_trends_ai.csv")
    tr["Date"] = pd.to_datetime(tr["date"], errors="coerce")

    tr = tr.rename(columns={
        "Date": "Date",
        "AI": "Trend_AI",
        "ChatGPT": "Trend_ChatGPT",
        "DeepSeek": "Trend_DeepSeek",
        "NVIDIA": "Trend_NVIDIA"
    })

    # Put all dfs into a list
    dfs = [stock, ai, cg, ds, ml, tr]

    # FIX: unify all Date columns to remove timezone
    for d in dfs:
        d["Date"] = pd.to_datetime(d["Date"], utc=True, errors="coerce").dt.tz_localize(None)

    # Merge
    df = dfs[0]
    for d in dfs[1:]:
        df = pd.merge(df, d, on="Date", how="outer")

    df = df.sort_values("Date").reset_index(drop=True)
    df = df.dropna(subset=["Close"])

    # Fill missing values
    news_cols = ["AI_news", "ChatGPT_news", "DeepSeek_news", "ML_news"]
    trend_cols = ["Trend_AI", "Trend_ChatGPT", "Trend_DeepSeek", "Trend_NVIDIA"]

    for col in news_cols:
        df[col] = df[col].fillna(0)

    for col in trend_cols:
        df[col] = df[col].ffill()

    # Convert everything to numeric to avoid string errors
    for col in ["Close"] + news_cols + trend_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Now safe to compute percent changes
    df["NVDA Returns"] = df["Close"].pct_change() * 100
    for col in news_cols + trend_cols:
        df[col + "_pct"] = df[col].pct_change() * 100

    # Correlation matrix
    rename_mapping = {
        "NVDA Returns": "NVDA Returns",
        "AI_news_pct": "AI News (pct)",
        "ChatGPT_news_pct": "ChatGPT News (pct)",
        "DeepSeek_news_pct": "DeepSeek News (pct)",
        "ML_news_pct": "ML News (pct)",
        "Trend_AI_pct": "AI Trend (pct)",
        "Trend_ChatGPT_pct": "ChatGPT Trend (pct)",
        "Trend_DeepSeek_pct": "DeepSeek Trend (pct)",
        "Trend_NVIDIA_pct": "NVIDIA Trend (pct)"
    }

    df = df.rename(columns=rename_mapping)

    corr_cols = list(rename_mapping.values())
    corr = df[corr_cols].corr()

    # Heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Between AI News/Trends and NVIDIA Stock Returns")
    plt.tight_layout()
    save_fig("correlation_heatmap", result_dir)
    plt.show()

    return corr


def model_lagged_correlation(
    stock_path="data/nvidia_stock_data.csv",
    result_dir="results"
):
    """
        Model 2: Lagged Correlation Analysis
    """


    # Load and clean stock data
    stock = pd.read_csv("data/nvidia_stock_data.csv")
    stock["Date"] = pd.to_datetime(stock["Date"], errors="coerce")
    stock["Date"] = stock["Date"].dt.tz_localize(None)
    stock["Close"] = pd.to_numeric(stock["Close"], errors="coerce")
    stock = stock.dropna(subset=["Close"])

    # Load and clean news data
    file_map = {
        "AI_news": "data/artificial_intelligence_news.csv",
        "ChatGPT_news": "data/chatgpt_news.csv",
        "DeepSeek_news": "data/deepseek_news.csv",
        "ML_news": "data/machine_learning_news.csv"
    }

    news_dfs = {}

    for col_name, path in file_map.items():
        df = pd.read_csv(path)
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.tz_localize(None)

        news_col = df.columns[1]
        df[col_name] = pd.to_numeric(df[news_col], errors="coerce")

        news_dfs[col_name] = df[["Date", col_name]]

    # Merge stock and news data
    df = stock.copy()
    for col_name, d in news_dfs.items():
        df = df.merge(d, on="Date", how="left")

    df = df.sort_values("Date")
    df = df.ffill().bfill()

    # Compute percentage returns
    df["NVDA_pct"] = df["Close"].pct_change() * 100

    lag_cols = list(news_dfs.keys())
    for col in lag_cols:
        df[col + "_pct"] = df[col].pct_change() * 100

    df = df.dropna(subset=["NVDA_pct"])

    # Compute lagged correlations (1–14 days)
    max_lag = 14
    lag_results = {}

    for col in lag_cols:
        target = df["NVDA_pct"].values
        source = df[col + "_pct"].values

        corr_list = []
        for lag in range(1, max_lag + 1):
            source_shifted = np.roll(source, lag)
            source_shifted[:lag] = np.nan

            corr = np.corrcoef(target[lag:], source_shifted[lag:])[0, 1]
            corr_list.append(corr)

        lag_results[col] = corr_list

    # Visualization: lagged correlation heatmap
    plt.figure(figsize=(12, 6))
    sns.heatmap(
        pd.DataFrame(lag_results, index=range(1, max_lag + 1)),
        annot=False,
        cmap="coolwarm"
    )
    plt.title("Lagged Correlation (1–14 days)")
    plt.xlabel("News Variable")
    plt.ylabel("Lag (days)")
    plt.tight_layout()

    os.makedirs(result_dir, exist_ok=True)
    plt.savefig(os.path.join(result_dir, "lagged_correlation_heatmap.png"), dpi=300)
    plt.show()




