import pandas as pd
import matplotlib.pyplot as plt
import os

def ensure_result_folder():
    os.makedirs("results", exist_ok=True)

def save_fig(name):
    ensure_result_folder()
    plt.savefig(f"results/{name}.png", dpi=300, bbox_inches='tight')

def plot_dual_axis_news_vs_stock():
    # --- 保存函数 ---

    # Load stock data
    stock = pd.read_csv("data/nvidia_stock_data.csv")
    stock["Date"] = pd.to_datetime(stock["Date"])

    # Load news data
    files = {
        "AI":               "data/artificial intelligence_news.csv",
        "Machine Learning": "data/machine learning_news.csv",
        "ChatGPT":          "data/chatgpt_news.csv",
        "DeepSeek":         "data/deepseek_news.csv"
    }

    for label, path in files.items():
        print(f"Plotting dual-axis for {label}...")

        df_news = pd.read_csv(path)
        df_news["date"] = pd.to_datetime(df_news["date"], errors="coerce").dt.tz_localize(None)

        # Align date range with stock
        start_date = max(df_news["date"].min(), stock["Date"].min())
        end_date   = min(df_news["date"].max(), stock["Date"].max())

        news_cut = df_news[(df_news["date"] >= start_date) & (df_news["date"] <= end_date)]
        stock_cut = stock[(stock["Date"] >= start_date) & (stock["Date"] <= end_date)]

        # Extract news column name (value renamed earlier)
        news_col = [c for c in df_news.columns if c.endswith("_news")][0]

        # -----------------------------
        # Plot dual-axis chart
        # -----------------------------
        fig, ax1 = plt.subplots(figsize=(12, 5))

        # Left axis = news volume
        ax1.plot(news_cut["date"], news_cut[news_col], color="blue", label=f"{label} News Volume")
        ax1.set_xlabel("Date")
        ax1.set_ylabel("News Volume", color="blue")
        ax1.tick_params(axis="y", labelcolor="blue")

        # Right axis = stock price
        ax2 = ax1.twinx()
        ax2.plot(stock_cut["Date"], stock_cut["Close"], color="purple", linewidth=2, label="NVDA Close Price")
        ax2.set_ylabel("NVDA Stock Price", color="purple")
        ax2.tick_params(axis="y", labelcolor="purple")

        plt.title(f"{label} News vs NVIDIA Stock Price", fontsize=15)
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.tight_layout()

        # --- 保存图片 ---
        filename = f"news_vs_stock_{label.lower().replace(' ', '_')}"
        save_fig(filename)

        plt.show()



def plot_ai_news_vs_stock():
    # 读取数据
    news = pd.read_csv("data/artificial intelligence_news.csv")
    stock = pd.read_csv("data/nvidia_stock_data.csv")

    # 统一日期格式
    news["date"] = pd.to_datetime(news["date"], errors="coerce")

    # 去掉时区（关键！）
    if news["date"].dt.tz is not None:
        news["date"] = news["date"].dt.tz_convert(None)

    stock["Date"] = pd.to_datetime(stock["Date"], errors="coerce")

    # 保留收盘价
    stock = stock[["Date", "Close"]]

    # --- 对齐日期 ---
    start = max(news["date"].min(), stock["Date"].min())
    end   = min(news["date"].max(), stock["Date"].max())

    news = news[(news["date"] >= start) & (news["date"] <= end)]
    stock = stock[(stock["Date"] >= start) & (stock["Date"] <= end)]

    # --- 绘图 ---
    fig, ax1 = plt.subplots(figsize=(14, 6))

    # 左轴：AI 新闻量
    ax1.plot(news["date"], news["artificial intelligence_news"],
             label="AI News Volume", color="blue", linewidth=2)
    ax1.set_xlabel("Date")
    ax1.set_ylabel("AI News Volume", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")

    # 右轴：NVIDIA 股价
    ax2 = ax1.twinx()
    ax2.plot(stock["Date"], stock["Close"],
             label="NVIDIA Close Price", color="purple", linewidth=2)
    ax2.set_ylabel("NVDA Stock Price", color="purple")
    ax2.tick_params(axis="y", labelcolor="purple")

    # 标题
    plt.title("AI News Volume vs NVIDIA Stock Price")

    # 合并图例
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper left")

    plt.tight_layout()
    plt.show()


def plot_trend_vs_stock_filtered():
    # --- 日期范围 ---
    start_date = pd.to_datetime("2025-01-01")
    end_date = pd.to_datetime("2025-03-01")

    # --- 读取数据 ---
    trends = pd.read_csv("data/google_trends_ai.csv")
    stock = pd.read_csv("data/nvidia_stock_data.csv")

    # 转换日期
    trends["date"] = pd.to_datetime(trends["date"], errors="coerce")
    stock["Date"] = pd.to_datetime(stock["Date"], errors="coerce")

    # 只保留 ChatGPT 和 DeepSeek
    if "ChatGPT" not in trends.columns or "Deepseek" not in trends.columns:
        raise ValueError("Google Trends 数据中找不到 ChatGPT 或 Deepseek 列，请检查 csv")

    trends = trends[["date", "ChatGPT", "Deepseek"]]

    # --- 按日期过滤 ---
    trends = trends[(trends["date"] >= start_date) & (trends["date"] <= end_date)]
    stock = stock[(stock["Date"] >= start_date) & (stock["Date"] <= end_date)]

    # --- 准备绘图的两个函数 ---
    def draw_dual_axis(x1, y1, x2, y2, label1, label2, title, filename):
        fig, ax1 = plt.subplots(figsize=(14, 6))

        # 左轴：Google Trends
        ax1.plot(x1, y1, color="red", linewidth=2, label=label1)
        ax1.set_xlabel("Date")
        ax1.set_ylabel(label1, color="red")
        ax1.tick_params(axis="y", labelcolor="red")

        # 右轴：NVIDIA Stock
        ax2 = ax1.twinx()
        ax2.plot(x2, y2, color="purple", linewidth=2, label=label2)
        ax2.set_ylabel(label2, color="purple")
        ax2.tick_params(axis="y", labelcolor="purple")

        plt.title(title)

        # 合并图例
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

        plt.tight_layout()
        save_fig(filename)  # ← 这是关键
        plt.show()

    # --- 图 1：ChatGPT vs NVDA ---
    draw_dual_axis(
        x1=trends["date"],
        y1=trends["ChatGPT"],
        x2=stock["Date"],
        y2=stock["Close"],
        label1="Google Trends: ChatGPT",
        label2="NVDA Stock Price",
        title="ChatGPT Google Trends vs NVIDIA Stock Price (2025-01 ~ 2025-03)",
        filename="ChatGPT_vs_nvda"
    )

    # --- 图 2：DeepSeek vs NVDA ---
    draw_dual_axis(
        x1=trends["date"],
        y1=trends["Deepseek"],
        x2=stock["Date"],
        y2=stock["Close"],
        label1="Google Trends: Deepseek",
        label2="NVDA Stock Price",
        title="Deepseek Google Trends vs NVIDIA Stock Price (2025-01 ~ 2025-03)",
        filename="Deepseek_vs_nvda"
    )

