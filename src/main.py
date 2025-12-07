import os
from load_reddit import collect_reddit_ai_posts
from load_api import get_nvidia_stock, get_google_trends_data, load_gdelt_news
from event_analyze import get_event_gpt4, get_event_gpt_store, get_event_deepseek
from reddit_sentiment import label_news, reddit_sentiment_heatmap
from analyze import plot_all_trends_split, plot_smooth_ai_vs_nvidia, model_correlation_analysis, model_lagged_correlation, plot_nvidia_trend_before
from event_analyze import plot_gpt4_event, plot_gpt_store_event, plot_deepseek_event

os.makedirs("data", exist_ok=True)
os.makedirs("data_event", exist_ok=True)
os.makedirs("results", exist_ok=True)

def main():

    #Collect Reddit Data
    collect_reddit_ai_posts(
        output_path="data/reddit_ai_raw.csv"
    )

    #Label Reddit Sentiment (DeepSeek)
    label_news(
        input_file="data/reddit_ai_raw.csv",
        output_file="data/reddit_ai_labeled.csv"
    )

    #Download Stock & Trend Data
    get_nvidia_stock(
        output_path="data/nvidia_stock_data.csv"
    )
    get_google_trends_data(
        output_path="data/google_trends_ai.csv"
    )
    load_gdelt_news(
        output_dir="data"
    )

    #Download Event Google Trends
    get_event_gpt4(data_dir="data_event")
    get_event_gpt_store(data_dir="data_event")
    get_event_deepseek(data_dir="data_event")

    #Core Trend & Correlation Models
    plot_all_trends_split(
        trends_path="data/google_trends_ai.csv",
        result_dir="results"
    )
    plot_smooth_ai_vs_nvidia(
        trends_path="data/google_trends_ai.csv",
        result_dir="results"
    )
    plot_nvidia_trend_before(
            trends_path="data/google_trends_ai.csv",
            result_dir="results"
    )
    model_correlation_analysis(
        stock_path="data/nvidia_stock_data.csv",
        trends_path="data/google_trends_ai.csv",
        result_dir="results"
    )
    model_lagged_correlation(
        stock_path="data/nvidia_stock_data.csv",
        result_dir="results"
    )

    #Event Studies
    plot_gpt4_event(
        data_dir="data_event",
        result_dir="results"
    )
    plot_gpt_store_event(
        data_dir="data_event",
        result_dir="results"
    )
    plot_deepseek_event(
        data_dir="data_event",
        result_dir="results"
    )

    #Reddit Sentiment vs Stock
    reddit_sentiment_heatmap(
        reddit_path="data/reddit_ai_labeled.csv",
        stock_path="data/nvidia_stock_data.csv",
        output_path="results/reddit_sentiment_heatmap.png"
    )

if __name__ == "__main__":
    main()
