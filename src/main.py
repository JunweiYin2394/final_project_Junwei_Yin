from load_api import get_nvidia_stock, get_google_trends_data, load_gdelt_news
from analyze import plot_dual_axis_news_vs_stock, plot_ai_news_vs_stock, plot_trend_vs_stock_filtered
def main():

    #data
    df_stock = get_nvidia_stock()
    df_trends = get_google_trends_data()
    df_news = load_gdelt_news()

    #analyze
    plot_dual_axis_news_vs_stock()
    plot_ai_news_vs_stock()
    plot_trend_vs_stock_filtered()
if __name__ == "__main__":
    main()
