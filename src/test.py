from load_api import get_nvidia_stock, get_google_trends_data, load_gdelt_news

def test_api_calls():
    print("\n=== Testing NVIDIA Stock API ===")
    df_stock = get_nvidia_stock()
    print("Fetched stock rows:", len(df_stock))
    print(df_stock.head())

    print("\n=== Testing Google Trends API ===")
    df_trend = get_google_trends_data()
    print("Fetched trends rows:", len(df_trend))
    print(df_trend.head())

    print("\n=== Testing GDELT News API ===")
    news_dict = load_gdelt_news()
    for kw, df in news_dict.items():
        print(f"{kw}: {len(df)} rows")

if __name__ == "__main__":
    test_api_calls()
