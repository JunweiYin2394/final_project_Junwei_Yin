import requests
import pandas as pd
import time
from datetime import datetime, timezone


def fetch_reddit_subreddit(query, subreddit, max_pages=10):
    """
    Fetch Reddit posts from a specific subreddit using the search API.
    """

    results = []
    after = None

    for page in range(max_pages):
        base = f"https://www.reddit.com/r/{subreddit}/search.json"
        url = f"{base}?q={query}&restrict_sr=on&sort=new&limit=100"

        if after:
            url += f"&after={after}"

        headers = {"User-Agent": "Mozilla/5.0"}

        r = requests.get(url, headers=headers)
        if r.status_code != 200:
            print("Error:", r.status_code)
            break

        data = r.json().get("data", {})
        children = data.get("children", [])
        if not children:
            break

        for item in children:
            post = item["data"]

            timestamp = post.get("created_utc")
            title = post.get("title", "")

            if timestamp is None or title == "":
                continue

            date_str = datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime("%Y-%m-%d")

            results.append({
                "date": date_str,
                "title": title
            })

        after = data.get("after")
        if not after:
            break

        time.sleep(0.3)

    return pd.DataFrame(results)

def collect_reddit_ai_posts(
    queries=["NVIDIA", "ChatGPT", "DeepSeek"],
    subreddit="technology",
    max_pages=10,
    output_path="data/reddit_ai_clean.csv"
):
    """
    Collect Reddit posts for multiple AI-related keywords and save as a clean CSV file.
    """

    all_dfs = []

    for q in queries:
        df_part = fetch_reddit_subreddit(
            query=q,
            subreddit=subreddit,
            max_pages=max_pages
        )
        all_dfs.append(df_part)

    if not all_dfs:
        print("No Reddit data collected.")
        return pd.DataFrame()

    df = pd.concat(all_dfs, ignore_index=True).drop_duplicates()
    df.to_csv(output_path, index=False)

    print(f"Reddit data saved to → {output_path}")
    return df

