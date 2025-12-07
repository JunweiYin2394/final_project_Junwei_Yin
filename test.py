"""
Minimal test file for Final Project.
This file ONLY tests individual functions.
It does NOT run the full pipeline.
"""

from load_reddit import collect_reddit_ai_posts

def test_reddit_loader():
    """
    Test whether Reddit data loader runs without crashing
    and returns a non-empty DataFrame.
    """
    df = collect_reddit_ai_posts(
        queries=["NVIDIA"],
        subreddit="technology",
        max_pages=1,
        output_path="data/test_reddit.csv"
    )

    assert df is not None
    assert len(df) > 0


if __name__ == "__main__":
    test_reddit_loader()
