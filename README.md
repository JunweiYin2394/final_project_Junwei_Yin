# Final Project - AI News, Reddit Sentiment, and NVIDIA Stock Analysis

## Introduction
This project analyzes the relationship between AI-related public attention and NVIDIA (NVDA) stock price movements. Specifically, it integrates multiple real-world data sources including Google Trends, Reddit discussions, and financial market data. The project also applies large language models (DeepSeek API) to perform sentiment analysis on Reddit posts and examines whether short-term events and sentiment shifts have an impact on NVIDIA stock returns.

The full pipeline is fully automated and can be reproduced by running a single `main.py` script.

## Data Sources
| Data Type | Source | Description |
|----------|--------|-------------|
| Stock Price | Yahoo Finance | Daily NVIDIA (NVDA) close prices |
| Google Trends | PyTrends API | Search interest for keywords such as "AI", "ChatGPT", and "DeepSeek" |
| Reddit | Reddit API | AI-related posts collected from technology-related subreddits |
| Sentiment Labels | DeepSeek API | LLM-based sentiment classification of Reddit post titles |

All datasets are fetched automatically through API calls. No raw data files are included in this repository.

## Analysis
The analysis includes the following components:
1.Correlation analysis between Google Trends search interest and NVIDIA stock price.
2.Heatmap visualization of pairwise correlations.
3.Lagged correlation analysis to explore delayed market responses.
4.Reddit AI post sentiment classification using DeepSeek large language model.
5.Event study analysis around major AI-related milestones such as:
GPT-4 Release,
GPT Store Launch,
DeepSeek Release

The analysis focuses on identifying both long-term structural relationships and short-term event-driven effects.

## Summary of Results
From 2023 to early 2025, both AI-related search interest and NVIDIA-related search interest exhibit strong long-term upward trends. The timing of peaks and dips in Google Trends data is highly synchronized, indicating that public attention toward NVIDIA generally moves together with overall AI popularity rather than independently.

However, correlation analysis shows that same-day correlations between AI news volume, Google Trends attention, and NVIDIA’s daily stock returns are extremely weak (mostly between –0.1 and 0.2). AI-related news categories are strongly correlated with each other, but none of them show meaningful correlation with NVIDIA’s short-term returns. 

Lagged correlation analysis (1–14 days) further confirms this result. After shifting AI news forward in time, correlations with NVIDIA’s returns remain close to zero (around –0.05 to 0.05), and no consistent lead–lag relationship is observed. This suggests that even delayed market reactions to AI news are weak and unreliable.

Event-study analysis around three major AI-related milestones (GPT-4 release, GPT Store launch, and DeepSeek release) shows that public attention responds immediately and sharply to these announcements. However, NVIDIA’s stock price reactions remain mixed and inconsistent—showing small increases after GPT-4, relatively steady movement around GPT Store, and a temporary drop after the DeepSeek release.

To further test whether sentiment explains NVIDIA’s returns, approximately 90 days of Reddit posts related to AI were collected and classified using the DeepSeek API. The resulting sentiment-based correlation with NVIDIA’s daily returns remains extremely weak (approximately –0.03), indicating that even sentiment-rich community discussions do not meaningfully explain short-term price movements.

Overall, the results suggest that AI-related hype clearly generates strong public attention “windows” for NVIDIA, but NVIDIA’s stock performance is driven primarily by broader market fundamentals and long-term industry trends rather than short-term AI news or sentiment alone.


## How to Run
This project is fully reproducible by running the main pipeline script `main.py`. After cloning the repository and installing all required Python dependencies from `requirements.txt`, the entire workflow—including data collection, data processing, analysis, and figure generation—can be executed automatically through a single command.

This project requires one external API key: the **DeepSeek API**, which is used exclusively for Reddit sentiment classification. Users must create a local `.env` file based on the provided `.env.example` template and place their own DeepSeek API key there. During development, the free DeepSeek quota was insufficient to process all sentiment requests, so a small paid balance was used. The total cost was only a few cents. Due to API speed limits under free or low-tier usage, the Reddit sentiment analysis step is the slowest part of the entire pipeline and may take approximately **8 to 12 minutes** to complete for about 600 news titles.
