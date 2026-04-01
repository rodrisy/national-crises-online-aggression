from twikit import Client
import asyncio
import json
import pandas as pd
import re
from datetime import datetime, timezone
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# ─── DATE RANGE CONFIG ────────────────────────────────────────────────────────
DATE_FROM = datetime(2025, 4, 28, tzinfo=timezone.utc)   # Start date (inclusive)
DATE_TO   = datetime(2025, 5, 10, tzinfo=timezone.utc)  # End date (inclusive)
# ──────────────────────────────────────────────────────────────────────────────


def convert_cookies(file):
    with open(file) as f:
        cookies = json.load(f)
    return {c["name"]: c["value"] for c in cookies}


def parse_tweet_date(date_str):
    """Parse tweet date string into a timezone-aware datetime."""
    return datetime.strptime(date_str, "%a %b %d %H:%M:%S %z %Y")


async def main():

    client = Client(language="en")

    cookies = convert_cookies("cookies.json")
    client.set_cookies(cookies)

    # Twitter's since: / until: operators filter by date (UTC)
    since_str = DATE_FROM.strftime("%Y-%m-%d")
    until_str = DATE_TO.strftime("%Y-%m-%d")

    query = f"from:SanchezCastejon since:{since_str} until:{until_str}"
    tweets = await client.search_tweet(query, "Latest")

    dataset = []

    print(f"Collecting tweets from {since_str} to {until_str}...")

    while tweets:
        for tweet in tweets:
            tweet_date = parse_tweet_date(tweet.created_at)

            # Secondary in-memory guard: skip tweets outside the window
            if not (DATE_FROM <= tweet_date <= DATE_TO):
                continue

            hashtag_count = len(re.findall(r"#\w+", tweet.text))
            mention_count = len(re.findall(r"@\w+", tweet.text))
            link_count    = len(re.findall(r"http\S+", tweet.text))

            dataset.append({
                "tweet_id": tweet.id,
                "date":     tweet.created_at,
                "text":     tweet.text,
                "likes":    tweet.favorite_count,
                "retweets": tweet.retweet_count,
                "replies":  tweet.reply_count,
                "views":    getattr(tweet, "view_count", None),
                "length":   len(tweet.text),
                "hashtags": hashtag_count,
                "mentions": mention_count,
                "links":    link_count,
            })

            if len(dataset) >= 1000:
                break

        if len(dataset) >= 1000:
            break

        # Paginate to the next batch
        try:
            tweets = await tweets.next()
        except Exception:
            break

    df = pd.DataFrame(dataset)

    if df.empty:
        print("No tweets found in the specified date range.")
        return

    # ── VADER SENTIMENT ───────────────────────────────────────────────────────
    analyzer = SentimentIntensityAnalyzer()

    df["sentiment"] = df["text"].apply(
        lambda x: analyzer.polarity_scores(x)["compound"]
    )

    def classify(score):
        if score > 0.05:
            return "positive"
        elif score < -0.05:
            return "negative"
        else:
            return "neutral"

    df["sentiment_label"] = df["sentiment"].apply(classify)
    # ─────────────────────────────────────────────────────────────────────────

    df.to_csv("tweets_dataset.csv", index=False)

    print("Dataset saved → tweets_dataset.csv")
    print("Tweets collected:", len(df))
    print("\nDate range covered:", df["date"].min(), "→", df["date"].max())
    print("Average likes:", df["likes"].mean())
    print("\nSentiment breakdown:")
    print(df["sentiment_label"].value_counts().to_string())


asyncio.run(main())