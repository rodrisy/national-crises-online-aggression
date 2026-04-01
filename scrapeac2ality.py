from twikit import Client
import asyncio
import json
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def convert_cookies(file):
    with open(file) as f:
        cookies = json.load(f)
    return {c["name"]: c["value"] for c in cookies}


def classify_period(date):
    if "2025-04-21" <= date[:10] <= "2025-04-27":
        return "before_blackout"
    elif "2025-04-28" <= date[:10] <= "2025-04-30":
        return "after_blackout"
    else:
        return "other"


async def main():

    client = Client(language="en")

    cookies = convert_cookies("cookies.json")
    client.set_cookies(cookies)

    query = "from:ac2ality since:2025-04-21 until:2025-05-01"

    tweets = await client.search_tweet(query, "Latest")

    dataset = []

    analyzer = SentimentIntensityAnalyzer()

    print("Collecting tweets...")

    while tweets:

        for tweet in tweets:

            date = str(tweet.created_at)

            period = classify_period(date)

            sentiment = analyzer.polarity_scores(tweet.text)["compound"]

            dataset.append({
                "date": date,
                "text": tweet.text,
                "likes": tweet.favorite_count,
                "retweets": tweet.retweet_count,
                "replies": tweet.reply_count,
                "views": getattr(tweet, "view_count", None),
                "sentiment": sentiment,
                "period": period
            })

        tweets = await tweets.next()

        if len(dataset) > 500:
            break

    df = pd.DataFrame(dataset)

    df = df[df["period"] != "other"]

    df.to_csv("ac2ality_blackout_dataset.csv", index=False)

    print("Dataset saved → ac2ality_blackout_dataset.csv")

    print("\nEngagement comparison")

    print(df.groupby("period")[["likes","retweets","replies"]].mean())

    print("\nSentiment comparison")

    print(df.groupby("period")["sentiment"].mean())


asyncio.run(main())