import asyncio
import argparse
import json
import re
from datetime import datetime, timezone

import pandas as pd
from twikit import Client
from twikit.x_client_transaction.transaction import ClientTransaction
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# --- Patch Twikit (prevents breaking) ---
async def _patched_init(self, session, headers):
    self.key = "patched"
    self.key_bytes = [0] * 50
    self.animation_key = "patched"
    self.DEFAULT_ROW_INDEX = 0
    self.DEFAULT_KEY_BYTES_INDICES = [0]


def _patched_transaction_id(self, method, path, **kwargs):
    return ""


ClientTransaction.init = _patched_init
ClientTransaction.generate_transaction_id = _patched_transaction_id


# --- Config ---
DEFAULT_YEAR = 2025
DELAY = 1  # slower to avoid rate limits

MONTHS = [
    "January","February","March","April","May","June",
    "July","August","September","October","November","December"
]

analyzer = SentimentIntensityAnalyzer()


# --- Helpers ---
def load_cookies(path):
    with open(path) as f:
        cookies = json.load(f)
    return {c["name"]: c["value"] for c in cookies if "name" in c}


def parse_date(date_str):
    return datetime.strptime(date_str, "%a %b %d %H:%M:%S %z %Y")


def sentiment_label(score):
    if score > 0.05:
        return "positive"
    if score < -0.05:
        return "negative"
    return "neutral"


def build_record(tweet, dt):
    text = tweet.text or ""
    score = analyzer.polarity_scores(text)["compound"]

    return {
        "tweet_id": tweet.id,
        "month": dt.month,
        "month_name": MONTHS[dt.month - 1],
        "date": tweet.created_at,
        "text": text,
        "likes": tweet.favorite_count,
        "retweets": tweet.retweet_count,
        "replies": tweet.reply_count,
        "views": getattr(tweet, "view_count", None),
        "length": len(text),
        "hashtags": len(re.findall(r"#\w+", text)),
        "mentions": len(re.findall(r"@\w+", text)),
        "links": len(re.findall(r"http\S+", text)),
        "sentiment": score,
        "sentiment_label": sentiment_label(score),
    }


# --- 🔥 NEW: scrape per month ---
async def scrape_month(client, account, year, month):
    start = f"{year}-{month:02d}-01"

    if month == 12:
        end = f"{year+1}-01-01"
    else:
        end = f"{year}-{month+1:02d}-01"

    query = f"from:{account} since:{start} until:{end}"

    print(f"[search] {query}")

    try:
        tweets = await client.search_tweet(query, product="Latest")
    except Exception as e:
        print(f"[error] {e}")
        return []

    results = []

    for tweet in tweets:
        try:
            dt = parse_date(tweet.created_at)
            results.append(build_record(tweet, dt))
        except:
            continue

    return results


async def get_user_id(client, account):
    try:
        print(f"[user lookup] @{account}")
        user = await client.get_user_by_screen_name(account)
        return user.id
    except Exception as e:
        print(f"[error] user lookup failed: {e}")
        return None

# --- 🔥 MAIN SCRAPER ---
async def scrape(account, year, client):

    year_start = datetime(year, 1, 1, tzinfo=timezone.utc)
    year_end = datetime(year, 12, 31, 23, 59, 59, tzinfo=timezone.utc)

    user_id = await get_user_id(client, account)
    if not user_id:
        print("User lookup failed.")
        return []

    tweets = await get_tweets_page(client, user_id)
    if not tweets:
        print("No tweets returned.")
        return []

    dataset = []
    page = 1
    seen_cursors = set()

    reached_target_year = False  # 🔥 NEW

    while tweets and page <= 200:  # 🔥 increased depth

        page_dates = []
        next_cursor = getattr(tweets, "next_cursor", None)

        for tweet in tweets:
            try:
                dt = parse_date(tweet.created_at)
            except:
                continue

            page_dates.append(dt)

            # 🔥 Ignore newer tweets until we reach target year
            if not reached_target_year:
                if dt.year == year:
                    reached_target_year = True
                else:
                    continue

            if dt > year_end:
                continue

            if dt < year_start:
                print(f"[stop] fully passed year {year}")
                return dataset

            dataset.append(build_record(tweet, dt))

        if page_dates:
            newest_dt = max(page_dates)
            oldest_dt = min(page_dates)

            print(
                f"[page {page}] {newest_dt.strftime('%Y-%m-%d')} → "
                f"{oldest_dt.strftime('%Y-%m-%d')} | total {len(dataset)}"
            )

            if not next_cursor:
                print("[stop] no cursor")
                return dataset

            if next_cursor in seen_cursors:
                print("[stop] repeated cursor")
                return dataset

            seen_cursors.add(next_cursor)

        await asyncio.sleep(0.5)
        tweets = await get_next_page(client, user_id, next_cursor)
        page += 1

    return dataset


# --- Reporting ---
def print_report(df, account, year):
    print("\n" + "="*60)
    print(f"Sentiment Report @{account} ({year})")
    print("="*60)

    for m in range(1, 13):
        sub = df[df["month"] == m]

        if sub.empty:
            print(f"{MONTHS[m-1]:<10} —")
            continue

        counts = sub["sentiment_label"].value_counts()
        total = len(sub)

        print(f"{MONTHS[m-1]:<10} {total:>4} | "
              f"+{counts.get('positive',0)} "
              f"~{counts.get('neutral',0)} "
              f"-{counts.get('negative',0)}")


# --- CLI ---
async def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--account", default="sanchezcastejon")
    parser.add_argument("--year", type=int, default=2025)
    args = parser.parse_args()

    account = args.account.lstrip("@")
    year = args.year

    print(f"Scraping @{account} for {year}")

    client = Client(language="en")
    client.set_cookies(load_cookies("cookies.json"))

    data = await scrape(account, year, client)

    if not data:
        print("No tweets found.")
        return

    df = pd.DataFrame(data)
    df.to_csv("tweets_by_month.csv", index=False)

    print("Saved → tweets_by_month.csv")

    print_report(df, account, year)


asyncio.run(main())