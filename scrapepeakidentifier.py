import asyncio
import argparse
import calendar
import json
import re
import sys
from datetime import datetime, timezone

import pandas as pd
from twikit import Client
from twikit.x_client_transaction.transaction import ClientTransaction
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# --- Patch Twikit transaction (prevents breakage) ---
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
DEFAULT_ACCOUNT = "example_account"
MAX_TIMELINE_PAGES = 150
TIMELINE_PAGE_SIZE = 100
DELAY = 0.2
MAX_RETRIES = 2
RETRY_WAIT = 30
MAX_STALLED_PAGES = 2

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


# --- Retry wrappers ---
async def get_user_id(client, account):
    for i in range(MAX_RETRIES):
        try:
            print(f"[user lookup] @{account}")
            user = await client.get_user_by_screen_name(account)
            return user.id
        except Exception as e:
            if "429" in str(e):
                print("Rate limit. Waiting...")
                await asyncio.sleep(RETRY_WAIT)
            else:
                raise
    return None


async def get_tweets_page(client, user_id):
    for i in range(MAX_RETRIES):
        try:
            print("[timeline] fetching tweets...")
            return await client.get_user_tweets(
                user_id,
                tweet_type="TweetsAndReplies",   # 🔑 FIXED
                count=TIMELINE_PAGE_SIZE
            )
        except Exception as e:
            if "429" in str(e):
                print("Rate limit. Waiting...")
                await asyncio.sleep(RETRY_WAIT)
            else:
                raise
    return None


async def get_next_page(client, user_id, cursor):
    for i in range(MAX_RETRIES):
        try:
            print(f"[pagination] using cursor: {cursor}")
            return await client.get_user_tweets(
                user_id,
                tweet_type="TweetsAndReplies",
                count=TIMELINE_PAGE_SIZE,
                cursor=cursor   # 🔥 THIS is the fix
            )
        except Exception as e:
            if "429" in str(e):
                print("Rate limit during pagination. Waiting...")
                await asyncio.sleep(RETRY_WAIT)
            else:
                raise
    return None


# --- Main scraping ---
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
    last_oldest_dt = None
    stalled_pages = 0

    while tweets and page <= MAX_TIMELINE_PAGES:

        page_dates = []

        for tweet in tweets:
            try:
                dt = parse_date(tweet.created_at)
            except:
                continue

            page_dates.append(dt)

            if dt > year_end:
                continue

            if dt < year_start:
                print(f"Reached tweets older than {year}. Stopping.")
                return dataset

            dataset.append(build_record(tweet, dt))

        if page_dates:
            newest_dt = max(page_dates)
            oldest_dt = min(page_dates)
            newest = newest_dt.strftime("%Y-%m-%d")
            oldest = oldest_dt.strftime("%Y-%m-%d")
            next_cursor = getattr(tweets, "next_cursor", None)

            print(f"[page {page}] cursor={next_cursor} | {newest} → {oldest} | total {len(dataset)}")
            print(f"Oldest tweet so far: {oldest}")

            if oldest_dt.year < year:
                print(f"[stop] Reached tweets older than target year {year}.")
                return dataset

            if last_oldest_dt is not None and oldest_dt >= last_oldest_dt:
                stalled_pages += 1
                print(f"[stall] Oldest tweet did not move back. stalled_pages={stalled_pages}")
            else:
                stalled_pages = 0
            last_oldest_dt = oldest_dt

            if not next_cursor:
                print("[stop] No next cursor returned.")
                return dataset

            if next_cursor in seen_cursors:
                print(f"[stop] Repeated cursor detected: {next_cursor}")
                return dataset

            if stalled_pages >= MAX_STALLED_PAGES:
                print("[stop] Pagination stalled; oldest tweet is repeating.")
                return dataset

            seen_cursors.add(next_cursor)

        await asyncio.sleep(DELAY)
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
    parser.add_argument("--account", required=True)
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
