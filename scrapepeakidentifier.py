import asyncio
import argparse
import json
import re
from datetime import datetime, timezone

import pandas as pd
from twikit import Client
from twikit.x_client_transaction.transaction import ClientTransaction
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# --- Patch Twikit ---
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
DELAY = 1.0
MAX_PAGES = 200

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


# --- API wrappers ---
async def get_user_id(client, account):
    try:
        print(f"[user lookup] @{account}")
        user = await client.get_user_by_screen_name(account)
        return user.id
    except Exception as e:
        print(f"[error] user lookup failed: {e}")
        return None


async def get_tweets_page(client, user_id):
    retries = 0

    while retries < 5:
        try:
            return await client.get_user_tweets(
                user_id,
                tweet_type="Tweets",
                count=100
            )
        except Exception as e:
            if "429" in str(e):
                wait = 30 * (retries + 1)
                print(f"[rate limit] retry {retries+1}/5 → waiting {wait}s")
                await asyncio.sleep(wait)
                retries += 1
            else:
                raise

    print("[fail] too many rate limits (initial page)")
    return None

async def get_next_page(client, user_id, cursor):
    retries = 0

    while retries < 5:
        try:
            return await client.get_user_tweets(
                user_id,
                tweet_type="Tweets",
                count=100,
                cursor=cursor
            )
        except Exception as e:
            if "429" in str(e):
                wait = 30 * (retries + 1)
                print(f"[rate limit] retry {retries+1}/5 → waiting {wait}s")
                await asyncio.sleep(wait)
                retries += 1
            else:
                raise

    print("[fail] too many rate limits (pagination)")
    return None

# --- MAIN SCRAPER ---
async def scrape(account, year, client):

    year_start = datetime(year, 1, 1, tzinfo=timezone.utc)
    year_end = datetime(year, 12, 31, 23, 59, 59, tzinfo=timezone.utc)

    user_id = await get_user_id(client, account)
    if not user_id:
        return []

    tweets = await get_tweets_page(client, user_id)
    if not tweets:
        return []

    dataset = []
    seen_cursors = set()
    reached_target_year = False

    last_oldest = None
    stalled_pages = 0

    page = 1

    while tweets and page <= 200:

        next_cursor = getattr(tweets, "next_cursor", None)
        page_dates = []

        for tweet in tweets:
            try:
                dt = parse_date(tweet.created_at)
            except:
                continue

            page_dates.append(dt)

            if not reached_target_year:
                if dt.year == year:
                    reached_target_year = True
                else:
                    continue

            if dt > year_end:
                continue

            if dt < year_start:
                print(f"[stop] passed {year}")
                return dataset

            dataset.append(build_record(tweet, dt))

        if page_dates:
            newest = max(page_dates)
            oldest = min(page_dates)

            print(
                f"[page {page}] {newest.strftime('%Y-%m-%d')} → "
                f"{oldest.strftime('%Y-%m-%d')} | total {len(dataset)}"
            )

            # 🔥 NEW: detect if we're stuck in same date range
            if last_oldest and oldest >= last_oldest:
                stalled_pages += 1
                print(f"[stall detected] {stalled_pages}")
            else:
                stalled_pages = 0

            last_oldest = oldest

            if stalled_pages >= 3:
                print("[stop] stuck in recent tweets (Twikit limitation)")
                break

        if not next_cursor:
            print("[stop] no cursor")
            break

        if next_cursor in seen_cursors:
            print("[stop] repeated cursor")
            break

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