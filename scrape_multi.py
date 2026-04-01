from twikit import Client
from twikit.x_client_transaction.transaction import ClientTransaction
import asyncio
import json
import os
import pandas as pd
import re
import calendar
from datetime import datetime, timezone
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ─── PATCH: twikit's ClientTransaction JS parsing is broken since Twitter ──────
# updated their frontend bundle. We skip the init and return a no-op transaction
# ID so requests go through without the broken key extraction.
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
# ──────────────────────────────────────────────────────────────────────────────

# ─── CONFIG ───────────────────────────────────────────────────────────────────
YEAR = 2025
MAX_TWEETS_PER_MONTH = 1000

DELAY_BETWEEN_PAGES  = 2    # seconds between pagination requests
DELAY_BETWEEN_MONTHS = 30   # seconds between months for the same account
DELAY_BETWEEN_ACCS   = 60   # seconds between accounts
MAX_RETRIES          = 6
RETRY_BASE_WAIT      = 60   # doubles each retry: 60, 120, 240...

# ─── POLITICIANS ──────────────────────────────────────────────────────────────
# Add or remove entries here. Each dict needs:
#   handle   — Twitter/X username (no @)
#   label    — display name used in charts
#   category — grouping for aggregate analysis
POLITICIANS = [
    {"handle": "sanchezcastejon",  "label": "Pedro Sánchez",        "category": "President"},
    {"handle": "santi_abascal",    "label": "Santiago Abascal",     "category": "Opposition"},
    {"handle": "NunezFeijoo",      "label": "Alberto Núñez Feijóo", "category": "Opposition"},
    {"handle": "Yolanda_Diaz_",    "label": "Yolanda Díaz",         "category": "Government"},
    # {"handle": "example_handle", "label": "Example Name", "category": "News"},
]
# ──────────────────────────────────────────────────────────────────────────────

MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]

analyzer = SentimentIntensityAnalyzer()


def load_credentials(file="credentials.json"):
    with open(file) as f:
        return json.load(f)


def parse_tweet_date(date_str):
    return datetime.strptime(date_str, "%a %b %d %H:%M:%S %z %Y")


def sentiment_label(score):
    if score > 0.05:
        return "positive"
    elif score < -0.05:
        return "negative"
    else:
        return "neutral"


async def search_with_retry(client, query):
    wait = RETRY_BASE_WAIT
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result = await client.search_tweet(query, "Latest")
            if result is None or (hasattr(result, '__len__') and len(result) == 0):
                return None
            return result
        except Exception as e:
            err = str(e)
            if "429" in err:
                print(f"\n  [rate limit] Waiting {wait}s before retry {attempt}/{MAX_RETRIES}...", flush=True)
                await asyncio.sleep(wait)
                wait *= 2
            elif "404" in err:
                return None
            else:
                print(f"\n  [!] Search failed: {e}")
                return None
    print(f"\n  [!] Gave up after {MAX_RETRIES} retries.")
    return None


async def next_page_with_retry(tweets):
    wait = RETRY_BASE_WAIT
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result = await tweets.next()
            if result is None or (hasattr(result, '__len__') and len(result) == 0):
                return None
            return result
        except Exception as e:
            err = str(e)
            if "429" in err:
                print(f"\n  [rate limit] Waiting {wait}s before retry {attempt}/{MAX_RETRIES}...", flush=True)
                await asyncio.sleep(wait)
                wait *= 2
            else:
                return None
    return None


async def scrape_month(client, account, year, month):
    last_day    = calendar.monthrange(year, month)[1]
    month_start = datetime(year, month, 1,        tzinfo=timezone.utc)
    month_end   = datetime(year, month, last_day, 23, 59, 59, tzinfo=timezone.utc)

    since_str = f"{year}-{month:02d}-01"
    until_str = f"{year}-{month:02d}-{last_day}"
    query     = f"from:{account} since:{since_str} until:{until_str}"

    tweets = await search_with_retry(client, query)
    if tweets is None:
        return []

    dataset        = []
    too_old_streak = 0

    while tweets:
        for tweet in tweets:
            try:
                tweet_date = parse_tweet_date(tweet.created_at)
            except Exception:
                continue

            if tweet_date > month_end:
                continue

            if tweet_date < month_start:
                too_old_streak += 1
                if too_old_streak >= 20:
                    return dataset
                continue

            too_old_streak = 0

            hashtag_count = len(re.findall(r"#\w+", tweet.text))
            mention_count = len(re.findall(r"@\w+", tweet.text))
            link_count    = len(re.findall(r"http\S+", tweet.text))
            score         = analyzer.polarity_scores(tweet.text)["compound"]

            dataset.append({
                "tweet_id":        tweet.id,
                "month":           month,
                "month_name":      MONTHS[month - 1],
                "date":            tweet.created_at,
                "text":            tweet.text,
                "likes":           tweet.favorite_count,
                "retweets":        tweet.retweet_count,
                "replies":         tweet.reply_count,
                "views":           getattr(tweet, "view_count", None),
                "length":          len(tweet.text),
                "hashtags":        hashtag_count,
                "mentions":        mention_count,
                "links":           link_count,
                "sentiment":       score,
                "sentiment_label": sentiment_label(score),
            })

            if len(dataset) >= MAX_TWEETS_PER_MONTH:
                return dataset

        await asyncio.sleep(DELAY_BETWEEN_PAGES)
        tweets = await next_page_with_retry(tweets)

    return dataset


def save_account_results(handle, all_tweets):
    out_dir = handle
    os.makedirs(out_dir, exist_ok=True)

    df = pd.DataFrame(all_tweets)
    df.to_csv(os.path.join(out_dir, "tweets_by_month.csv"), index=False)

    # Monthly sentiment summary
    rows = []
    for month_num in range(1, 13):
        month_df = df[df["month"] == month_num]
        total    = len(month_df)
        if total == 0:
            continue
        counts = month_df["sentiment_label"].value_counts()
        rows.append({
            "month":          month_num,
            "month_name":     MONTHS[month_num - 1],
            "total":          total,
            "avg_sentiment":  round(month_df["sentiment"].mean(), 4),
            "positive":       counts.get("positive", 0),
            "positive_pct":   round(counts.get("positive", 0) / total * 100, 1),
            "neutral":        counts.get("neutral",  0),
            "neutral_pct":    round(counts.get("neutral",  0) / total * 100, 1),
            "negative":       counts.get("negative", 0),
            "negative_pct":   round(counts.get("negative", 0) / total * 100, 1),
        })

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(os.path.join(out_dir, "sentiment_summary.csv"), index=False)
    print(f"  Saved → {out_dir}/tweets_by_month.csv and sentiment_summary.csv")


async def scrape_account(client, politician):
    handle   = politician["handle"]
    label    = politician["label"]
    all_tweets = []

    print(f"\n{'─'*60}")
    print(f"  Scraping @{handle} ({label})")
    print(f"{'─'*60}")

    # Skip if already fully scraped
    existing = os.path.join(handle, "tweets_by_month.csv")
    if os.path.exists(existing):
        print(f"  [skip] {existing} already exists. Delete it to re-scrape.")
        return

    for month in range(1, 13):
        print(f"  {MONTHS[month-1]} {YEAR}...", end=" ", flush=True)
        month_data = await scrape_month(client, handle, YEAR, month)
        print(f"{len(month_data)} tweets.")
        all_tweets.extend(month_data)

        # Partial save after each month
        if all_tweets:
            os.makedirs(handle, exist_ok=True)
            pd.DataFrame(all_tweets).to_csv(
                os.path.join(handle, "tweets_partial.csv"), index=False
            )

        await asyncio.sleep(DELAY_BETWEEN_MONTHS)

    if all_tweets:
        save_account_results(handle, all_tweets)
    else:
        print(f"  [!] No tweets found for @{handle}.")


def convert_cookies(file):
    with open(file) as f:
        cookies = json.load(f)
    return {c["name"]: c["value"] for c in cookies}


async def main():
    client  = Client(language="en-US")
    cookies = convert_cookies("cookies.json")
    client.set_cookies(cookies)
    print("Cookies loaded.")

    for i, politician in enumerate(POLITICIANS):
        await scrape_account(client, politician)

        if i < len(POLITICIANS) - 1:
            print(f"\nWaiting {DELAY_BETWEEN_ACCS}s before next account...")
            await asyncio.sleep(DELAY_BETWEEN_ACCS)

    print("\nAll accounts scraped.")


asyncio.run(main())
