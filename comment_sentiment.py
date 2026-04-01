from twikit import Client
import asyncio
import json
import pandas as pd
import re
from datetime import datetime, timezone
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# ─── CONFIG ───────────────────────────────────────────────────────────────────
ACCOUNT           = "sanchezcastejon"   # account whose posts to look at
N_POSTS           = 10                  # how many recent posts to fetch comments from
MAX_COMMENTS_POST = 200                 # max comments to collect per post
DELAY_BETWEEN_REQ = 4                   # seconds between requests
MAX_RETRIES       = 5
RETRY_BASE_WAIT   = 60                  # seconds, doubles on each retry
# ──────────────────────────────────────────────────────────────────────────────

analyzer = SentimentIntensityAnalyzer()


def convert_cookies(file):
    with open(file) as f:
        cookies = json.load(f)
    return {c["name"]: c["value"] for c in cookies}


def parse_date(date_str):
    try:
        return datetime.strptime(date_str, "%a %b %d %H:%M:%S %z %Y")
    except Exception:
        return None


def sentiment_label(score):
    if score > 0.05:   return "positive"
    elif score < -0.05: return "negative"
    else:               return "neutral"


class RateLimitExceeded(Exception):
    pass


async def with_retry(coro_fn, label="request"):
    """Call an async function with exponential backoff on 429."""
    wait = RETRY_BASE_WAIT
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result = await coro_fn()
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
                print(f"\n  [!] {label} failed: {e}")
                return None
    print(f"\n  [!] Gave up on {label} after {MAX_RETRIES} retries.")
    return None


async def get_recent_posts(client, username):
    """Fetch the N most recent posts from an account."""
    print(f"Fetching recent posts from @{username}...")
    user = await with_retry(lambda: client.get_user_by_screen_name(username), "get_user")
    if user is None:
        print(f"  [!] Could not find user @{username}")
        return []

    tweets = await with_retry(lambda: client.get_user_tweets(user.id, "Tweets"), "get_tweets")
    if tweets is None:
        print(f"  [!] Could not fetch tweets for @{username}")
        return []

    posts = []
    while tweets and len(posts) < N_POSTS:
        for tweet in tweets:
            if tweet.text.startswith('RT '):  # skip retweets — they have no own reply thread
                continue
            posts.append(tweet)
            if len(posts) >= N_POSTS:
                break
        if len(posts) < N_POSTS:
            await asyncio.sleep(DELAY_BETWEEN_REQ)
            tweets = await with_retry(lambda: tweets.next(), "next_page")

    print(f"  Found {len(posts)} posts.")
    return posts


async def get_comments(client, tweet):
    """Fetch comments (replies) under a single tweet."""
    comments = []

    # Workaround for twikit get_tweet_by_id itemContent bug:
    # use search_tweet with conversation_id: to fetch replies instead
    query = f"conversation_id:{tweet.id}"
    replies = await with_retry(
        lambda: client.search_tweet(query, "Latest"),
        f"replies for tweet {tweet.id}"
    )
    if replies is None:
        return comments

    while replies:
        for reply in replies:
            # Skip the original author's own replies
            if reply.user and reply.user.screen_name.lower() == ACCOUNT.lower():
                continue

            score = analyzer.polarity_scores(reply.text)["compound"]
            hashtag_count = len(re.findall(r"#\w+", reply.text))
            mention_count = len(re.findall(r"@\w+", reply.text))
            link_count    = len(re.findall(r"http\S+", reply.text))

            comments.append({
                "post_id":         tweet.id,
                "post_date":       tweet.created_at,
                "post_text":       tweet.text[:80] + "..." if len(tweet.text) > 80 else tweet.text,
                "comment_id":      reply.id,
                "comment_date":    reply.created_at,
                "comment_author":  reply.user.screen_name if reply.user else "unknown",
                "comment_text":    reply.text,
                "likes":           reply.favorite_count,
                "retweets":        reply.retweet_count,
                "replies":         reply.reply_count,
                "length":          len(reply.text),
                "hashtags":        hashtag_count,
                "mentions":        mention_count,
                "links":           link_count,
                "sentiment":       score,
                "sentiment_label": sentiment_label(score),
            })

            if len(comments) >= MAX_COMMENTS_POST:
                return comments

        await asyncio.sleep(DELAY_BETWEEN_REQ)
        try:
            current = replies
            result = await current.next()
            replies = result if result and len(result) > 0 else None
        except Exception as e:
            if "429" in str(e):
                print(f"  [rate limit] Skipping to next post with {len(comments)} comments collected.", flush=True)
            replies = None

    return comments


def print_report(df):
    print(f"\n{'═'*60}")
    print(f"  Comment Sentiment Report — @{ACCOUNT}")
    print(f"{'═'*60}")

    total = len(df)
    if total == 0:
        print("  No comments collected.")
        return

    counts = df["sentiment_label"].value_counts()
    pos = counts.get("positive", 0)
    neu = counts.get("neutral",  0)
    neg = counts.get("negative", 0)

    print(f"  Total comments : {total}")
    print(f"  Positive       : {pos} ({pos/total*100:.1f}%)")
    print(f"  Neutral        : {neu} ({neu/total*100:.1f}%)")
    print(f"  Negative       : {neg} ({neg/total*100:.1f}%)")
    print(f"{'─'*60}")

    print(f"\n  Per-post breakdown:\n")
    print(f"  {'Post (truncated)':<45} {'Total':>5}  {'Pos':>5} {'Neu':>5} {'Neg':>5}")
    print(f"  {'─'*45} {'─'*5}  {'─'*5} {'─'*5} {'─'*5}")

    post_avg_sentiments = []

    for post_id, group in df.groupby("post_id"):
        label = group["post_text"].iloc[0][:44]
        t = len(group)
        c = group["sentiment_label"].value_counts()
        p = c.get("positive", 0)
        n = c.get("neutral",  0)
        ng = c.get("negative", 0)
        avg = group["sentiment"].mean()
        post_avg_sentiments.append(avg)
        print(f"  {label:<45} {t:>5}  {p:>5} {n:>5} {ng:>5}  avg: {avg:+.3f}")

    print(f"{chr(8212)*60}")
    mean_of_avgs = sum(post_avg_sentiments) / len(post_avg_sentiments)
    label_str = "positive" if mean_of_avgs > 0.05 else ("negative" if mean_of_avgs < -0.05 else "neutral")
    print(f"  Mean of per-post averages : {mean_of_avgs:+.3f}  ({label_str})")
    print(f"{'═'*60}\n")


async def main():
    client  = Client(language="en")
    cookies = convert_cookies("cookies.json")
    client.set_cookies(cookies)

    posts = await get_recent_posts(client, ACCOUNT)
    if not posts:
        print("No posts found. Exiting.")
        return

    all_comments = []

    for i, post in enumerate(posts, 1):
        print(f"Scraping comments for post {i}/{len(posts)}: {post.text[:60]}...", flush=True)
        comments = await get_comments(client, post)
        print(f"  → {len(comments)} comments collected.")
        all_comments.extend(comments)

        # Save partial progress
        if all_comments:
            pd.DataFrame(all_comments).to_csv("comments_partial.csv", index=False)

        await asyncio.sleep(DELAY_BETWEEN_REQ)

    if not all_comments:
        print("No comments collected.")
        return

    df = pd.DataFrame(all_comments)
    df.to_csv("comments_dataset.csv", index=False)
    print(f"\nFull dataset saved → comments_dataset.csv ({len(df)} comments)")

    print_report(df)

    # Summary per post
    summary = df.groupby("post_id").apply(lambda g: pd.Series({
        "post_text":       g["post_text"].iloc[0],
        "post_date":       g["post_date"].iloc[0],
        "total_comments":  len(g),
        "positive":        (g["sentiment_label"] == "positive").sum(),
        "positive_pct":    round((g["sentiment_label"] == "positive").mean() * 100, 1),
        "neutral":         (g["sentiment_label"] == "neutral").sum(),
        "neutral_pct":     round((g["sentiment_label"] == "neutral").mean() * 100, 1),
        "negative":        (g["sentiment_label"] == "negative").sum(),
        "negative_pct":    round((g["sentiment_label"] == "negative").mean() * 100, 1),
        "avg_sentiment":   round(g["sentiment"].mean(), 3),
    })).reset_index()

    summary.to_csv("comments_summary.csv", index=False)
    print("Summary saved → comments_summary.csv")


asyncio.run(main())