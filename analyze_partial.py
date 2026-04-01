import pandas as pd

ACCOUNT = "SanchezCastejon"

df = pd.read_csv("comments_partial.csv")

if df.empty:
    print("No data found in comments_partial.csv")
    exit()

print(f"\n{'═'*60}")
print(f"  Comment Sentiment Report — @{ACCOUNT}")
print(f"{'═'*60}")

total = len(df)
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
print(f"  {'Post (truncated)':<45} {'Total':>5}  {'Pos':>5} {'Neu':>5} {'Neg':>5}  {'Avg':>7}")
print(f"  {'─'*45} {'─'*5}  {'─'*5} {'─'*5} {'─'*5}  {'─'*7}")

post_avgs = []
for post_id, group in df.groupby("post_id"):
    label = group["post_text"].iloc[0][:44]
    t = len(group)
    c = group["sentiment_label"].value_counts()
    p  = c.get("positive", 0)
    n  = c.get("neutral",  0)
    ng = c.get("negative", 0)
    avg = group["sentiment"].mean()
    post_avgs.append(avg)
    print(f"  {label:<45} {t:>5}  {p:>5} {n:>5} {ng:>5}  {avg:>+.3f}")

print(f"{'─'*60}")
mean_of_avgs = sum(post_avgs) / len(post_avgs)
label_str = "positive" if mean_of_avgs > 0.05 else ("negative" if mean_of_avgs < -0.05 else "neutral")
print(f"  Mean of per-post averages : {mean_of_avgs:+.3f}  ({label_str})")
print(f"{'═'*60}\n")

df.to_csv("comments_dataset.csv", index=False)
print("Full dataset saved → comments_dataset.csv")