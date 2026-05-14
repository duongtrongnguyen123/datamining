"""
Person 1: Data Cleaning & Transaction Formatting
Input:  raw CSV from Kaggle COVID-19 tweets dataset
Output: transactions.csv  (one row per tweet, columns = hashtag presence as 0/1)
        transactions_list.csv (one row per tweet, comma-separated hashtags)
"""

import pandas as pd
import re
import ast
import sys
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────────────────
# Update this to whichever CSV file you downloaded from Kaggle
RAW_FILE = "covid19_tweets.csv"   # change if your file has a different name
MIN_HASHTAG_FREQ = 50             # drop hashtags that appear in fewer tweets
# ──────────────────────────────────────────────────────────────────────────────


def extract_hashtags_from_text(text: str) -> list[str]:
    """Pull #hashtags out of raw tweet text."""
    if not isinstance(text, str):
        return []
    tags = re.findall(r"#(\w+)", text.lower())
    return list(set(tags))  # deduplicate within a single tweet


def extract_hashtags_from_column(value) -> list[str]:
    """
    Some Kaggle datasets store hashtags as a stringified Python list.
    Try to parse it; fall back to regex on the raw string.
    """
    if pd.isna(value) or value == "" or value == "[]":
        return []
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return [str(t).lower().lstrip("#") for t in parsed]
    except (ValueError, SyntaxError):
        pass
    return extract_hashtags_from_text(str(value))


def load_and_detect(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    print(f"Loaded {len(df):,} rows, columns: {list(df.columns)}")
    return df


def get_hashtag_series(df: pd.DataFrame) -> pd.Series:
    """
    Try common Kaggle column names in order of preference.
    Returns a Series of lists of hashtag strings.
    """
    candidates = ["hashtags", "Hashtags", "entities", "text", "full_text", "tweet"]
    for col in candidates:
        if col in df.columns:
            print(f"  Using column '{col}' for hashtag extraction")
            if col in ("text", "full_text", "tweet"):
                return df[col].apply(extract_hashtags_from_text)
            else:
                return df[col].apply(extract_hashtags_from_column)
    raise ValueError(f"Cannot find a usable column. Available: {list(df.columns)}")


def build_transactions(df: pd.DataFrame, hashtag_series: pd.Series) -> pd.DataFrame:
    df = df.copy()
    df["_hashtags"] = hashtag_series

    # Drop tweets with no hashtags
    df = df[df["_hashtags"].map(len) > 0].reset_index(drop=True)
    print(f"Tweets with ≥1 hashtag: {len(df):,}")

    # Count global hashtag frequency
    from collections import Counter
    freq: Counter = Counter()
    for tags in df["_hashtags"]:
        freq.update(tags)

    # Keep only frequent hashtags
    kept = {tag for tag, cnt in freq.items() if cnt >= MIN_HASHTAG_FREQ}
    print(f"Unique hashtags before filter : {len(freq):,}")
    print(f"Unique hashtags after  filter : {len(kept):,}  (min freq={MIN_HASHTAG_FREQ})")

    # Filter each tweet's list
    df["_hashtags"] = df["_hashtags"].apply(lambda tags: [t for t in tags if t in kept])
    df = df[df["_hashtags"].map(len) > 0].reset_index(drop=True)
    print(f"Tweets remaining after filter : {len(df):,}")

    return df, sorted(kept)


def save_outputs(df: pd.DataFrame, all_tags: list[str]):
    # ── Format 1: one-hot / boolean matrix (for mlxtend TransactionEncoder) ──
    # Each column = one hashtag, value = True/False
    rows = []
    for tags in df["_hashtags"]:
        tag_set = set(tags)
        rows.append({t: (t in tag_set) for t in all_tags})

    onehot = pd.DataFrame(rows, columns=all_tags)
    onehot.to_csv("transactions_onehot.csv", index=False)
    print(f"Saved transactions_onehot.csv  ({onehot.shape[0]} rows × {onehot.shape[1]} cols)")

    # ── Format 2: list format (human-readable, also accepted by mlxtend) ──
    list_df = pd.DataFrame({"hashtags": df["_hashtags"].apply(lambda x: ",".join(x))})
    list_df.to_csv("transactions_list.csv", index=False)
    print(f"Saved transactions_list.csv    ({len(list_df)} rows)")

    # ── Quick stats ──
    print("\nTop 20 most frequent hashtags:")
    freq = onehot.sum().sort_values(ascending=False).head(20)
    print(freq.to_string())


def main():
    raw = RAW_FILE
    if not Path(raw).exists():
        # Try to auto-detect any CSV in the current directory
        csvs = list(Path(".").glob("*.csv"))
        if not csvs:
            sys.exit(
                "No CSV found. Download the Kaggle dataset into this folder "
                "and update RAW_FILE at the top of this script."
            )
        raw = str(csvs[0])
        print(f"Auto-detected: {raw}")

    df = load_and_detect(raw)
    hashtag_series = get_hashtag_series(df)
    df, all_tags = build_transactions(df, hashtag_series)
    save_outputs(df, all_tags)
    print("\nDone! Hand transactions_onehot.csv to Person 2.")


if __name__ == "__main__":
    main()
