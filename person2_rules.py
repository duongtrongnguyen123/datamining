"""
Person 2: Association Rule Mining (Apriori + FP-Growth)
Input:  transactions_onehot.csv  (produced by Person 1)
Output: rules_apriori.csv, rules_fpgrowth.csv, top_rules.csv
        + printed summary for slides
"""

import pandas as pd
import sys
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────────────────
MIN_SUPPORT    = 0.005  # 0.5% of tweets
MIN_CONFIDENCE = 0.10   # 10% confidence
MIN_LIFT       = 1.5    # lift > 1.5 means meaningful positive correlation
TOP_N          = 10     # how many rules to highlight for slides
# ──────────────────────────────────────────────────────────────────────────────

INPUT_FILE = "transactions_onehot.csv"


def load_transactions(path: str) -> pd.DataFrame:
    # pandas reads True/False as object dtype; mlxtend needs actual bool dtype
    df = pd.read_csv(path)
    df = df.astype({col: "bool" for col in df.columns})
    print(f"Loaded {len(df):,} transactions × {len(df.columns)} items")
    return df


def run_apriori(df: pd.DataFrame) -> pd.DataFrame:
    from mlxtend.frequent_patterns import apriori, association_rules

    print(f"\n[Apriori] min_support={MIN_SUPPORT}")
    freq = apriori(df, min_support=MIN_SUPPORT, use_colnames=True, verbose=1)
    print(f"  Frequent itemsets: {len(freq)}")

    if freq.empty:
        print("  No frequent itemsets — try lowering MIN_SUPPORT")
        return pd.DataFrame()

    rules = association_rules(freq, metric="confidence", min_threshold=MIN_CONFIDENCE)
    rules = rules[rules["lift"] >= MIN_LIFT].copy()
    print(f"  Rules after confidence + lift filter: {len(rules)}")
    return rules


def run_fpgrowth(df: pd.DataFrame) -> pd.DataFrame:
    from mlxtend.frequent_patterns import fpgrowth, association_rules

    print(f"\n[FP-Growth] min_support={MIN_SUPPORT}")
    freq = fpgrowth(df, min_support=MIN_SUPPORT, use_colnames=True)
    print(f"  Frequent itemsets: {len(freq)}")

    if freq.empty:
        print("  No frequent itemsets — try lowering MIN_SUPPORT")
        return pd.DataFrame()

    rules = association_rules(freq, metric="confidence", min_threshold=MIN_CONFIDENCE)
    rules = rules[rules["lift"] >= MIN_LIFT].copy()
    print(f"  Rules after confidence + lift filter: {len(rules)}")
    return rules


def format_rules(rules: pd.DataFrame) -> pd.DataFrame:
    """Stringify frozensets and round floats for readability."""
    r = rules.copy()
    r["antecedents"] = r["antecedents"].apply(lambda x: ", ".join(sorted(x)))
    r["consequents"] = r["consequents"].apply(lambda x: ", ".join(sorted(x)))
    for col in ["support", "confidence", "lift", "leverage", "conviction"]:
        if col in r.columns:
            r[col] = r[col].round(4)
    return r.sort_values("lift", ascending=False).reset_index(drop=True)


def tune_support(df: pd.DataFrame):
    """
    Helper: print itemset counts at several support levels so you can
    pick a good threshold without trial-and-error.
    """
    from mlxtend.frequent_patterns import fpgrowth
    print("\n── Support tuning (FP-Growth itemset counts) ──")
    for sup in [0.005, 0.01, 0.02, 0.05, 0.10]:
        freq = fpgrowth(df, min_support=sup, use_colnames=True)
        print(f"  support={sup:.3f} → {len(freq):,} frequent itemsets")


def print_top_rules(rules: pd.DataFrame, label: str):
    if rules.empty:
        return
    top = rules.head(TOP_N)
    print(f"\n{'='*60}")
    print(f"Top {TOP_N} rules by Lift  [{label}]")
    print(f"{'='*60}")
    for _, row in top.iterrows():
        print(
            f"  #{int(row.name)+1:02d}  {{{row['antecedents']}}} → {{{row['consequents']}}}"
            f"\n       support={row['support']:.4f}  confidence={row['confidence']:.4f}"
            f"  lift={row['lift']:.4f}\n"
        )


def main():
    if not Path(INPUT_FILE).exists():
        sys.exit(
            f"'{INPUT_FILE}' not found. Run person1_clean.py first "
            "and make sure the output is in this directory."
        )

    df = load_transactions(INPUT_FILE)

    # Uncomment this line to see a support sweep before committing to a threshold:
    # tune_support(df)

    rules_apriori  = run_apriori(df)
    rules_fpgrowth = run_fpgrowth(df)

    if not rules_apriori.empty:
        fmt_a = format_rules(rules_apriori)
        fmt_a.to_csv("rules_apriori.csv", index=False)
        print_top_rules(fmt_a, "Apriori")
        print("Saved rules_apriori.csv")

    if not rules_fpgrowth.empty:
        fmt_fp = format_rules(rules_fpgrowth)
        fmt_fp.to_csv("rules_fpgrowth.csv", index=False)
        print_top_rules(fmt_fp, "FP-Growth")
        print("Saved rules_fpgrowth.csv")

    # Merge both and pick the best unique rules for the slides
    frames = [f for f in [
        rules_apriori if not rules_apriori.empty else None,
        rules_fpgrowth if not rules_fpgrowth.empty else None,
    ] if f is not None]

    if frames:
        combined = format_rules(pd.concat(frames).drop_duplicates(
            subset=["antecedents", "consequents"]
        ))
        top = combined.head(TOP_N)
        top.to_csv("top_rules.csv", index=False)
        print(f"\nSaved top_rules.csv ({len(top)} rules) — hand this to Person 3 for slides.")

    print("\nDone!")


if __name__ == "__main__":
    main()
