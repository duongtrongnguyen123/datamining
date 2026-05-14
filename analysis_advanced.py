"""
Advanced Analysis: COVID-19 Hashtag Mining
Covers:
  1. Rule mining stratified by verified vs normal users
  2. Temporal trend analysis (week-by-week hashtag frequency)
  3. Co-occurrence heatmap of top hashtags
  4. Hashtag co-occurrence network (communities + centrality)
  5. Platform (source) breakdown of hashtag usage
  6. Influence-weighted analysis (follower-count-weighted support)
  7. Closed & maximal itemset summary
"""

import pandas as pd
import numpy as np
import ast, re, warnings
from collections import Counter
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

warnings.filterwarnings("ignore")

# ── helpers ───────────────────────────────────────────────────────────────────

def parse_hashtags(val):
    if pd.isna(val) or val in ("", "[]"):
        return []
    try:
        p = ast.literal_eval(val)
        if isinstance(p, list):
            return [str(t).lower().lstrip("#") for t in p]
    except Exception:
        pass
    return list(set(re.findall(r"#(\w+)", str(val).lower())))


def load_raw():
    df = pd.read_csv("covid19_tweets.csv", parse_dates=["date"])
    df["tags"] = df["hashtags"].apply(parse_hashtags)
    df["n_tags"] = df["tags"].apply(len)
    df = df[df["n_tags"] > 0].copy()
    return df


def keep_top(df, n=40, min_freq=50):
    freq = Counter(t for tags in df["tags"] for t in tags)
    top = {t for t, c in freq.most_common(n) if c >= min_freq}
    df = df.copy()
    df["tags"] = df["tags"].apply(lambda ts: [t for t in ts if t in top])
    df = df[df["tags"].map(len) > 0].copy()
    return df, top, freq


def build_onehot(df, cols):
    rows = [{t: (t in set(ts)) for t in cols} for ts in df["tags"]]
    return pd.DataFrame(rows, columns=sorted(cols), dtype=bool)


def mine_rules(oh, min_sup=0.005, min_conf=0.10, min_lift=1.5, label=""):
    from mlxtend.frequent_patterns import fpgrowth, association_rules
    freq = fpgrowth(oh, min_support=min_sup, use_colnames=True)
    if freq.empty:
        print(f"  [{label}] no frequent itemsets at sup={min_sup}")
        return pd.DataFrame()
    rules = association_rules(freq, metric="confidence", min_threshold=min_conf)
    rules = rules[rules["lift"] >= min_lift]
    rules["antecedents"] = rules["antecedents"].apply(lambda x: ", ".join(sorted(x)))
    rules["consequents"] = rules["consequents"].apply(lambda x: ", ".join(sorted(x)))
    for c in ["support","confidence","lift","leverage","conviction"]:
        if c in rules: rules[c] = rules[c].round(4)
    rules = rules.sort_values("lift", ascending=False).reset_index(drop=True)
    print(f"  [{label}] {len(freq)} itemsets → {len(rules)} rules")
    return rules


# ══════════════════════════════════════════════════════════════════════════════
# 1. STRATIFIED MINING — verified vs non-verified
# ══════════════════════════════════════════════════════════════════════════════

def analysis_verified(df, top_tags):
    print("\n── 1. Stratified Mining: Verified vs Non-Verified ──")
    results = {}
    for flag, label in [(True, "verified"), (False, "non_verified")]:
        sub = df[df["user_verified"] == flag]
        sub, _, _ = keep_top(sub, n=30, min_freq=5)
        oh = build_onehot(sub, top_tags & {t for tags in sub["tags"] for t in tags})
        if oh.empty or len(oh) < 50:
            print(f"  [{label}] too few rows"); continue
        sup = 0.002 if flag else 0.005  # lower threshold for smaller verified group
        rules = mine_rules(oh, min_sup=sup, min_conf=0.10, min_lift=2.0, label=label)
        results[label] = rules
        if not rules.empty:
            rules.to_csv(f"rules_{label}.csv", index=False)

    # Compare top rules
    print("\n  Top 5 rules for VERIFIED accounts:")
    if "verified" in results and not results["verified"].empty:
        for _, r in results["verified"].head(5).iterrows():
            print(f"    {{{r['antecedents']}}} → {{{r['consequents']}}}  lift={r['lift']}")

    print("\n  Top 5 rules for NON-VERIFIED accounts:")
    if "non_verified" in results and not results["non_verified"].empty:
        for _, r in results["non_verified"].head(5).iterrows():
            print(f"    {{{r['antecedents']}}} → {{{r['consequents']}}}  lift={r['lift']}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. TEMPORAL TREND ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def analysis_temporal(df, top_tags, top_n_plot=12):
    print("\n── 2. Temporal Trend Analysis ──")
    df = df.copy()
    df["week"] = df["date"].dt.to_period("W").apply(lambda p: str(p.start_time.date()))

    # Weekly frequency per hashtag (normalised by total tweets that week)
    records = []
    for week, grp in df.groupby("week"):
        total = len(grp)
        freq = Counter(t for tags in grp["tags"] for t in tags if t in top_tags)
        for tag, cnt in freq.items():
            records.append({"week": week, "tag": tag, "pct": cnt / total * 100})
    trend = pd.DataFrame(records)
    trend.to_csv("temporal_trends.csv", index=False)

    # Pick top_n_plot tags by overall count for plotting
    overall = Counter(t for tags in df["tags"] for t in tags if t in top_tags)
    plot_tags = [t for t, _ in overall.most_common(top_n_plot) if t != "covid19"][:top_n_plot]

    fig, ax = plt.subplots(figsize=(14, 6))
    weeks = sorted(trend["week"].unique())
    cmap = matplotlib.colormaps.get_cmap("tab20").resampled(len(plot_tags))
    for i, tag in enumerate(plot_tags):
        sub = trend[trend["tag"] == tag].set_index("week").reindex(weeks, fill_value=0)
        ax.plot(weeks, sub["pct"], marker="o", markersize=3, label=f"#{tag}", color=cmap(i))

    ax.set_title("Weekly Hashtag Usage (% of tweets that week)", fontsize=13)
    ax.set_xlabel("Week")
    ax.set_ylabel("% of tweets")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig("temporal_trends.png", dpi=150)
    plt.close()
    print(f"  Saved temporal_trends.csv + temporal_trends.png  ({len(weeks)} weeks)")

    # Per-week rule count
    print("\n  Rules per week (min_sup=0.01, min_lift=1.5):")
    for week, grp in df.groupby("week"):
        sub_tags = {t for tags in grp["tags"] for t in tags} & set(plot_tags + ["covid19"])
        oh = build_onehot(grp, sub_tags)
        if len(oh) < 200: continue
        from mlxtend.frequent_patterns import fpgrowth, association_rules
        freq = fpgrowth(oh, min_support=0.01, use_colnames=True)
        if freq.empty: rules_n = 0
        else:
            r = association_rules(freq, metric="lift", min_threshold=1.5)
            rules_n = len(r)
        print(f"    {week}: {len(grp):,} tweets → {rules_n} rules")


# ══════════════════════════════════════════════════════════════════════════════
# 3. CO-OCCURRENCE HEATMAP
# ══════════════════════════════════════════════════════════════════════════════

def analysis_heatmap(df, n=25):
    print("\n── 3. Co-occurrence Heatmap ──")
    overall = Counter(t for tags in df["tags"] for t in tags)
    plot_tags = sorted([t for t, _ in overall.most_common(n)])

    # Build co-occurrence matrix (lift)
    oh = build_onehot(df, plot_tags)
    total = len(oh)
    cooc = pd.DataFrame(0.0, index=plot_tags, columns=plot_tags)
    sup = oh.mean()
    for i, a in enumerate(plot_tags):
        for b in plot_tags[i:]:
            joint = (oh[a] & oh[b]).mean()
            lift = joint / (sup[a] * sup[b]) if sup[a] * sup[b] > 0 else 0
            cooc.loc[a, b] = lift
            cooc.loc[b, a] = lift
    arr = cooc.values.copy()
    np.fill_diagonal(arr, 1.0)
    cooc = pd.DataFrame(arr, index=plot_tags, columns=plot_tags)

    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.eye(len(plot_tags), dtype=bool)
    sns.heatmap(
        cooc, ax=ax, mask=mask,
        cmap="YlOrRd", vmin=1, vmax=cooc.values[~mask].max(),
        linewidths=0.3, annot=False,
        xticklabels=[f"#{t}" for t in plot_tags],
        yticklabels=[f"#{t}" for t in plot_tags],
    )
    ax.set_title(f"Hashtag Lift (co-occurrence strength) — Top {n} hashtags", fontsize=13)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig("cooccurrence_heatmap.png", dpi=150)
    plt.close()
    cooc.to_csv("cooccurrence_lift.csv")
    print(f"  Saved cooccurrence_heatmap.png + cooccurrence_lift.csv")

    # Print strongest pairs (excluding diagonal & covid19 self-pairs)
    pairs = []
    for a in plot_tags:
        for b in plot_tags:
            if a < b:
                pairs.append((a, b, cooc.loc[a, b]))
    pairs.sort(key=lambda x: -x[2])
    print("\n  Strongest co-occurrence pairs (by lift):")
    for a, b, l in pairs[:10]:
        print(f"    #{a} + #{b}: lift={l:.2f}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. HASHTAG CO-OCCURRENCE NETWORK
# ══════════════════════════════════════════════════════════════════════════════

def analysis_network(df, n=30, min_lift=3.0):
    print("\n── 4. Hashtag Network (Communities + Centrality) ──")
    try:
        import networkx as nx
        from networkx.algorithms.community import greedy_modularity_communities
    except ImportError:
        print("  networkx not installed — skipping"); return

    overall = Counter(t for tags in df["tags"] for t in tags)
    plot_tags = [t for t, _ in overall.most_common(n)]
    oh = build_onehot(df, plot_tags)
    sup = oh.mean()
    total = len(oh)

    G = nx.Graph()
    G.add_nodes_from(plot_tags)
    for i, a in enumerate(plot_tags):
        for b in plot_tags[i+1:]:
            joint = (oh[a] & oh[b]).mean()
            lift = joint / (sup[a] * sup[b]) if sup[a] * sup[b] > 0 else 0
            if lift >= min_lift:
                G.add_edge(a, b, weight=round(lift, 2))

    print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges (lift≥{min_lift})")

    # Community detection
    communities = list(greedy_modularity_communities(G))
    community_map = {}
    for i, comm in enumerate(communities):
        for node in comm:
            community_map[node] = i
    print(f"  Communities found: {len(communities)}")
    for i, comm in enumerate(communities):
        print(f"    Community {i+1}: {', '.join(f'#{t}' for t in sorted(comm))}")

    # Centrality
    centrality = nx.betweenness_centrality(G, weight="weight")
    degree_c   = nx.degree_centrality(G)
    print("\n  Top 10 nodes by betweenness centrality (bridge hashtags):")
    for tag, c in sorted(centrality.items(), key=lambda x: -x[1])[:10]:
        print(f"    #{tag}: betweenness={c:.4f}  degree={degree_c[tag]:.4f}")

    # Draw network
    colors_list = list(mcolors.TABLEAU_COLORS.values())
    node_colors = [colors_list[community_map.get(n, 0) % len(colors_list)] for n in G.nodes()]
    node_sizes  = [300 + 3000 * degree_c[n] for n in G.nodes()]
    edge_widths = [G[u][v]["weight"] / 10 for u, v in G.edges()]

    fig, ax = plt.subplots(figsize=(14, 10))
    pos = nx.spring_layout(G, seed=42, k=2.5)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.85, ax=ax)
    nx.draw_networkx_labels(G, pos, labels={n: f"#{n}" for n in G.nodes()}, font_size=7, ax=ax)
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.4, edge_color="gray", ax=ax)
    ax.set_title(f"Hashtag Co-occurrence Network (lift≥{min_lift}) — color=community, size=degree", fontsize=12)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig("hashtag_network.png", dpi=150)
    plt.close()
    print("  Saved hashtag_network.png")

    # Save edge list
    edges_df = pd.DataFrame(
        [(u, v, d["weight"]) for u, v, d in G.edges(data=True)],
        columns=["source", "target", "lift"]
    ).sort_values("lift", ascending=False)
    edges_df.to_csv("network_edges.csv", index=False)
    print("  Saved network_edges.csv")


# ══════════════════════════════════════════════════════════════════════════════
# 5. PLATFORM (SOURCE) BREAKDOWN
# ══════════════════════════════════════════════════════════════════════════════

def analysis_platform(df):
    print("\n── 5. Platform Breakdown ──")
    def simplify_source(s):
        if not isinstance(s, str): return "Other"
        s = s.lower()
        if "iphone" in s: return "Twitter iPhone"
        if "android" in s: return "Twitter Android"
        if "web app" in s: return "Twitter Web"
        if "tweetdeck" in s: return "TweetDeck"
        if "hootsuite" in s: return "Hootsuite"
        if "ipad" in s: return "Twitter iPad"
        if "buffer" in s: return "Buffer"
        return "Other"

    df = df.copy()
    df["platform"] = df["source"].apply(simplify_source)
    overall = Counter(t for tags in df["tags"] for t in tags)
    top20 = {t for t, _ in overall.most_common(20)}

    platform_stats = []
    platforms = df["platform"].value_counts()
    print(f"  {'Platform':<20} {'Tweets':>8}  Top hashtags")
    for platform, count in platforms.items():
        sub = df[df["platform"] == platform]
        freq = Counter(t for tags in sub["tags"] for t in tags if t in top20)
        top3 = [f"#{t}" for t, _ in freq.most_common(3)]
        print(f"  {platform:<20} {count:>8,}  {', '.join(top3)}")
        for t, c in freq.items():
            platform_stats.append({"platform": platform, "hashtag": t, "count": c,
                                   "pct_in_platform": c / count * 100})

    stats_df = pd.DataFrame(platform_stats)
    stats_df.to_csv("platform_hashtag_stats.csv", index=False)

    # Heatmap: platform × top hashtags
    plot_tags = [t for t, _ in overall.most_common(15) if t != "covid19"][:15]
    pivot = stats_df[stats_df["hashtag"].isin(plot_tags)].pivot_table(
        index="platform", columns="hashtag", values="pct_in_platform", fill_value=0
    )
    fig, ax = plt.subplots(figsize=(14, 5))
    sns.heatmap(pivot, ax=ax, cmap="Blues", annot=True, fmt=".1f",
                linewidths=0.3, cbar_kws={"label": "% of tweets in platform"})
    ax.set_title("Hashtag usage rate (%) by Platform", fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.tight_layout()
    plt.savefig("platform_heatmap.png", dpi=150)
    plt.close()
    print("  Saved platform_hashtag_stats.csv + platform_heatmap.png")


# ══════════════════════════════════════════════════════════════════════════════
# 6. INFLUENCE-WEIGHTED SUPPORT
# ══════════════════════════════════════════════════════════════════════════════

def analysis_influence(df):
    print("\n── 6. Influence-Weighted Analysis ──")
    # Weight each tweet by log(followers+1) — louder voices get more weight
    df = df.copy()
    df["weight"] = np.log1p(df["user_followers"])
    total_weight = df["weight"].sum()

    overall = Counter(t for tags in df["tags"] for t in tags)
    top30 = [t for t, _ in overall.most_common(30)]

    weighted_sup = {}
    for tag in top30:
        mask = df["tags"].apply(lambda ts: tag in ts)
        weighted_sup[tag] = df.loc[mask, "weight"].sum() / total_weight

    raw_sup = {t: c / len(df) for t, c in overall.most_common(30)}

    rows = []
    for t in top30:
        rows.append({
            "hashtag": t,
            "raw_support": round(raw_sup.get(t, 0), 4),
            "influence_weighted_support": round(weighted_sup.get(t, 0), 4),
            "influence_boost": round(weighted_sup.get(t, 0) / max(raw_sup.get(t, 1e-9), 1e-9), 3),
        })
    inf_df = pd.DataFrame(rows).sort_values("influence_boost", ascending=False)
    inf_df.to_csv("influence_weighted.csv", index=False)

    print("  Top 10 hashtags BOOSTED by high-follower accounts (influence_boost > 1 = amplified):")
    for _, r in inf_df.head(10).iterrows():
        bar = "▲" if r["influence_boost"] > 1 else "▼"
        print(f"    #{r['hashtag']:<25} raw={r['raw_support']:.4f}  weighted={r['influence_weighted_support']:.4f}  boost={r['influence_boost']:.2f} {bar}")
    print("  Saved influence_weighted.csv")


# ══════════════════════════════════════════════════════════════════════════════
# 7. CLOSED & MAXIMAL ITEMSETS
# ══════════════════════════════════════════════════════════════════════════════

def analysis_closed_maximal(oh, min_sup=0.005):
    print("\n── 7. Closed & Maximal Itemsets ──")
    from mlxtend.frequent_patterns import fpgrowth

    freq = fpgrowth(oh, min_support=min_sup, use_colnames=True)
    freq["itemset_str"] = freq["itemsets"].apply(lambda x: frozenset(x))
    freq["size"] = freq["itemsets"].apply(len)

    # Closed: no superset with the same support
    sup_map = dict(zip(freq["itemset_str"], freq["support"]))
    closed = []
    for _, row in freq.iterrows():
        is_closed = True
        for _, row2 in freq[freq["size"] == row["size"] + 1].iterrows():
            if row["itemset_str"].issubset(row2["itemset_str"]):
                if abs(row2["support"] - row["support"]) < 1e-9:
                    is_closed = False; break
        if is_closed:
            closed.append(row)

    # Maximal: no frequent superset at all
    maximal = []
    for _, row in freq.iterrows():
        is_maximal = True
        for _, row2 in freq[freq["size"] == row["size"] + 1].iterrows():
            if row["itemset_str"].issubset(row2["itemset_str"]):
                is_maximal = False; break
        if is_maximal:
            maximal.append(row)

    closed_df  = pd.DataFrame(closed).drop(columns=["itemset_str"])
    maximal_df = pd.DataFrame(maximal).drop(columns=["itemset_str"])

    print(f"  Total frequent itemsets : {len(freq)}")
    print(f"  Closed itemsets         : {len(closed_df)}  ({100*len(closed_df)/len(freq):.1f}%)")
    print(f"  Maximal itemsets        : {len(maximal_df)}  ({100*len(maximal_df)/len(freq):.1f}%)")

    print("\n  Maximal itemsets (most compact complete patterns):")
    for _, row in maximal_df.sort_values("support", ascending=False).iterrows():
        items = ", ".join(f"#{t}" for t in sorted(row["itemsets"]))
        print(f"    [{items}]  support={row['support']:.4f}  size={row['size']}")

    closed_df.to_csv("closed_itemsets.csv", index=False)
    maximal_df.to_csv("maximal_itemsets.csv", index=False)
    print("  Saved closed_itemsets.csv + maximal_itemsets.csv")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("Loading data...")
    df = load_raw()
    print(f"  {len(df):,} tweets with hashtags")

    df, top_tags, freq = keep_top(df, n=40, min_freq=50)
    print(f"  Kept {len(top_tags)} hashtags, {len(df):,} tweets")

    oh = build_onehot(df, top_tags)

    analysis_verified(df, top_tags)
    analysis_temporal(df, top_tags)
    analysis_heatmap(df, n=25)
    analysis_network(df, n=30, min_lift=3.0)
    analysis_platform(df)
    analysis_influence(df)
    analysis_closed_maximal(oh, min_sup=0.005)

    print("\n\nAll outputs:")
    for f in sorted(list(Path(".").glob("*.csv")) + list(Path(".").glob("*.png"))):
        size = f.stat().st_size
        print(f"  {f.name:<40} {size/1024:.1f} KB")


if __name__ == "__main__":
    main()
