# Association Rule Mining on COVID-19 Tweets
### Data Mining Project — Group Presentation

---

## Agenda

1. Dataset Overview
2. Data Preprocessing
3. Methodology — Apriori & FP-Growth
4. Results — Top Association Rules
5. Interpretation
6. Conclusion

---

## 1. Dataset

| | |
|---|---|
| **Source** | Kaggle — COVID-19 Tweets (Gabriel Preda) |
| **License** | CC0 Public Domain |
| **Total tweets** | 179,108 |
| **Date range** | July–August 2020 |
| **Fields** | text, hashtags, user info, date, source |

---

## 2. Data Preprocessing (Person 1)

**Goal:** Convert tweets → transaction format (each tweet = a basket of hashtags)

**Steps:**
1. Loaded 179,108 tweets from CSV
2. Extracted hashtags from the `hashtags` column
3. Lowercased and deduplicated hashtags within each tweet
4. Removed tweets with no hashtags → **112,937 tweets kept**
5. Filtered rare hashtags (appeared in < 50 tweets)
   - Before: 37,692 unique hashtags
   - After: **401 unique hashtags**

**Output:** Boolean transaction matrix — 112,937 rows × 401 columns

---

## 3. Top 20 Most Frequent Hashtags

| Rank | Hashtag | Tweets |
|------|---------|--------|
| 1 | #covid19 | 99,912 |
| 2 | #coronavirus | 10,174 |
| 3 | #pandemic | 1,624 |
| 4 | #covid | 1,290 |
| 5 | #india | 1,171 |
| 6 | #corona | 1,161 |
| 7 | #trump | 1,091 |
| 8 | #lockdown | 961 |
| 9 | #coronaviruspandemic | 882 |
| 10 | #coronavirusupdate | 721 |

> **#covid19** dominates — appears in **88% of all tweets**

---

## 4. Methodology

### Apriori Algorithm
- Generates frequent itemsets level-by-level (breadth-first)
- Prunes itemsets below minimum support at each step
- Classic, interpretable, slower on large data

### FP-Growth Algorithm
- Builds a compressed FP-tree — no candidate generation
- Much faster on large datasets
- Produces identical rules to Apriori

### Parameters Used

| Parameter | Value | Meaning |
|-----------|-------|---------|
| Min Support | 0.5% | Appears in ≥ 565 tweets |
| Min Confidence | 10% | Rule is correct ≥ 10% of the time |
| Min Lift | 1.5 | At least 1.5× better than random |

---

## 5. Results — Frequent Itemsets

| Algorithm | Frequent Itemsets Found | Rules (after filters) |
|-----------|------------------------|-----------------------|
| Apriori | 35 | 27 |
| FP-Growth | 35 | **27 (identical)** |

Both algorithms produced the **same 27 rules** — confirming correctness.

---

## 6. Top Association Rules (by Lift)

| # | Antecedent → Consequent | Support | Confidence | Lift |
|---|------------------------|---------|------------|------|
| 1 | {coronavirusupdate, covid19} → {coronavirus, coronaviruspandemic} | 0.55% | **89.7%** | **143.1** |
| 2 | {coronavirus, coronaviruspandemic} → {coronavirusupdate, covid19} | 0.55% | 88.4% | 143.1 |
| 3 | {coronavirus, coronaviruspandemic, covid19} → {coronavirusupdate} | 0.55% | 89.8% | 140.7 |
| 4 | {coronavirusupdate} → {coronavirus, coronaviruspandemic, covid19} | 0.55% | 86.8% | 140.7 |
| 5 | {coronavirus, coronaviruspandemic} → {coronavirusupdate} | 0.56% | 89.0% | 139.4 |
| 6 | {coronavirus, coronavirusupdate} → {coronaviruspandemic, covid19} | 0.55% | **96.8%** | 131.0 |
| 7 | {coronaviruspandemic, covid19} → {coronavirus, coronavirusupdate} | 0.55% | 75.1% | 131.0 |
| 8 | {coronaviruspandemic} → {coronavirus, coronavirusupdate, covid19} | 0.55% | 71.0% | 124.9 |
| 9 | {coronavirus, coronavirusupdate, covid19} → {coronaviruspandemic} | 0.55% | **97.5%** | 124.9 |

---

## 7. Interpretation

### What does Lift = 143 mean?
> Tweets using **#coronavirusupdate** and **#covid19** are **143× more likely** to also use **#coronavirus** and **#coronaviruspandemic** than if hashtag usage were completely random.

### Key Finding: A "Pandemic Cluster"
The 4 hashtags below form a tightly co-occurring group:

```
#covid19  ←→  #coronavirus  ←→  #coronaviruspandemic  ←→  #coronavirusupdate
```

- These are used together by news/update accounts with very consistent vocabulary
- High confidence (75–97%) means this co-occurrence is not accidental
- High lift (124–143) confirms strong positive association

### Other Observations
- **#trump** appears in top 20 but does not form strong rules with health hashtags → political vs. health discourse are separate
- **#india**, **#lockdown**, **#wearamask** appear frequently but are region/behavior specific — not globally co-occurring

---

## 8. Conclusion

| | |
|---|---|
| **Dataset** | 112,937 COVID-19 tweets, 401 hashtags |
| **Algorithms** | Apriori + FP-Growth (same results — validates correctness) |
| **Rules found** | 27 meaningful rules |
| **Main finding** | A cluster of 4 update/news hashtags co-occur at extremely high rates (lift > 124) |
| **Implication** | Pandemic information spreaders use a consistent, predictable hashtag vocabulary — useful for content tracking and misinformation detection |

---

## Thank You

**Questions?**

> Dataset: `gpreda/covid19-tweets` on Kaggle (CC0)  
> Tools: Python, pandas, mlxtend (Apriori + FP-Growth)
