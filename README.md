# COVID-19 Hashtag Association Rules

## Setup (everyone does this once)
```bash
pip install pandas mlxtend
```

---

## Person 1 — Data Cleaning
1. Download dataset from Kaggle: search **"COVID-19 tweets"** (pick the one by Gabriel Preda or similar)
2. Put the `.csv` file in this folder
3. Run:
   ```bash
   python person1_clean.py
   ```
4. Outputs produced:
   - `transactions_onehot.csv` — boolean matrix, one column per hashtag (give this to Person 2)
   - `transactions_list.csv`  — human-readable, one tweet per row

**Tune:** edit `MIN_HASHTAG_FREQ` at the top of the script if you want more/fewer hashtags.

---

## Person 2 — Association Rules
1. Make sure `transactions_onehot.csv` is in this folder (from Person 1)
2. Run:
   ```bash
   python person2_rules.py
   ```
3. Outputs produced:
   - `rules_apriori.csv`  — all rules via Apriori
   - `rules_fpgrowth.csv` — all rules via FP-Growth
   - `top_rules.csv`      — best 10 rules by lift (give this to Person 3)

**Tune:** edit `MIN_SUPPORT`, `MIN_CONFIDENCE`, `MIN_LIFT` at the top of the script.  
If you get 0 rules → lower `MIN_SUPPORT` (try 0.005).  
If you get thousands of rules → raise `MIN_SUPPORT` or `MIN_CONFIDENCE`.

Uncomment the `tune_support(df)` line in `main()` to scan support levels automatically.

---

## Person 3 — Slides
Use `top_rules.csv` and interpret each rule like:

> "Tweets with `#covid19` and `#pandemic` → `#lockdown`  
> (support=0.03, confidence=0.72, lift=2.1)"  
> Meaning: when people tweet about covid19 + pandemic, they are **2.1× more likely** to also use #lockdown than by chance.
