"""
02_monte_carlo_ml.py  —  LAYER 2 (Borrower-level decision)
==========================================================
Two-Stage AI Credit Decision Assistant — Stage 2.

This is the core of the dissertation. Because real bank loan-book
data is not available, we generate N = 100,000 synthetic Uzbek
borrower profiles, simulate their 5-year monthly budgets, label
defaults, and train a RandomForestClassifier that outputs a
probability of default for any new applicant.

LINK TO STAGE 1 (this is the key change vs. the old script):
    The macro scenarios are NO LONGER hard-coded as
        ("baseline", 0.0082, 0.0055, 0.55) ...
    Instead they are derived from macro_forecast.csv, produced by
    01_macro_forecast.py with Holt-Winters. The forecasted path
    anchors the "baseline" scenario; "stress" and "benign" are
    symmetric deviations around it. This is what makes the two
    layers a single product instead of two disconnected models.

The Random Forest saved here (loan_risk_model.pkl) is the SINGLE
decision-maker for the borrower in app.py. No if/else scoring.

Run AFTER Stage 1:
    pip install scikit-learn joblib numpy pandas
    python 02_monte_carlo_ml.py

Output: loan_risk_model.pkl  → place next to app.py
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import joblib

SEED = 42
N    = 100_000
np.random.seed(SEED)

MACRO_FORECAST = "macro_forecast.csv"

# ── Step 0: Load macro scenarios from Stage 1 ──────────────────
print("═" * 60)
print("Step 0: Loading macro forecast from Stage 1...")
print("═" * 60)

# Fallback values are used only if Stage 1 has not been run yet,
# so the script still runs stand-alone — but the intended workflow
# is to run 01_macro_forecast.py first.
FALLBACK_INFL_MO, FALLBACK_WAGE_MO = 0.0082, 0.0055

if os.path.exists(MACRO_FORECAST):
    mf = pd.read_csv(MACRO_FORECAST)
    base_infl_mo = float(np.clip(mf["monthly_inflation"].mean(), 0.001, 0.03))
    base_wage_mo = float(np.clip(mf["monthly_wage_growth"].mean(), 0.0, 0.02))
    macro_high_share = float((mf["predicted_macro_pressure"] == "High").mean())
    print(f"   Loaded {len(mf)} forecast months from {MACRO_FORECAST}")
    print(f"   Forecast baseline: inflation {base_infl_mo*100:.2f}%/mo, "
          f"wage growth {base_wage_mo*100:.2f}%/mo")
    print(f"   Months forecast as High macro pressure: {macro_high_share:.0%}")
else:
    base_infl_mo, base_wage_mo, macro_high_share = FALLBACK_INFL_MO, FALLBACK_WAGE_MO, 0.30
    print(f"   WARNING: {MACRO_FORECAST} not found — using fallback macro rates.")
    print("   Run 01_macro_forecast.py first to link the two layers properly.")

# Build 3 weighted scenarios ANCHORED on the forecast.
# stress = worse inflation / weaker wage growth; benign = the opposite.
# When Stage 1 forecasts a tense environment, the stress weight rises.
stress_w  = float(np.clip(0.25 + 0.30 * macro_high_share, 0.20, 0.45))
benign_w  = float(np.clip(0.20 - 0.10 * macro_high_share, 0.08, 0.20))
base_w    = round(1.0 - stress_w - benign_w, 4)

SCENARIOS = [
    ("baseline (forecast)", base_infl_mo,        base_wage_mo,        base_w),
    ("stress",              base_infl_mo * 1.45, base_wage_mo * 0.60, stress_w),
    ("benign",              base_infl_mo * 0.60, base_wage_mo * 1.30, benign_w),
]
print("   Scenarios derived from forecast:")
for name, infl, wage, w in SCENARIOS:
    print(f"     {name:<22} infl {infl*100:.2f}%/mo  wage {wage*100:.2f}%/mo  weight {w:.0%}")

# ── Step 1: Generate 100,000 synthetic borrowers ──────────────
print("\n" + "═" * 60)
print("Step 1: Generating 100,000 synthetic borrowers...")
print("═" * 60)

income = np.random.lognormal(mean=np.log(5_000_000), sigma=0.50, size=N)
income = np.clip(income, 1_500_000, 25_000_000)

amount_ratio = np.random.lognormal(mean=np.log(4), sigma=0.6, size=N)
amount_ratio = np.clip(amount_ratio, 1.0, 12.0)
loan_amount  = np.clip(income * amount_ratio, 2_000_000, 80_000_000)

term = np.random.choice([6, 12, 18, 24, 36, 48, 60, 84, 120],
                        p=[0.04, 0.12, 0.10, 0.18, 0.22, 0.14, 0.12, 0.05, 0.03],
                        size=N)
rate = np.random.uniform(18.0, 36.0, size=N)

expense_ratio = np.random.uniform(0.35, 0.85, size=N)
essential_exp = income * expense_ratio * np.random.uniform(0.75, 0.90, size=N)
flex_exp      = income * expense_ratio * np.random.uniform(0.10, 0.25, size=N)

has_exist = np.random.rand(N) < 0.40
existing  = np.where(has_exist, income * np.random.uniform(0.05, 0.25, size=N), 0.0)
savings   = income * np.random.uniform(0, 5, size=N)

unstable    = (np.random.rand(N) < 0.20).astype(float)
timing_risk = (np.random.rand(N) < 0.30).astype(float)

df = pd.DataFrame({
    "income": income, "loan_amount": loan_amount, "term_months": term,
    "interest_rate": rate, "essential_exp": essential_exp, "flex_exp": flex_exp,
    "existing_loans": existing, "savings": savings,
    "unstable": unstable, "timing_risk": timing_risk,
})

# ── Step 2: Monthly annuity payment ────────────────────────────
print("\nStep 2: Computing annuity payments...")

def annuity(p, r_pct, m):
    r = (r_pct / 100) / 12
    return p / m if r == 0 else p * r * (1 + r) ** m / ((1 + r) ** m - 1)

df["monthly_payment"] = [
    annuity(row.loan_amount, row.interest_rate, row.term_months)
    for row in df.itertuples()
]

# ── Step 3: 5-year simulation under forecast-anchored scenarios ─
print("\nStep 3: 5-year budget simulation (forecast-anchored scenarios)...")

def simulate_batch(df_in, infl_mo, wage_mo):
    n    = len(df_in)
    inc  = df_in["income"].values.copy().astype(float)
    ess  = (df_in["essential_exp"] + df_in["flex_exp"]).values.copy().astype(float)
    pmt  = df_in["monthly_payment"].values.astype(float)
    ext  = df_in["existing_loans"].values.astype(float)
    term = df_in["term_months"].values.astype(int)
    min_cash = np.full(n, np.inf)
    for m in range(1, int(term.max()) + 1):
        active = (m <= term)
        if m > 1:
            inc *= (1 + wage_mo)
            ess *= (1 + infl_mo)
        cash = inc - ess - pmt - ext
        min_cash = np.minimum(min_cash, np.where(active, cash, np.inf))
        if not active.any():
            break
    return min_cash

weighted_min_cash = np.zeros(N)
for name, infl, wage, w in SCENARIOS:
    print(f"   '{name}' (weight {w:.0%})...")
    weighted_min_cash += w * simulate_batch(df, infl, wage)
df["min_free_cash"] = weighted_min_cash

# ── Step 4: Label defaults ─────────────────────────────────────
print("\nStep 4: Labelling defaults...")
df["initial_pti"] = df["monthly_payment"] / df["income"]
df["initial_tdb"] = (df["monthly_payment"] + df["existing_loans"]) / df["income"]
df["is_default"] = np.where(
    (df["min_free_cash"] < -df["monthly_payment"]) |
    (df["initial_tdb"] > 0.65) |
    ((df["initial_pti"] > 0.55) & (df["savings"] < df["monthly_payment"])),
    1, 0
).astype(int)
default_rate = df["is_default"].mean() * 100
print(f"   Default rate: {default_rate:.1f}%  (target: 20–50%)")

# ── Step 5: Train Random Forest (the single decision-maker) ────
print("\nStep 5: Training Random Forest (200 trees)...")
FEATURES = [
    "income", "loan_amount", "term_months", "interest_rate",
    "essential_exp", "flex_exp", "existing_loans", "savings",
    "unstable", "timing_risk",
]
X, y = df[FEATURES], df["is_default"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=SEED, stratify=y
)
model = RandomForestClassifier(
    n_estimators=200, max_depth=12, min_samples_leaf=20,
    class_weight="balanced", random_state=SEED, n_jobs=-1,
)
model.fit(X_train, y_train)

# ── Step 6: Evaluate ──────────────────────────────────────────
print("\n" + "═" * 60)
print("MODEL RESULTS  (paste into dissertation Chapter 4)")
print("═" * 60)
y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_proba)
print(f"\nROC-AUC Score : {auc:.4f}")
print(f"Test samples  : {len(y_test):,}\n")
print(classification_report(y_test, y_pred, target_names=["No default", "Default"]))
cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix:")
print(f"  TN = {cm[0,0]:,}   FP = {cm[0,1]:,}")
print(f"  FN = {cm[1,0]:,}   TP = {cm[1,1]:,}")
print("\nFeature importances:")
fi = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
for feat, imp in fi.items():
    print(f"  {feat:<22} {imp:.4f}  {'█' * int(imp * 40)}")
fi.rename("importance").to_csv("feature_importance_borrower.csv")

# ── Step 7: Save ──────────────────────────────────────────────
print("\nStep 7: Saving model...")
joblib.dump(
    {"model": model, "features": FEATURES, "version": "2.0-montecarlo-linked",
     "n_simulated": N, "default_rate": round(default_rate, 2),
     "roc_auc": round(auc, 4),
     "macro_scenarios": [(s[0], s[1], s[2], s[3]) for s in SCENARIOS]},
    "loan_risk_model.pkl", compress=3,
)
print("✅  Saved: loan_risk_model.pkl")
print("    → Copy next to app.py (alongside macro_forecast.csv & data_fw.xlsx)\n")
