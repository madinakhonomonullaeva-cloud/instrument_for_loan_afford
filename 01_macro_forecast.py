"""
01_macro_forecast.py  —  LAYER 1 (Macro environment)
=====================================================
Two-Stage AI Credit Decision Assistant — Stage 1.

This script does ONE job: it analyses 15 years of monthly Uzbek
macro-financial data and forecasts the external economic environment
5 years (60 months) ahead with Holt-Winters exponential smoothing.

It also trains a small, explainable macro-pressure classifier
(Low / Medium / High) on the historical data — this classifier
describes the *environment*, NOT the individual borrower.

IMPORTANT (architecture note for the dissertation):
    This layer NEVER scores a user. There is no rule-based
    `calculate_user_repayment_pressure`, no `generate_recommendation`,
    no `if PTI >= 0.40: score += 2`. Those heuristics were removed on
    purpose: the borrower-level decision is made by the Random Forest
    in Stage 2 (02_monte_carlo_ml.py), not by hand-written rules.

Outputs (consumed by Stage 2 and by app.py):
    macro_forecast.csv   — 60-month forecast of the macro path
    macro_model.pkl      — trained macro-pressure classifier
    feature_importance_macro.csv — for Chapter 4 (macro feature table)

Run once locally:
    pip install pandas numpy scikit-learn statsmodels joblib openpyxl
    python 01_macro_forecast.py
"""

import warnings
import numpy as np
import pandas as pd
import joblib
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score

warnings.filterwarnings("ignore")

# ── Settings ──────────────────────────────────────────────────
DATA_PATH        = "data_fw.xlsx"
MAIN_SHEET       = "Final_Monthly_ML_Dataset"
FORECAST_MONTHS  = 60          # 5 years ahead
SEED             = 42

TARGET   = "repayment_pressure_level"
FEATURES = [
    "policy_rate_pct", "inflation_yoy_pct", "cpi_mom_pct",
    "nominal_wage_monthly_approx", "real_wage_growth_pct",
    "real_policy_rate_pct", "debt_burden_indicator_pct",
]
# Variables forecast directly with Holt-Winters; the rest are derived.
FORECAST_VARS = [
    "policy_rate_pct", "inflation_yoy_pct", "cpi_mom_pct",
    "nominal_wage_monthly_approx", "debt_burden_indicator_pct",
]

print("=" * 60)
print("STAGE 1 — MACRO ENVIRONMENT FORECAST")
print("=" * 60)

# ── 1. Load & clean ──────────────────────────────────────────
xls = pd.ExcelFile(DATA_PATH)
sheet = MAIN_SHEET if MAIN_SHEET in xls.sheet_names else xls.sheet_names[0]
df = pd.read_excel(DATA_PATH, sheet_name=sheet)
df.columns = df.columns.astype(str).str.strip().str.replace(" ", "_").str.replace("-", "_")

df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

for c in FEATURES:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df[TARGET] = df[TARGET].astype(str).str.strip()
df = df[df[TARGET].isin(["Low", "Medium", "High"])].copy()

print(f"Loaded sheet '{sheet}': {df.shape[0]} months "
      f"({df['date'].min().date()} → {df['date'].max().date()})")

# ── 2. Train macro-pressure classifier (describes environment) ─
ml = df[["date"] + FEATURES + [TARGET]].dropna().copy()
split = int(len(ml) * 0.8)                      # time-based split
X_tr, X_te = ml[FEATURES].iloc[:split], ml[FEATURES].iloc[split:]
y_tr, y_te = ml[TARGET].iloc[:split],   ml[TARGET].iloc[split:]

macro_clf = RandomForestClassifier(
    n_estimators=300, max_depth=5, min_samples_leaf=4, random_state=SEED
)
macro_clf.fit(X_tr, y_tr)

y_pred = macro_clf.predict(X_te)
print("\n--- Macro-pressure classifier (Low/Medium/High) ---")
print(f"Test macro-F1: {f1_score(y_te, y_pred, average='macro', zero_division=0):.3f}")
print(classification_report(y_te, y_pred, zero_division=0))

fi = (pd.Series(macro_clf.feature_importances_, index=FEATURES)
        .sort_values(ascending=False))
print("Macro feature importance (for Chapter 4):")
for f, v in fi.items():
    print(f"  {f:<30} {v:.4f}")
fi.rename("importance").to_csv("feature_importance_macro.csv")

# Refit on the full history before forecasting the future.
macro_clf.fit(ml[FEATURES], ml[TARGET])

# ── 3. Holt-Winters forecast of the macro path, 60 months ──────
def hw_forecast(series, horizon):
    series = series.astype(float).dropna()
    try:
        fit = ExponentialSmoothing(
            series, trend="add", damped_trend=True,
            seasonal=None, initialization_method="estimated"
        ).fit(optimized=True)
        return fit.forecast(horizon).values
    except Exception as e:
        print(f"   fallback (flat) for forecast: {e}")
        return np.repeat(series.iloc[-1], horizon)

base = df.set_index("date").sort_index()
future_dates = pd.date_range(
    start=base.index.max() + pd.DateOffset(months=1),
    periods=FORECAST_MONTHS, freq="MS"
)

fut = pd.DataFrame(index=future_dates)
for v in FORECAST_VARS:
    fut[v] = hw_forecast(base[v], FORECAST_MONTHS)

# Safety clipping (realistic Uzbek bounds)
fut["policy_rate_pct"]              = fut["policy_rate_pct"].clip(0, 40)
fut["inflation_yoy_pct"]            = fut["inflation_yoy_pct"].clip(0, 50)
fut["cpi_mom_pct"]                  = fut["cpi_mom_pct"].clip(-5, 10)
fut["nominal_wage_monthly_approx"]  = fut["nominal_wage_monthly_approx"].clip(lower=1)
fut["debt_burden_indicator_pct"]    = fut["debt_burden_indicator_pct"].clip(0, 100)

# Derive real wage growth & real policy rate (same formulas as history)
combined = pd.concat([base[FORECAST_VARS], fut[FORECAST_VARS]])
combined["nominal_wage_yoy_growth_pct"] = combined["nominal_wage_monthly_approx"].pct_change(12) * 100
combined["real_wage_growth_pct"] = combined["nominal_wage_yoy_growth_pct"] - combined["inflation_yoy_pct"]
combined["real_policy_rate_pct"] = combined["policy_rate_pct"] - combined["inflation_yoy_pct"]

fut = combined.loc[future_dates].copy()
fut["real_wage_growth_pct"] = fut["real_wage_growth_pct"].fillna(
    df["real_wage_growth_pct"].dropna().iloc[-1]
)

# ── 4. Classify the forecast environment month by month ────────
fut = fut.reset_index().rename(columns={"index": "date"})
fut["loan_month"] = np.arange(1, len(fut) + 1)
fut["predicted_macro_pressure"] = macro_clf.predict(fut[FEATURES])
proba = macro_clf.predict_proba(fut[FEATURES])
for i, cls in enumerate(macro_clf.classes_):
    fut[f"prob_{cls}"] = proba[:, i]

# Monthly rates that Stage 2 (Monte Carlo) will use as scenario inputs
fut["monthly_inflation"] = fut["cpi_mom_pct"] / 100
fut["monthly_wage_growth"] = (
    fut["nominal_wage_monthly_approx"].pct_change()
       .fillna((1 + fut["real_wage_growth_pct"].iloc[0] / 100
                + fut["inflation_yoy_pct"].iloc[0] / 100) ** (1 / 12) - 1)
)

# ── 5. Save artifacts ─────────────────────────────────────────
keep = ["date", "loan_month", "policy_rate_pct", "inflation_yoy_pct",
        "cpi_mom_pct", "nominal_wage_monthly_approx", "real_wage_growth_pct",
        "real_policy_rate_pct", "debt_burden_indicator_pct",
        "predicted_macro_pressure", "prob_Low", "prob_Medium", "prob_High",
        "monthly_inflation", "monthly_wage_growth"]
keep = [c for c in keep if c in fut.columns]
fut[keep].to_csv("macro_forecast.csv", index=False)

joblib.dump(
    {"model": macro_clf, "features": FEATURES,
     "classes": list(macro_clf.classes_), "version": "1.0-macro"},
    "macro_model.pkl", compress=3,
)

print("\nForecast summary (60 months):")
print(f"  avg inflation YoY     : {fut['inflation_yoy_pct'].mean():.1f}%")
print(f"  avg policy rate       : {fut['policy_rate_pct'].mean():.1f}%")
print(f"  avg monthly inflation : {fut['monthly_inflation'].mean()*100:.2f}%")
print(f"  avg monthly wage grow : {fut['monthly_wage_growth'].mean()*100:.2f}%")
print(f"  macro pressure split  : "
      f"{fut['predicted_macro_pressure'].value_counts().to_dict()}")
print("\nSaved: macro_forecast.csv, macro_model.pkl, feature_importance_macro.csv")
print("→ Stage 2 (02_monte_carlo_ml.py) will read macro_forecast.csv")
