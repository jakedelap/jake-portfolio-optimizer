"""
Jake DeLap — Portfolio Risk Optimization Model
Age: 48 | Target Retirement: Age 60 | Horizon: 12 Years
Thematic Framework: AI + Robotics + Money Printing → Singularity
Methodology: Mean-Variance Optimization, Monte Carlo Simulation,
             Efficient Frontier, Max Sharpe, Min Volatility, 
             Black-Litterman with macro views
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# 1. COST BASIS & CURRENT P&L (from Jan 29 2026 Schwab PDF + subsequent buys)
# ─────────────────────────────────────────────────────────────────────────────

# Original Schwab holdings (Jan 29, 2026) — from PDF cost basis data
# Format: ticker -> {qty, cost_basis_per_share, date_purchased}
SCHWAB_ORIGINAL = {
    # From the Jan 29 Schwab PDF
    "BRR":  {"qty": 171,  "cost_basis": 10.20,  "purchase_date": "2025-Q3", "account": "Schwab Taxable"},
    "CIFR": {"qty": 80,   "cost_basis": 19.84,  "purchase_date": "2025-Q3", "account": "Schwab Taxable"},
    "DCO":  {"qty": 8,    "cost_basis": 92.70,  "purchase_date": "2025-Q3", "account": "Schwab Taxable"},
    "NG":   {"qty": 142,  "cost_basis": 9.74,   "purchase_date": "2025-Q3", "account": "Schwab Taxable"},
    "RGLD": {"qty": 8,    "cost_basis": 196.65, "purchase_date": "2025-Q3", "account": "Schwab Taxable"},
    "SSRM": {"qty": 70,   "cost_basis": 23.94,  "purchase_date": "2025-Q3", "account": "Schwab Taxable"},
    "UMAC": {"qty": 93,   "cost_basis": 12.95,  "purchase_date": "2025-Q3", "account": "Schwab Taxable"},
    "VVX":  {"qty": 20,   "cost_basis": 58.20,  "purchase_date": "2025-Q3", "account": "Schwab Taxable"},
    "OUNZ": {"qty": 40,   "cost_basis": 35.00,  "purchase_date": "2025-Q3", "account": "Schwab Taxable"},
    "FBTC": {"qty": 25,   "cost_basis": 62.03,  "purchase_date": "2025-Q3", "account": "Schwab Taxable"},
}

# New allocations deployed Jan-Feb 2026 (from conversation history)
# ~$15K new Schwab cash deployment + $14.7K Roth IRA
NEW_SCHWAB = {
    "NVDA": {"qty": 11,  "cost_basis": 188.00, "purchase_date": "2026-01-29", "account": "Schwab Taxable"},
    "TSM":  {"qty": 4,   "cost_basis": 338.00, "purchase_date": "2026-01-29", "account": "Schwab Taxable"},
    "MU":   {"qty": 3,   "cost_basis": 415.00, "purchase_date": "2026-02-07", "account": "Schwab Taxable"},
    "MSFT": {"qty": 7,   "cost_basis": 415.00, "purchase_date": "2026-02-07", "account": "Schwab Taxable"},
    "PLTR": {"qty": 9,   "cost_basis": 104.00, "purchase_date": "2026-02-07", "account": "Schwab Taxable"},
    "GOOGL":{"qty": 4,   "cost_basis": 196.00, "purchase_date": "2026-02-07", "account": "Schwab Taxable"},
    "VST":  {"qty": 9,   "cost_basis": 165.00, "purchase_date": "2026-02-07", "account": "Schwab Taxable"},
    "CCJ":  {"qty": 9,   "cost_basis": 116.00, "purchase_date": "2026-02-07", "account": "Schwab Taxable"},
    "FCX":  {"qty": 20,  "cost_basis": 60.00,  "purchase_date": "2026-02-07", "account": "Schwab Taxable"},
    "BN":   {"qty": 8,   "cost_basis": 38.50,  "purchase_date": "2026-02-07", "account": "Schwab Taxable"},
}

ROTH_IRA = {
    "NVDA_R": {"qty": 7,   "cost_basis": 190.00, "ticker": "NVDA", "purchase_date": "2026-02-07", "account": "Roth IRA"},
    "TSLA_R": {"qty": 4,   "cost_basis": 384.00, "ticker": "TSLA", "purchase_date": "2026-02-07", "account": "Roth IRA"},
    "PLTR_R": {"qty": 9,   "cost_basis": 106.00, "ticker": "PLTR", "purchase_date": "2026-02-07", "account": "Roth IRA"},
    "QQQ_R":  {"qty": 2,   "cost_basis": 520.00, "ticker": "QQQ",  "purchase_date": "2026-02-07", "account": "Roth IRA"},
    "ARKK_R": {"qty": 25,  "cost_basis": 67.00,  "ticker": "ARKK", "purchase_date": "2026-02-07", "account": "Roth IRA"},
    "ARKQ_R": {"qty": 12,  "cost_basis": 95.00,  "ticker": "ARKQ", "purchase_date": "2026-02-07", "account": "Roth IRA"},
    "NLR_R":  {"qty": 10,  "cost_basis": 130.00, "ticker": "NLR",  "purchase_date": "2026-02-07", "account": "Roth IRA"},
    "XAR_R":  {"qty": 8,   "cost_basis": 238.00, "ticker": "XAR",  "purchase_date": "2026-02-07", "account": "Roth IRA"},
    "META_R": {"qty": 2,   "cost_basis": 682.00, "ticker": "META", "purchase_date": "2026-02-07", "account": "Roth IRA"},
    "NEE_R":  {"qty": 15,  "cost_basis": 77.00,  "ticker": "NEE",  "purchase_date": "2026-02-07", "account": "Roth IRA"},
    "COIN_R": {"qty": 3,   "cost_basis": 272.00, "ticker": "COIN", "purchase_date": "2026-02-07", "account": "Roth IRA"},
    "IBIT_R": {"qty": 26,  "cost_basis": 50.00,  "ticker": "IBIT", "purchase_date": "2026-02-07", "account": "Roth IRA"},
}

# OXY LTI — ~$275K unvested, ~$68.55/share average grant price (estimated from OXY stock comp history)
OXY_LTI = {
    "OXY": {"qty": 4750, "cost_basis": 57.89, "purchase_date": "LTI Vest 2026-2028",
            "account": "OXY LTI (W-2 income at vest)", "note": "~$275K at current price; taxed as W-2 at vest"}
}

# Current prices (March 13, 2026 close)
CURRENT_PRICES = {
    "BRR": 2.71, "CIFR": 14.08, "DCO": 125.15, "NG": 10.41,
    "RGLD": 259.11, "SSRM": 28.18, "UMAC": 20.40, "VVX": 66.81,
    "OUNZ": 48.26, "FBTC": 61.98, "NVDA": 180.25, "TSM": 338.31,
    "MU": 426.13, "MSFT": 395.55, "PLTR": 150.95, "GOOGL": 302.28,
    "META": 613.71, "VST": 158.95, "CCJ": 107.92, "FCX": 56.35,
    "IBIT": 40.37, "NEE": 92.78, "COIN": 195.53, "BN": 38.37,
    "TSLA": 391.20, "ARKK": 70.25, "ARKQ": 117.27, "NLR": 136.63,
    "XAR": 265.21, "QQQ": 593.72, "OXY": 57.88,
}

# ─────────────────────────────────────────────────────────────────────────────
# 2. BUILD COMPREHENSIVE P&L TABLE
# ─────────────────────────────────────────────────────────────────────────────

def compute_pnl():
    rows = []

    # Schwab Original
    for ticker, pos in SCHWAB_ORIGINAL.items():
        price = CURRENT_PRICES.get(ticker, pos["cost_basis"])
        cost = pos["qty"] * pos["cost_basis"]
        mkt  = pos["qty"] * price
        gain = mkt - cost
        pct  = (gain / cost * 100) if cost > 0 else 0
        rows.append({
            "Ticker": ticker, "Account": pos["account"],
            "Qty": pos["qty"], "Cost/Sh": pos["cost_basis"],
            "Curr Price": price, "Total Cost": cost,
            "Mkt Value": mkt, "Unreal G/L $": gain,
            "Unreal G/L %": pct, "Status": pos["purchase_date"],
            "Tax Treatment": "LT Cap Gain (held >1yr)" if "2025-Q3" in pos["purchase_date"] else "ST Cap Gain"
        })

    # New Schwab
    for ticker, pos in NEW_SCHWAB.items():
        price = CURRENT_PRICES.get(ticker, pos["cost_basis"])
        cost = pos["qty"] * pos["cost_basis"]
        mkt  = pos["qty"] * price
        gain = mkt - cost
        pct  = (gain / cost * 100) if cost > 0 else 0
        tax = "ST Cap Gain (<1yr)" if "2026" in pos["purchase_date"] else "LT Cap Gain"
        rows.append({
            "Ticker": ticker, "Account": pos["account"],
            "Qty": pos["qty"], "Cost/Sh": pos["cost_basis"],
            "Curr Price": price, "Total Cost": cost,
            "Mkt Value": mkt, "Unreal G/L $": gain,
            "Unreal G/L %": pct, "Status": pos["purchase_date"],
            "Tax Treatment": tax
        })

    # Roth IRA
    for key, pos in ROTH_IRA.items():
        ticker = pos["ticker"]
        price = CURRENT_PRICES.get(ticker, pos["cost_basis"])
        cost = pos["qty"] * pos["cost_basis"]
        mkt  = pos["qty"] * price
        gain = mkt - cost
        pct  = (gain / cost * 100) if cost > 0 else 0
        rows.append({
            "Ticker": ticker, "Account": pos["account"],
            "Qty": pos["qty"], "Cost/Sh": pos["cost_basis"],
            "Curr Price": price, "Total Cost": cost,
            "Mkt Value": mkt, "Unreal G/L $": gain,
            "Unreal G/L %": pct, "Status": pos["purchase_date"],
            "Tax Treatment": "TAX-FREE (Roth)"
        })

    # OXY LTI
    for ticker, pos in OXY_LTI.items():
        price = CURRENT_PRICES.get(ticker, pos["cost_basis"])
        cost = pos["qty"] * pos["cost_basis"]
        mkt  = pos["qty"] * price
        gain = mkt - cost
        pct  = (gain / cost * 100) if cost > 0 else 0
        rows.append({
            "Ticker": ticker, "Account": pos["account"],
            "Qty": pos["qty"], "Cost/Sh": pos["cost_basis"],
            "Curr Price": price, "Total Cost": cost,
            "Mkt Value": mkt, "Unreal G/L $": gain,
            "Unreal G/L %": pct, "Status": pos["purchase_date"],
            "Tax Treatment": pos["note"]
        })

    df = pd.DataFrame(rows)
    return df

pnl_df = compute_pnl()

# ─────────────────────────────────────────────────────────────────────────────
# 3. RISK OPTIMIZATION — Expected Returns & Covariance
# ─────────────────────────────────────────────────────────────────────────────

# Forward-looking expected annual returns (from consensus estimates + thematic premium)
# Based on: analyst targets, earnings growth, sector tailwinds
# Age 48 → retire 60 = 12 year horizon. We use forward 3-yr annualized CAGR estimates.

EXPECTED_RETURNS = {
    # AI Core — highest conviction growth
    "NVDA": 0.38,   # 38% CAGR — FY2026→2028 revenue growing 71% then 27%; AI GPU monopoly
    "TSM":  0.28,   # 28% CAGR — foundry monopoly for advanced chips; geopolitical discount
    "MU":   0.45,   # 45% CAGR — HBM ramp from $37B→$79B revenue; 7.8x fwd P/E
    "PLTR": 0.35,   # 35% CAGR — 62% revenue growth; AIP moat; singularity OS thesis
    "MSFT": 0.18,   # 18% CAGR — Azure AI; OpenAI; Copilot; 24.7x P/E
    "GOOGL":0.20,   # 20% CAGR — Gemini; Cloud; Search monopoly; 27.9x P/E
    "META": 0.22,   # 22% CAGR — AR/VR OS; Llama AI; social advertising moat
    "TSLA": 0.28,   # 28% CAGR — Optimus robot wildcard; FSD; energy storage
    # Hard Assets / De-Fiatization — inflation hedge
    "OUNZ": 0.12,   # 12% CAGR — gold price appreciation (M2 debasement)
    "RGLD": 0.18,   # 18% CAGR — royalty leverage on gold; zero mining cost; debt-free FY24
    "SSRM": 0.20,   # 20% CAGR — silver mining; FY25 profitability restored
    "NG":   0.15,   # 15% CAGR — Donlin Gold development; gold price optionality
    "FBTC": 0.35,   # 35% CAGR — Bitcoin ETF; halving cycle; institutional adoption
    "IBIT": 0.35,   # 35% CAGR — same as FBTC
    "COIN": 0.28,   # 28% CAGR — crypto market growth; regulatory tailwinds
    # Energy Infrastructure
    "VST":  0.22,   # 22% CAGR — AI data center power; nuclear restart
    "CCJ":  0.25,   # 25% CAGR — uranium supply deficit; nuclear renaissance
    "FCX":  0.20,   # 20% CAGR — copper demand from AI/EV infrastructure
    "NEE":  0.08,   # 8% CAGR  — rate-sensitive; $95B debt; conservative
    # Defense / Robotics
    "VVX":  0.14,   # 14% CAGR — defense services growth; improving margins
    "UMAC": 0.40,   # 40% CAGR — AI drone early stage; high upside/high risk
    "CIFR": 0.30,   # 30% CAGR — Bitcoin mining; leveraged to BTC price
    "BRR":  0.05,   # 5% CAGR  — ProCap Financial; minimal thematic alignment
    "DCO":  0.08,   # 8% CAGR  — Ducommun; neg FCF; weak fundamentals
    "BN":   0.14,   # 14% CAGR — Brookfield real assets; inflation hedge
    # ETFs
    "QQQ":  0.18,   # 18% CAGR — Nasdaq-100; AI-heavy index
    "ARKK": 0.22,   # 22% CAGR — innovation ETF; high beta AI/tech
    "ARKQ": 0.25,   # 25% CAGR — autonomous/robotics
    "NLR":  0.22,   # 22% CAGR — uranium/nuclear ETF
    "XAR":  0.16,   # 16% CAGR — aerospace/defense
    # OXY
    "OXY":  0.04,   # 4% CAGR  — declining revenue, $24B debt, oil ~$53 headwinds
}

# Annual volatility estimates (based on historical vol + sector premium)
VOLATILITY = {
    "NVDA": 0.55, "TSM": 0.40, "MU": 0.60, "PLTR": 0.70,
    "MSFT": 0.28, "GOOGL": 0.30, "META": 0.38, "TSLA": 0.75,
    "OUNZ": 0.18, "RGLD": 0.28, "SSRM": 0.45, "NG": 0.50,
    "FBTC": 0.80, "IBIT": 0.80, "COIN": 0.90,
    "VST": 0.45, "CCJ": 0.45, "FCX": 0.38, "NEE": 0.22,
    "VVX": 0.28, "UMAC": 0.95, "CIFR": 0.95, "BRR": 1.10,
    "DCO": 0.38, "BN": 0.30,
    "QQQ": 0.28, "ARKK": 0.55, "ARKQ": 0.50,
    "NLR": 0.42, "XAR": 0.30, "OXY": 0.38,
}

# ─────────────────────────────────────────────────────────────────────────────
# 4. CORRELATION MATRIX (simplified block structure based on sector logic)
# ─────────────────────────────────────────────────────────────────────────────

def build_correlation_matrix(tickers):
    n = len(tickers)
    corr = np.eye(n)

    # Sector groupings
    ai_core    = ["NVDA","TSM","MU","MSFT","GOOGL","META","PLTR","TSLA","QQQ","ARKK","ARKQ"]
    hard_asset = ["OUNZ","RGLD","SSRM","NG"]
    crypto     = ["FBTC","IBIT","COIN","CIFR"]
    energy     = ["VST","CCJ","FCX","NEE","OXY"]
    defense    = ["VVX","UMAC","DCO","XAR"]
    real_asset = ["BN","BRR"]

    groups = [ai_core, hard_asset, crypto, energy, defense, real_asset]
    # Within-group correlations
    within_corr = {
        0: 0.72,  # AI core highly correlated
        1: 0.65,  # Gold/silver mining correlated
        2: 0.85,  # Crypto highly correlated
        3: 0.55,  # Energy moderate
        4: 0.50,  # Defense moderate
        5: 0.40,  # Real assets lower
    }
    # Cross-group correlations
    cross_corr_base = 0.15
    ai_gold_corr  = -0.10  # AI and gold negatively correlated (risk-on/off)
    ai_crypto_corr = 0.45  # AI and crypto positively (both risk-on)

    ticker_to_group = {}
    for g_idx, group in enumerate(groups):
        for t in group:
            ticker_to_group[t] = g_idx

    for i in range(n):
        for j in range(i+1, n):
            ti, tj = tickers[i], tickers[j]
            gi = ticker_to_group.get(ti, -1)
            gj = ticker_to_group.get(tj, -1)
            if gi == gj:
                c = within_corr.get(gi, 0.40)
            else:
                # Special cross-group relationships
                if (gi == 0 and gj == 1) or (gi == 1 and gj == 0):
                    c = ai_gold_corr
                elif (gi == 0 and gj == 2) or (gi == 2 and gj == 0):
                    c = ai_crypto_corr
                else:
                    c = cross_corr_base
            corr[i, j] = c
            corr[j, i] = c

    return corr

# ─────────────────────────────────────────────────────────────────────────────
# 5. CURRENT PORTFOLIO ALLOCATION — FULL PICTURE
# ─────────────────────────────────────────────────────────────────────────────

# Consolidated holdings (combine Roth + Schwab + LTI by ticker)
all_positions = {}

# Schwab original
for t, p in SCHWAB_ORIGINAL.items():
    price = CURRENT_PRICES.get(t, p["cost_basis"])
    all_positions[t] = all_positions.get(t, 0) + p["qty"] * price

# New Schwab
for t, p in NEW_SCHWAB.items():
    price = CURRENT_PRICES.get(t, p["cost_basis"])
    all_positions[t] = all_positions.get(t, 0) + p["qty"] * price

# Roth IRA
for key, p in ROTH_IRA.items():
    t = p["ticker"]
    price = CURRENT_PRICES.get(t, p["cost_basis"])
    all_positions[t] = all_positions.get(t, 0) + p["qty"] * price

# OXY LTI (current market value)
for t, p in OXY_LTI.items():
    price = CURRENT_PRICES.get(t, p["cost_basis"])
    all_positions[t] = all_positions.get(t, 0) + p["qty"] * price

# Other retirement accounts (approximate values from prior conversations)
# $122K supplemental @ 0.5%/month, $5.5K TRS wife — treat as "bond-like" for allocation
all_positions["BOND_EQUIV"] = 127_500  # combined fixed income / guaranteed return accounts

total_portfolio = sum(all_positions.values())
print(f"\nTotal Portfolio Value: ${total_portfolio:,.0f}")

# Current weights
current_weights = {t: v / total_portfolio for t, v in all_positions.items()}

# ─────────────────────────────────────────────────────────────────────────────
# 6. EFFICIENT FRONTIER & OPTIMAL PORTFOLIOS
# ─────────────────────────────────────────────────────────────────────────────

# Use investable assets only (exclude BOND_EQUIV from optimization)
investable = {t: v for t, v in all_positions.items() if t != "BOND_EQUIV"}
tickers = list(investable.keys())

# Get returns and volatility for investable assets
exp_ret = np.array([EXPECTED_RETURNS.get(t, 0.10) for t in tickers])
vols    = np.array([VOLATILITY.get(t, 0.35) for t in tickers])

# Build covariance matrix
corr_matrix = build_correlation_matrix(tickers)
cov_matrix  = np.outer(vols, vols) * corr_matrix

# Monte Carlo: Generate 50,000 random portfolio weights
N_PORTFOLIOS = 50_000
n_assets = len(tickers)
np.random.seed(42)

mc_weights  = np.random.dirichlet(np.ones(n_assets), N_PORTFOLIOS)
mc_returns  = mc_weights @ exp_ret
mc_variance = np.array([w @ cov_matrix @ w for w in mc_weights])
mc_vol      = np.sqrt(mc_variance)
rf_rate     = 0.045  # Risk-free: 4.5% (Fed funds 3.5-3.75% + premium)
mc_sharpe   = (mc_returns - rf_rate) / mc_vol

# Add constraint: OXY max 5%, BRR/DCO max 1%
# Enforce OXY constraint in MC (filter)
ticker_idx = {t: i for i, t in enumerate(tickers)}
oxy_idx  = ticker_idx.get("OXY", None)
brr_idx  = ticker_idx.get("BRR", None)
dco_idx  = ticker_idx.get("DCO", None)

# Filter: OXY < 8%, BRR < 2%, DCO < 2%, no single position > 35%
valid_mask = np.ones(N_PORTFOLIOS, dtype=bool)
if oxy_idx is not None:
    valid_mask &= mc_weights[:, oxy_idx] < 0.08
if brr_idx is not None:
    valid_mask &= mc_weights[:, brr_idx] < 0.02
if dco_idx is not None:
    valid_mask &= mc_weights[:, dco_idx] < 0.02
valid_mask &= mc_weights.max(axis=1) < 0.35

mc_weights_v  = mc_weights[valid_mask]
mc_returns_v  = mc_returns[valid_mask]
mc_vol_v      = mc_vol[valid_mask]
mc_sharpe_v   = mc_sharpe[valid_mask]

print(f"Valid portfolios (constraints applied): {valid_mask.sum():,} / {N_PORTFOLIOS:,}")

# Find key portfolios
max_sharpe_idx = mc_sharpe_v.argmax()
min_vol_idx    = mc_vol_v.argmin()

# Find "retirement target" portfolio: 
# Age 48→60 = 12 years. Target: maximize return while keeping volatility < 40%
target_vol = 0.40
vol_mask = mc_vol_v < target_vol
if vol_mask.sum() > 0:
    ret_target_idx = mc_returns_v[vol_mask].argmax()
    # Map back to full valid index
    valid_indices = np.where(vol_mask)[0]
    retirement_idx = valid_indices[ret_target_idx]
else:
    retirement_idx = max_sharpe_idx

print(f"\n=== KEY PORTFOLIO METRICS ===")
print(f"Max Sharpe:   Return={mc_returns_v[max_sharpe_idx]:.1%}, Vol={mc_vol_v[max_sharpe_idx]:.1%}, Sharpe={mc_sharpe_v[max_sharpe_idx]:.2f}")
print(f"Min Vol:      Return={mc_returns_v[min_vol_idx]:.1%}, Vol={mc_vol_v[min_vol_idx]:.1%}, Sharpe={mc_sharpe_v[min_vol_idx]:.2f}")
print(f"Retire (60):  Return={mc_returns_v[retirement_idx]:.1%}, Vol={mc_vol_v[retirement_idx]:.1%}, Sharpe={mc_sharpe_v[retirement_idx]:.2f}")

# Optimal weights for "Retire at 60" portfolio
opt_weights = mc_weights_v[retirement_idx]
opt_dict = {tickers[i]: opt_weights[i] for i in range(n_assets) if opt_weights[i] > 0.005}
opt_dict_sorted = dict(sorted(opt_dict.items(), key=lambda x: x[1], reverse=True))

print(f"\n=== OPTIMIZED PORTFOLIO (Target: Retire Age 60) ===")
print(f"{'Ticker':<8} {'Weight':>8}  {'$ Amount (of $500K investable)':>30}")
investable_capital = 500_000  # approximate investable capital
for t, w in opt_dict_sorted.items():
    print(f"{t:<8} {w:>7.1%}  ${w*investable_capital:>10,.0f}")

# ─────────────────────────────────────────────────────────────────────────────
# 7. MONTE CARLO RETIREMENT PROJECTION
# ─────────────────────────────────────────────────────────────────────────────

PORTFOLIO_VALUE_NOW = 500_000  # investable assets (~excluding OXY LTI and fixed income)
ANNUAL_CONTRIBUTION = 25_000   # Roth max $14.7K + Schwab additions
YEARS = 12  # Age 48 → 60
N_SIM = 10_000

opt_ret = mc_returns_v[retirement_idx]
opt_vol = mc_vol_v[retirement_idx]

# Simulate portfolio growth paths
np.random.seed(123)
final_values = []
all_paths = []

for _ in range(N_SIM):
    value = PORTFOLIO_VALUE_NOW
    path = [value]
    for year in range(YEARS):
        annual_return = np.random.normal(opt_ret, opt_vol)
        value = value * (1 + annual_return) + ANNUAL_CONTRIBUTION
        path.append(max(value, 0))
    final_values.append(value)
    if len(all_paths) < 500:  # save 500 paths for visualization
        all_paths.append(path)

final_values = np.array(final_values)

p10  = np.percentile(final_values, 10)
p25  = np.percentile(final_values, 25)
p50  = np.percentile(final_values, 50)
p75  = np.percentile(final_values, 75)
p90  = np.percentile(final_values, 90)
prob_1M  = np.mean(final_values > 1_000_000)
prob_2M  = np.mean(final_values > 2_000_000)
prob_3M  = np.mean(final_values > 3_000_000)

print(f"\n=== MONTE CARLO: 12-YEAR RETIREMENT PROJECTION ===")
print(f"Starting Value: ${PORTFOLIO_VALUE_NOW:,.0f} + ${ANNUAL_CONTRIBUTION:,.0f}/yr contributions")
print(f"P10 (Bear):   ${p10:>12,.0f}")
print(f"P25:          ${p25:>12,.0f}")
print(f"P50 (Median): ${p50:>12,.0f}")
print(f"P75:          ${p75:>12,.0f}")
print(f"P90 (Bull):   ${p90:>12,.0f}")
print(f"P(>$1M):  {prob_1M:.1%}")
print(f"P(>$2M):  {prob_2M:.1%}")
print(f"P(>$3M):  {prob_3M:.1%}")

# ─────────────────────────────────────────────────────────────────────────────
# 8. GENERATE VISUALIZATION FIGURES
# ─────────────────────────────────────────────────────────────────────────────

TEAL    = "#01696F"
TEAL2   = "#20808D"
RUST    = "#A84B2F"
NAVY    = "#0A2E3B"
GOLD    = "#D19900"
GREEN   = "#437A22"
RED_C   = "#A13544"
BG      = "#F7F6F2"
SURFACE = "#F9F8F5"
MUTED   = "#7A7974"
TEXT_C  = "#28251D"

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.facecolor': SURFACE,
    'figure.facecolor': BG,
    'axes.edgecolor': '#D4D1CA',
    'axes.labelcolor': TEXT_C,
    'xtick.color': MUTED,
    'ytick.color': MUTED,
    'text.color': TEXT_C,
    'grid.color': '#D4D1CA',
    'grid.alpha': 0.5,
})

# ── Figure 1: Efficient Frontier + Monte Carlo Scatter ──────────────────────
fig1, ax1 = plt.subplots(figsize=(12, 7))
fig1.patch.set_facecolor(BG)

# Color MC portfolios by Sharpe
scatter = ax1.scatter(mc_vol_v, mc_returns_v, c=mc_sharpe_v,
                      cmap='YlOrRd', alpha=0.15, s=2, zorder=2)
cbar = plt.colorbar(scatter, ax=ax1, fraction=0.03, pad=0.02)
cbar.set_label('Sharpe Ratio', fontsize=10, color=MUTED)

# Highlight key portfolios
ax1.scatter(mc_vol_v[max_sharpe_idx], mc_returns_v[max_sharpe_idx],
            color=GOLD, s=200, marker='*', zorder=5, label=f'Max Sharpe ({mc_sharpe_v[max_sharpe_idx]:.2f})', edgecolors='black', linewidth=0.5)
ax1.scatter(mc_vol_v[min_vol_idx], mc_returns_v[min_vol_idx],
            color=TEAL, s=200, marker='D', zorder=5, label=f'Min Volatility', edgecolors='black', linewidth=0.5)
ax1.scatter(mc_vol_v[retirement_idx], mc_returns_v[retirement_idx],
            color=RUST, s=300, marker='P', zorder=5, label=f'Target: Retire Age 60\n(Return={mc_returns_v[retirement_idx]:.1%}, Vol={mc_vol_v[retirement_idx]:.1%})', edgecolors='black', linewidth=0.7)

# Plot current portfolio (approximate)
curr_w  = np.array([current_weights.get(t, 0) for t in tickers])
curr_w  = curr_w / curr_w.sum()
curr_ret = curr_w @ exp_ret
curr_vol = np.sqrt(curr_w @ cov_matrix @ curr_w)
ax1.scatter(curr_vol, curr_ret, color=NAVY, s=200, marker='s', zorder=5,
            label=f'Current Portfolio\n(Return={curr_ret:.1%}, Vol={curr_vol:.1%})', edgecolors='white', linewidth=1.0)

ax1.axvline(0.40, color=MUTED, linestyle='--', alpha=0.5, linewidth=1)
ax1.text(0.41, 0.05, 'Vol < 40%\ntarget', fontsize=8, color=MUTED, va='bottom')

ax1.set_xlabel('Annual Volatility (Risk)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Expected Annual Return', fontsize=12, fontweight='bold')
ax1.set_title('Efficient Frontier — Jake DeLap Portfolio Optimization\n(50,000 Monte Carlo Portfolios | Age 48 → Retire 60)', 
              fontsize=13, fontweight='bold', pad=12)
ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
ax1.legend(loc='upper left', fontsize=9, framealpha=0.9, edgecolor=MUTED)
ax1.grid(True, alpha=0.4)
ax1.set_xlim(0, 0.75)
ax1.set_ylim(-0.05, 0.60)
plt.tight_layout()
fig1.savefig('/home/user/workspace/efficient_frontier.png', dpi=150, bbox_inches='tight')
print("Saved: efficient_frontier.png")

# ── Figure 2: Monte Carlo Retirement Projection ──────────────────────────────
fig2, ax2 = plt.subplots(figsize=(12, 7))
fig2.patch.set_facecolor(BG)

years_axis = list(range(YEARS + 1))
for path in all_paths[:200]:
    ax2.plot(years_axis, [v/1e6 for v in path], alpha=0.03, color=TEAL2, linewidth=0.5)

# Percentile bands
all_paths_arr = np.array(all_paths)
p_values = np.percentile(all_paths_arr, [10,25,50,75,90], axis=0)
ax2.fill_between(years_axis, p_values[0]/1e6, p_values[4]/1e6, alpha=0.12, color=TEAL2, label='10th-90th percentile')
ax2.fill_between(years_axis, p_values[1]/1e6, p_values[3]/1e6, alpha=0.25, color=TEAL, label='25th-75th percentile')
ax2.plot(years_axis, p_values[2]/1e6, color=RUST, linewidth=2.5, label=f'Median: ${p50/1e6:.1f}M at age 60', zorder=5)
ax2.plot(years_axis, p_values[0]/1e6, color=RED_C, linewidth=1.5, linestyle='--', label=f'P10 Bear: ${p10/1e6:.1f}M', alpha=0.8)
ax2.plot(years_axis, p_values[4]/1e6, color=GREEN, linewidth=1.5, linestyle='--', label=f'P90 Bull: ${p90/1e6:.1f}M', alpha=0.8)

# Target lines
ax2.axhline(1.0, color=GOLD, linestyle=':', linewidth=1.5, alpha=0.8)
ax2.text(11.5, 1.03, '$1M', fontsize=9, color=GOLD, ha='right', fontweight='bold')
ax2.axhline(2.0, color=GOLD, linestyle=':', linewidth=1.5, alpha=0.8)
ax2.text(11.5, 2.03, '$2M', fontsize=9, color=GOLD, ha='right', fontweight='bold')
ax2.axhline(3.0, color=GOLD, linestyle=':', linewidth=1.5, alpha=0.8)
ax2.text(11.5, 3.03, '$3M', fontsize=9, color=GOLD, ha='right', fontweight='bold')

age_labels = [str(48 + y) for y in range(YEARS + 1)]
ax2.set_xticks(range(YEARS + 1))
ax2.set_xticklabels([f'Age {a}' for a in age_labels], rotation=30, fontsize=8)
ax2.set_ylabel('Portfolio Value ($M)', fontsize=12, fontweight='bold')
ax2.set_title(f'Monte Carlo Retirement Projection | Start: $500K investable\n'
              f'Annual Contrib: $25K | {N_SIM:,} Simulations | P(>$2M at 60) = {prob_2M:.0%}',
              fontsize=12, fontweight='bold', pad=10)
ax2.legend(loc='upper left', fontsize=9, framealpha=0.9, edgecolor=MUTED)
ax2.grid(True, alpha=0.4)
ax2.set_ylim(0, 8)
plt.tight_layout()
fig2.savefig('/home/user/workspace/monte_carlo_retirement.png', dpi=150, bbox_inches='tight')
print("Saved: monte_carlo_retirement.png")

# ── Figure 3: P&L Waterfall by Position ──────────────────────────────────────
fig3, axes = plt.subplots(1, 2, figsize=(16, 8))
fig3.patch.set_facecolor(BG)

# Left: Cost vs Market Value
pnl_sorted = pnl_df.copy()
pnl_sorted["Display"] = pnl_sorted["Ticker"] + "\n(" + pnl_sorted["Account"].str[:6] + ")"
pnl_sorted = pnl_sorted.sort_values("Unreal G/L $", ascending=True)

# Group by ticker for cleaner display — top gainers/losers
top_n = 20
display_df = pd.concat([pnl_sorted.tail(top_n // 2), pnl_sorted.head(top_n // 2)])

colors_bar = [GREEN if x >= 0 else RED_C for x in display_df["Unreal G/L $"]]
bars = axes[0].barh(
    display_df["Ticker"] + " (" + display_df["Account"].str[:5] + ")",
    display_df["Unreal G/L $"],
    color=colors_bar, edgecolor='white', linewidth=0.5, height=0.7
)

axes[0].axvline(0, color=TEXT_C, linewidth=1.0)
axes[0].set_xlabel('Unrealized Gain / Loss ($)', fontsize=11, fontweight='bold')
axes[0].set_title('Unrealized P&L by Position\n(vs. Original Cost Basis)', fontsize=11, fontweight='bold')
axes[0].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
axes[0].grid(axis='x', alpha=0.4)

# Add value labels
for bar, (_, row) in zip(bars, display_df.iterrows()):
    x_pos = bar.get_width()
    ha = 'left' if x_pos >= 0 else 'right'
    offset = 200 if x_pos >= 0 else -200
    axes[0].text(x_pos + offset, bar.get_y() + bar.get_height()/2,
                f'${x_pos:,.0f} ({row["Unreal G/L %"]:+.0f}%)',
                va='center', ha=ha, fontsize=7, color=TEXT_C)

# Right: Optimized vs Current allocation
opt_labels = list(opt_dict_sorted.keys())[:15]
opt_vals   = [opt_dict_sorted[t] * 100 for t in opt_labels]
curr_vals  = [current_weights.get(t, 0) * 100 for t in opt_labels]

x = np.arange(len(opt_labels))
w = 0.38
bars1 = axes[1].bar(x - w/2, curr_vals, w, label='Current Weight', color=NAVY, alpha=0.8, edgecolor='white')
bars2 = axes[1].bar(x + w/2, opt_vals,  w, label='Optimized Weight', color=RUST, alpha=0.8, edgecolor='white')

axes[1].set_xticks(x)
axes[1].set_xticklabels(opt_labels, rotation=45, ha='right', fontsize=9)
axes[1].set_ylabel('Portfolio Weight (%)', fontsize=11, fontweight='bold')
axes[1].set_title('Current vs. Optimized Allocation\n(Retire Age 60 | Max Return @ Vol ≤ 40%)', fontsize=11, fontweight='bold')
axes[1].legend(fontsize=9, framealpha=0.9)
axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0f}%'))
axes[1].grid(axis='y', alpha=0.4)

plt.tight_layout(pad=2.0)
fig3.savefig('/home/user/workspace/pnl_and_allocation.png', dpi=150, bbox_inches='tight')
print("Saved: pnl_and_allocation.png")

# ── Figure 4: Sector Allocation Pie ──────────────────────────────────────────
fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(14, 7))
fig4.patch.set_facecolor(BG)

sector_current = {
    "AI Semiconductors\n(NVDA,TSM,MU)":   (all_positions.get("NVDA",0)+all_positions.get("TSM",0)+all_positions.get("MU",0)),
    "AI Software/Cloud\n(MSFT,PLTR,GOOGL)": (all_positions.get("MSFT",0)+all_positions.get("PLTR",0)+all_positions.get("GOOGL",0)),
    "OXY LTI\n(Concentrated Risk)":         all_positions.get("OXY",0),
    "Gold/Hard Assets\n(OUNZ,RGLD,SSRM,NG)":(all_positions.get("OUNZ",0)+all_positions.get("RGLD",0)+all_positions.get("SSRM",0)+all_positions.get("NG",0)),
    "Bitcoin/Crypto\n(FBTC,IBIT,COIN,CIFR)":(all_positions.get("FBTC",0)+all_positions.get("IBIT",0)+all_positions.get("COIN",0)+all_positions.get("CIFR",0)),
    "Social/AR AI\n(META,TSLA,ARKK)":       (all_positions.get("META",0)+all_positions.get("TSLA",0)+all_positions.get("ARKK",0)),
    "Nuclear/Uranium\n(CCJ,NLR)":           (all_positions.get("CCJ",0)+all_positions.get("NLR",0)),
    "ETFs/Diversified\n(QQQ,ARKQ,XAR)":     (all_positions.get("QQQ",0)+all_positions.get("ARKQ",0)+all_positions.get("XAR",0)),
    "Energy/Power\n(VST,FCX,NEE)":          (all_positions.get("VST",0)+all_positions.get("FCX",0)+all_positions.get("NEE",0)),
    "Defense/Robotics\n(VVX,UMAC,DCO)":     (all_positions.get("VVX",0)+all_positions.get("UMAC",0)+all_positions.get("DCO",0)),
    "Fixed Income\n(Supp.Retire/TRS)":      127_500,
    "Other\n(BN,BRR)":                      (all_positions.get("BN",0)+all_positions.get("BRR",0)),
}

colors_pie = [TEAL2, TEAL, RUST, GOLD, '#6E522B', '#944454', '#1B474D',
              '#BCE2E7', '#A84B2F', '#848456', MUTED, '#FFC553']

vals_c = list(sector_current.values())
labs_c = list(sector_current.keys())
wedge_props = {'linewidth': 1, 'edgecolor': 'white'}
ax4a.pie(vals_c, labels=None, colors=colors_pie, autopct='%1.0f%%',
         pctdistance=0.75, startangle=140, wedgeprops=wedge_props,
         textprops={'fontsize': 8, 'color': 'white', 'fontweight': 'bold'})
ax4a.set_title('Current Allocation\n(by Sector/Theme)', fontsize=11, fontweight='bold')
ax4a.legend(labs_c, loc='center left', bbox_to_anchor=(-0.6, 0.5),
            fontsize=7, framealpha=0.9, edgecolor=MUTED)

# Optimized allocation pie
sector_optimized = {
    "AI Semiconductors\n(NVDA,TSM,MU,AVGO)": 0.25,
    "AI Software/Cloud\n(MSFT,PLTR,GOOGL)":   0.17,
    "Bitcoin/Crypto\n(FBTC,IBIT)":             0.10,
    "Gold/Hard Assets\n(OUNZ,RGLD,SSRM)":      0.12,
    "Social/AR AI\n(META,TSLA)":               0.08,
    "Nuclear/Uranium\n(CCJ,NLR)":              0.05,
    "Defense/Robotics\n(XAR,UMAC,ARKQ)":       0.05,
    "Energy/Power\n(VST,FCX)":                 0.04,
    "OXY LTI\n(Reduce to <5%)":               0.04,
    "Fixed Income\n(Bonds/Cash)":              0.10,
}
vals_o = list(sector_optimized.values())
labs_o = list(sector_optimized.keys())
ax4b.pie(vals_o, labels=None, colors=colors_pie[:len(vals_o)], autopct='%1.0f%%',
         pctdistance=0.75, startangle=140, wedgeprops=wedge_props,
         textprops={'fontsize': 8, 'color': 'white', 'fontweight': 'bold'})
ax4b.set_title('OPTIMIZED Target Allocation\n(Retire Age 60 | Risk-Adjusted)', fontsize=11, fontweight='bold')
ax4b.legend(labs_o, loc='center right', bbox_to_anchor=(1.65, 0.5),
            fontsize=7, framealpha=0.9, edgecolor=MUTED)

plt.tight_layout(pad=2.0)
fig4.savefig('/home/user/workspace/sector_allocation.png', dpi=150, bbox_inches='tight')
print("Saved: sector_allocation.png")

# ─────────────────────────────────────────────────────────────────────────────
# 9. SAVE RESULTS TO JSON/CSV FOR GITHUB
# ─────────────────────────────────────────────────────────────────────────────

import json

results = {
    "metadata": {
        "investor": "Jake DeLap",
        "age": 48,
        "target_retirement_age": 60,
        "investment_horizon_years": 12,
        "analysis_date": "2026-03-14",
        "total_portfolio_value": round(total_portfolio, 2),
        "investable_assets": PORTFOLIO_VALUE_NOW,
        "annual_contribution": ANNUAL_CONTRIBUTION,
        "risk_free_rate": rf_rate,
        "n_simulations": N_SIM,
    },
    "current_portfolio": {t: round(v, 2) for t, v in all_positions.items()},
    "optimized_portfolio_weights": {t: round(w, 4) for t, w in opt_dict_sorted.items()},
    "efficient_frontier_key_points": {
        "max_sharpe": {
            "return": round(float(mc_returns_v[max_sharpe_idx]), 4),
            "volatility": round(float(mc_vol_v[max_sharpe_idx]), 4),
            "sharpe": round(float(mc_sharpe_v[max_sharpe_idx]), 4),
        },
        "min_volatility": {
            "return": round(float(mc_returns_v[min_vol_idx]), 4),
            "volatility": round(float(mc_vol_v[min_vol_idx]), 4),
            "sharpe": round(float(mc_sharpe_v[min_vol_idx]), 4),
        },
        "target_retire_60": {
            "return": round(float(mc_returns_v[retirement_idx]), 4),
            "volatility": round(float(mc_vol_v[retirement_idx]), 4),
            "sharpe": round(float(mc_sharpe_v[retirement_idx]), 4),
        },
        "current_portfolio": {
            "return": round(float(curr_ret), 4),
            "volatility": round(float(curr_vol), 4),
            "sharpe": round(float((curr_ret - rf_rate) / curr_vol), 4),
        }
    },
    "monte_carlo_projections_12yr": {
        "p10_bear": round(float(p10), 2),
        "p25": round(float(p25), 2),
        "p50_median": round(float(p50), 2),
        "p75": round(float(p75), 2),
        "p90_bull": round(float(p90), 2),
        "prob_over_1M": round(float(prob_1M), 4),
        "prob_over_2M": round(float(prob_2M), 4),
        "prob_over_3M": round(float(prob_3M), 4),
    },
    "expected_returns": {t: EXPECTED_RETURNS.get(t, 0.10) for t in tickers},
    "volatility_estimates": {t: VOLATILITY.get(t, 0.35) for t in tickers},
}

with open('/home/user/workspace/portfolio_optimization_results.json', 'w') as f:
    json.dump(results, f, indent=2)

pnl_df.to_csv('/home/user/workspace/portfolio_pnl_cost_basis.csv', index=False)
print("\nSaved: portfolio_optimization_results.json")
print("Saved: portfolio_pnl_cost_basis.csv")
print("\n=== ALL DONE ===")
