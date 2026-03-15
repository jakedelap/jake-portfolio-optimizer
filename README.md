# Jake DeLap — Portfolio Risk Optimizer
## AI/Robotics/Singularity Framework | Retire-at-60 Model

A quantitative portfolio optimization engine combining Modern Portfolio Theory, 
Monte Carlo simulation, and Efficient Frontier analysis. Built for a 48-year-old 
petroleum engineer targeting retirement at age 60 with a high-conviction AI/singularity thesis.

### Model Features
- **Monte Carlo Simulation**: 10,000 paths × 12-year horizon (age 48→60)
- **Efficient Frontier**: 50,000 randomly weighted portfolios
- **Three optimal portfolios**: Max Sharpe | Min Volatility | Retire-at-60 (max return ≤40% vol)
- **Full cost basis P&L**: Every position with tax treatment classification
- **4 visualization charts**: Efficient Frontier, Monte Carlo, P&L waterfall, Sector allocation

### Key Results (March 14, 2026)
| Portfolio | Return | Volatility | Sharpe |
|-----------|--------|-----------|--------|
| Max Sharpe | 21.8% | 21.3% | 0.81 |
| Min Volatility | 18.0% | 18.0% | 0.75 |
| **Retire at 60 ★** | **30.7%** | **39.1%** | **0.67** |

### Monte Carlo Outcomes (Starting $500K + $25K/yr contributions)
| Percentile | Portfolio at Age 60 |
|-----------|-------------------|
| P10 (Bear) | $1.95M |
| P25 | $4.0M |
| **P50 (Median)** | **$8.5M** |
| P75 | $17.7M |
| P90 (Bull) | $33.3M |

**P(>$1M at 60) = 96.3%** | **P(>$2M at 60) = 89.6%** | **P(>$3M at 60) = 82.2%**

### Dependencies
```
pip install numpy scipy matplotlib pandas
```

### Usage
```python
python portfolio_optimizer.py
```

Outputs:
- `efficient_frontier.png` — Efficient Frontier visualization
- `monte_carlo_retirement.png` — Monte Carlo projection chart
- `pnl_and_allocation.png` — P&L waterfall + target allocation
- `sector_allocation.png` — Sector breakdown
- `portfolio_optimization_results.json` — Full results JSON
- `portfolio_pnl_cost_basis.csv` — Complete P&L with cost basis

### Portfolio Thematic Framework
**Dominant themes: AI, Robotics, Money Printing → Singularity**
- M2 Money Supply: $22.44T (ALL-TIME HIGH, Jan 2026)
- US Public Debt: $38.9T (+$50B/week)
- AI capex: $120B+ in 2025 (TD Economics)
- Fed funds: 3.5–3.75% with easing expected

### Disclaimer
This model is for research and educational purposes only. Not financial advice.
Consult a qualified Registered Investment Advisor before making investment decisions.
