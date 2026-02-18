# Quantitative Mean Reversion Engine v3.3

A production-grade **long-only mean reversion backtester** featuring a two-layer portfolio architecture (core + tactical sleeves), adaptive regime classification, volatility-adjusted position sizing, and comprehensive statistical validation.

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub release](https://img.shields.io/github/v/release/chaulemuoichin/quant-mean-reversion-engine-v3.3)](https://github.com/chaulemuoichin/quant-mean-reversion-engine-v3.3/releases)

---

## Overview

This backtester implements a two-layer portfolio system separating core (buy-and-hold) and tactical (mean reversion) allocations. The regime classification engine combines four statistical tests — Hurst exponent, ADF stationarity, OU half-life, and Lo-MacKinlay variance ratio — with adaptive rolling thresholds to assign per-bar regime labels that gate trade entry.

**Sample result (NVDA, 5y, adaptive deployment, min confidence 60%):**
- Total Return: +732.12% | Sharpe: 1.262 | Max Drawdown: -53.91%
- Tactical sleeve: +553.40% on $20,000 starting capital
- Blocked signal rate: 88.5% (regime + confidence filters combined)
- Return attribution: Core 84.9% / Tactical alpha 15.1%

---

## Features

### Portfolio Architecture
- **Two-Layer Design**: Independent core (70-80%) and tactical (20-30%) sleeves
- **Adaptive Deployment**: 6-state capital deployment machine (CALM_UPTREND, PULLBACK, HIGH_VOL_DRAWDOWN, NEUTRAL, WAITING, DEPLETED)
- **DCA Support**: Dollar-cost averaging with configurable slippage and commission modeling
- **Cash Yield**: Daily compounding interest on idle capital

### Regime Classification
- **Multi-Factor Scoring**: Hurst exponent (R/S, R² >= 0.80), ADF stationarity, OU half-life, Lo-MacKinlay variance ratio
- **Adaptive Thresholds**: Rolling 10th/90th percentile boundaries — no fixed cutoffs
- **Per-Bar Labels**: MEAN_REVERTING, SIDEWAYS, TRENDING, AMBIGUOUS
- **Configurable Frequency**: Expensive tests run every N bars (default: 5)

### Signal Generation & Execution
- **Z-Score Entry/Exit**: Price-to-SMA200 ratio with rolling lookback normalization
- **Staged Exits**: Three-level trim [(z>=0.5: 25%), (z>=1.0: 50%), (z>=2.0: 100%)]
- **Reversal Confirmation**: RSI turning points + close-above-prior validation
- **Quality Filters**: Minimum trade notional ($1,000) and minimum shares (5)

### Risk Management
- **Volatility-Adjusted Sizing**: Risk-parity inspired position scaling (0.5x-1.5x multipliers)
- **ATR-Based Stops**: Dynamic stop-loss at 3x ATR with intraday gap-down handling
- **Thesis Break Detection**: Consecutive negative SMA slope triggers early exit
- **Max Holding Period**: 30-day automatic liquidation timer

### Performance Analytics
- **Metrics**: Sharpe, Sortino, Calmar, profit factor, max drawdown
- **Capture Ratios**: Upside/downside vs. buy-and-hold benchmark
- **Return Attribution**: Core beta vs. tactical alpha decomposition
- **Exit Analysis**: Stop-loss, target, time-decay, and thesis-break breakdown
- **Confidence Bins**: Trade performance stratified by entry confidence (0.02 intervals)

---

## Sample Output

### Equity Curves (NVDA, 5y, adaptive)
![Equity Curves](samples/01_NVDA_TL_mc060_5y_adaptive_sample_benchmark.png)

### Return Attribution
![Return Contribution](samples/01_NVDA_TL_mc060_5y_adaptive_sample_contribution.png)

### Drawdown vs. Baseline
![Drawdown](samples/01_NVDA_TL_mc060_5y_adaptive_sample_drawdown.png)

### Tactical Exposure vs. Price
![Exposure](samples/01_NVDA_TL_mc060_5y_adaptive_sample_exposure.png)

---

## Quick Start

```bash
git clone https://github.com/chaulemuoichin/quant-mean-reversion-engine-v3.3.git
cd quant-mean-reversion-engine-v3.3
pip install -r requirements.txt
```

```bash
# Basic backtest
python mean_reversion_standalone.py AAPL

# Two-layer with visualization
python mean_reversion_standalone.py AAPL --two-layer --plot

# Adaptive core deployment (recommended)
python mean_reversion_standalone.py NVDA --two-layer --core-entry adaptive --plot

# DCA over 60 days
python mean_reversion_standalone.py MSFT --two-layer --core-entry dca --dca-days 60

# Disable regime filter (baseline comparison)
python mean_reversion_standalone.py TSLA --no-regime --entry-at same_close

# Custom allocation
python mean_reversion_standalone.py NVDA --two-layer --core-pct 0.7 --tactical-pct 0.3 --cash-yield 4.5
```

---

## Configuration

```python
MeanReversionConfig(
    # Portfolio
    starting_capital=100_000.0,
    core_allocation_pct=0.80,
    tactical_allocation_pct=0.20,
    core_entry_mode="adaptive",        # instant | dca | adaptive

    # Signals
    entry_z=-1.5,                      # Enter at -1.5 sigma
    add_z=-2.0,                        # Add at -2.0 sigma
    trim_levels=[
        (0.5, 0.25),                   # Trim 25% at +0.5 sigma
        (1.0, 0.50),                   # Trim 50% at +1.0 sigma
        (2.0, 1.00),                   # Exit 100% at +2.0 sigma
    ],

    # Position sizing
    max_position_pct=0.15,
    vol_adjust_sizing=True,
    confidence_sizing_enabled=True,

    # Risk
    stop_atr_multiple=3.0,
    max_holding_days=30,
    thesis_break_sma_bars=10,

    # Regime
    regime_filter_enabled=True,
    allowed_regimes=["MEAN_REVERTING", "SIDEWAYS"],
    regime_adaptive_thresholds=True,

    # Execution
    entry_at="next_open",              # next_open | same_close
    slippage_pct=0.0005,               # 5 bps
    commission_per_trade=1.0,          # $1 per fill
    cash_yield_annual_pct=4.5,
)
```

---

## Testing

50+ unit tests across statistical accuracy, accounting identities, capital conservation, and no-lookahead properties.

```bash
# Run all tests
python -m unittest test_backtester -v

# Run specific suite
python -m unittest test_backtester.TestAdaptiveCoreDeployment -v
```

Coverage includes: Hurst/ADF/half-life accuracy, regime classification logic, z-score signals, staged exits, vol-adjusted sizing, confidence weighting, slippage/commission simulation, two-layer capital conservation, adaptive deployment monotonicity.

---

## Technical Reference

### Statistical Tests

| Test | Method | Mean Reversion Signal |
|---|---|---|
| Hurst Exponent | R/S with R² >= 0.80 gate | H < 0.5 |
| ADF | Augmented Dickey-Fuller | p < 0.05 |
| Half-Life | OU process (5-60 day range) | lambda > 0 |
| Variance Ratio | Lo-MacKinlay | VR < 1.0 |

### Regime Classification

Regime score = average of 4 binary signals (0 or 1 each):
- Hurst in [0, 0.45]: +1
- ADF p-value < 0.05: +1
- Half-life in [5, 60] days: +1
- Variance ratio < 0.80: +1

Adaptive thresholds (rolling 252-bar 10th/90th percentiles):
- Score > p90: MEAN_REVERTING
- p10 < Score < p90: SIDEWAYS
- Score < p10: TRENDING
- Insufficient data: AMBIGUOUS

### Adaptive Deployment States

```
CALM_UPTREND:      trend=1, dd<10%, vol<target  ->  deploy C/slow_days
PULLBACK:          trend=0, 5%<=dd<=20%         ->  deploy C/fast_days
HIGH_VOL_DRAWDOWN: dd>=15% or vol>=target       ->  deploy C/fast_days x vol_scale
NEUTRAL:           everything else              ->  deploy C/base_days
WAITING:           before adaptive_start        ->  no deployment
DEPLETED:          cash exhausted               ->  no deployment
```

---

## Project Structure

```
quant-mean-reversion-engine-v3.3/
├── docs/
│   ├── ARCHITECTURE.md
│   ├── CONTRIBUTING.md
│   └── SETUP.md
├── samples/                          # Sample output charts and reports
├── mean_reversion_standalone.py
├── test_backtester.py
├── aggregate_confidence_bins.py
├── requirements.txt
└── README.md
```

---

## Known Limitations

- Long-only: no short strategies
- Single-asset per run (no cross-asset portfolio mode)
- Daily bars only — intraday not tested
- Assumes perfect liquidity at open/close
- Fixed slippage + flat commission (no bid-ask spread model)

---

## License

MIT — see [LICENSE](LICENSE) for details.

## Contact

**Chau Le** — chau.le@marquette.edu
[LinkedIn](https://www.linkedin.com/in/lechau1801/) | [GitHub](https://github.com/chaulemuoichin)

---

*Educational purposes only. Not financial advice. Past performance does not guarantee future results.*
