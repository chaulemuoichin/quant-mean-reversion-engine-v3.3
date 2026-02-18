# Quantitative Mean Reversion Engine v3.3

A sophisticated **long-only mean reversion trading strategy backtester** featuring a two-layer portfolio architecture (core + tactical sleeves), adaptive regime classification, volatility-adjusted position sizing, and comprehensive statistical validation.

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub release](https://img.shields.io/github/v/release/chaulemuoichin/quant-mean-reversion-engine-v3.3)](https://github.com/chaulemuoichin/quant-mean-reversion-engine-v3.3/releases)
[![GitHub stars](https://img.shields.io/github/stars/chaulemuoichin/quant-mean-reversion-engine-v3.3)](https://github.com/chaulemuoichin/quant-mean-reversion-engine-v3.3/stargazers)

## Overview

This backtester implements an advanced mean reversion trading system that combines statistical testing, machine learning-inspired regime detection, and quantitative risk management. Built for equity pairs trading and long-only strategies, it features real-time regime adaptation, multi-level exit strategies, and production-ready performance analytics.

**Key Innovation:** Two-layer portfolio architecture separating core (buy-and-hold) and tactical (mean reversion) allocations with optional DCA or adaptive market-state deployment strategies.

## Key Features

### Portfolio Architecture
- **Two-Layer Design**: Independent core (70-80%) and tactical (20-30%) sleeves with automated rebalancing
- **Adaptive Core Deployment**: State-based capital deployment (CALM_UPTREND, PULLBACK, HIGH_VOL_DRAWDOWN, NEUTRAL)
- **DCA Support**: Dollar-cost averaging with configurable slippage and commission modeling
- **Cash Yield**: Daily compounding interest on idle capital

### Regime Classification
- **Multi-Factor Analysis**: Combines Hurst exponent, ADF stationarity test, half-life, and variance ratio
- **Adaptive Thresholds**: Dynamic regime boundaries based on historical distribution
- **Per-Bar Labeling**: Real-time regime assignment (MEAN_REVERTING, SIDEWAYS, TRENDING, AMBIGUOUS)
- **Configurable Frequency**: Computationally expensive tests run every N bars (default: 5)

### Signal Generation & Execution
- **Z-Score Based Entry/Exit**: Price-to-SMA200 ratio with rolling lookback normalization
- **Staged Exits**: Multi-level trim thresholds [(z>=0.5: 25%), (z>=1.0: 50%), (z>=2.0: 100%)]
- **Reversal Confirmation**: RSI turning points + close-above-prior validation
- **Quality Filters**: Minimum trade notional ($1,000) and minimum shares (5)

### Risk Management
- **Volatility-Adjusted Sizing**: Risk-parity inspired position scaling (0.5x-1.5x multipliers)
- **ATR-Based Stops**: Dynamic stop-loss at 3x ATR with intraday gap-down handling
- **Thesis Break Detection**: Consecutive negative SMA slope triggers early exit
- **Max Holding Period**: 30-day automatic liquidation timer

### Advanced Features
- **Confidence-Weighted Sizing**: Position size scales with regime classification confidence (60-100% range)
- **Cost-Aware Entry Gate**: Expected return vs. transaction cost hurdle rate
- **Better Exits**: Time-decay and adverse-z exit logic (min 3 days, max 30 days)
- **Tactical Vol Targeting**: Realized volatility targeting at 15% annualized

### Performance Analytics
- **Comprehensive Metrics**: Sharpe, Sortino, Calmar, profit factor, max drawdown
- **Capture Ratios**: Upside/downside capture vs buy-and-hold benchmark
- **Return Attribution**: Decompose total return into core (beta) vs tactical (alpha) contributions
- **Regime Performance**: Breakdown of P&L by regime state
- **Exit Reason Analysis**: Track stop-loss, target, time-decay, and thesis-break exits

### Visualization & Reporting
- **Equity Curves**: Total, core, tactical, and baseline comparison plots
- **Drawdown Analysis**: Underwater plots with peak-to-trough annotations
- **Exposure Tracking**: Position sizing over time scatter plots
- **Confidence Bins**: Trade performance stratified by entry confidence (0.02 intervals)
- **CSV Export**: Full trade ledger with entry confidence, realized P&L, hold days

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/chaulemuoichin/quant-mean-reversion-engine-v3.3.git
cd mean-reversion-backtester

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```bash
# Single-ticker backtest with default settings
python mean_reversion_standalone.py AAPL

# Two-layer portfolio mode with visualization
python mean_reversion_standalone.py AAPL --two-layer --plot

# Custom allocation and cash yield
python mean_reversion_standalone.py NVDA --two-layer --core-pct 0.7 --tactical-pct 0.3 --cash-yield 4.5

# Adaptive core deployment
python mean_reversion_standalone.py GOOG --two-layer --core-entry adaptive --plot

# DCA core deployment over 60 days
python mean_reversion_standalone.py MSFT --two-layer --core-entry dca --dca-days 60

# Disable regime filter (baseline comparison)
python mean_reversion_standalone.py TSLA --no-regime --entry-at same_close

# Debug mode with extended logs
python mean_reversion_standalone.py META --debug --window 30
```

## Project Structure

```
quant-mean-reversion-engine-v3.3/
├── docs/                             
│   ├── ARCHITECTURE.md
│   ├── CONTRIBUTING.md
│   └── SETUP.md
├── samples/
│   ├── 01_NVDA_TL_mc0_20260218_011958.txt
│   └── 01_NVDA_TL_mc060_20260218_012003.txt
├── .gitattributes
├── .gitignore
├── LICENSE
├── README.md
├── aggregate_confidence_bins.py
├── mean_reversion_standalone.py
├── requirements.txt
└── test_backtester.py
```

## Testing

The project includes 50+ unit tests covering:
- Statistical test accuracy (Hurst, ADF, half-life)
- Regime classification logic (adaptive thresholds, distribution)
- Signal generation (z-score, staged exits, dust filters)
- Position sizing (vol-adjusted, confidence-weighted)
- Execution simulation (slippage, commission, stop fills)
- Two-layer accounting (capital conservation, no lookahead)
- Adaptive deployment (monotonic risk, drawdown acceleration)

```bash
# Run all tests
python -m unittest test_backtester -v

# Run specific test suite
python -m unittest test_backtester.TestAdaptiveCoreDeployment -v
```

## Configuration

### Core Parameters

```python
MeanReversionConfig(
    # Portfolio allocation
    starting_capital=100_000.0,
    core_allocation_pct=0.80,          # Core sleeve: 80%
    tactical_allocation_pct=0.20,      # Tactical sleeve: 20%
    core_entry_mode="adaptive",        # instant | dca | adaptive
    
    # Signal thresholds (z-score)
    entry_z=-1.5,                      # Enter at -1.5σ
    add_z=-2.0,                        # Add at -2.0σ
    trim_levels=[                       # Staged exits
        (0.5, 0.25),                   # Trim 25% at +0.5σ
        (1.0, 0.50),                   # Trim 50% at +1.0σ
        (2.0, 1.00),                   # Exit 100% at +2.0σ
    ],
    
    # Position sizing
    max_position_pct=0.15,             # Max 15% allocation
    vol_adjust_sizing=True,            # Vol-parity scaling
    confidence_sizing_enabled=True,    # Confidence-weighted sizing
    
    # Risk management
    stop_atr_multiple=3.0,             # Stop at -3x ATR
    max_holding_days=30,               # Force exit after 30 days
    thesis_break_sma_bars=10,          # 10-bar SMA slope check
    
    # Regime classification
    regime_filter_enabled=True,
    allowed_regimes=["MEAN_REVERTING", "SIDEWAYS"],
    regime_adaptive_thresholds=True,
    
    # Execution
    entry_at="next_open",              # next_open | same_close
    slippage_pct=0.0005,               # 5 bps
    commission_per_trade=1.0,          # $1 per fill
    cash_yield_annual_pct=4.5,         # 4.5% APY on idle cash
)
```

## Sample Output

**Command:** `python mean_reversion_standalone.py NVDA --two-layer --core-entry adaptive --plot --period 5y`

### Equity Curves (NVDA, 5y, adaptive)
<img width="750" height="375" alt="NVDA_benchmark" src="https://github.com/user-attachments/assets/0af442f7-0cdc-40b2-bda1-a95c38881c3c" />

### Return Contribution
<img width="750" height="375" alt="NVDA_contribution" src="https://github.com/user-attachments/assets/2e88d45d-3dce-4fb1-9f4e-07e5cbee79e4" />

### Drawdown vs. Baseline
<img width="750" height="262" alt="NVDA_drawdown" src="https://github.com/user-attachments/assets/eccccdc2-668f-4408-bc64-45b2982423c0" />

### Exposure vs. Price
<img width="600" height="375" alt="NVDA_exposure" src="https://github.com/user-attachments/assets/0baccd61-e8a5-4cfa-946f-93cd12d68528" />

### Underwater Chart
<img width="750" height="262" alt="NVDA_underwater" src="https://github.com/user-attachments/assets/81b06430-c2a1-4aa1-a972-18381859f2de" />

### Performance Report
```
==========================================================================
           MEAN REVERSION BACKTESTER v3.3 - TWO-LAYER PORTFOLIO           
NVDA -- NVIDIA Corporation
==========================================================================
Capital   : $100,000  (Core 80% = $80,000  |  Tactical 20% = $20,000)
Core Entry: Adaptive (base=60, slow=120, fast=40, vol_target=12%, dd_window=252)

==========================================================================
TOTAL PORTFOLIO PERFORMANCE (Core + Tactical)
==========================================================================
  Starting Capital : $  100,000.00
  Ending Capital   : $  832,124.90
  Total Return     :    +732.12%
  Sharpe Ratio     :      1.262
  Max Drawdown     :     -53.91%

==========================================================================
SIDE-BY-SIDE COMPARISON
==========================================================================
  Metric                        Total     Baseline     Tactical         Core
  ------------------------------------------------------------
  Return %                   +732.12%     +920.69%     +553.40%     +776.81%
  Sharpe                        1.262        1.230        0.812        1.117
  Max DD %                     -53.91       -59.71       -39.89       -60.90

--------------------------------------------------------------------------
RETURN ATTRIBUTION
--------------------------------------------------------------------------
  Core (beta)      :   +84.9%  ($ +621,444.09)
  Tactical (alpha) :   +15.1%  ($ +110,680.81)

--------------------------------------------------------------------------
TACTICAL LAYER DIAGNOSTICS
--------------------------------------------------------------------------
  Time in Market   : 9.3%
  Avg Exposure     : 0.6%
  Blocked Rate     : 88.5%
  blocked_by_confidence : 23
  blocked_by_regime     : 23
```

## Advanced Usage

### Aggregating Multi-Run Results

```bash
# Run multiple tickers
for ticker in AAPL NVDA GOOG MSFT; do
    python mean_reversion_standalone.py $ticker --two-layer --plot
done

# Aggregate confidence bins across all runs
python aggregate_confidence_bins.py --reports-dir reports --low-sample-n 10
```

### Custom Strategy Development

```python
from mean_reversion_standalone import (
    MeanReversionConfig,
    TwoLayerPortfolioEngine,
    fetch_data,
)

# Fetch data
df, _ = fetch_data("AAPL", period="2y", interval="1d")

# Configure strategy
config = MeanReversionConfig(
    starting_capital=100_000,
    core_allocation_pct=0.70,
    tactical_allocation_pct=0.30,
    entry_z=-1.8,
    # ... customize parameters
)

# Run backtest
strategy = MeanReversionStrategy(config)
ratio_z = strategy.compute_ratio_z(strategy.compute_ratio(df))
regime_labels, regime_scores = strategy.classify_regime(df)

engine = TwoLayerPortfolioEngine(config)
results = engine.run(df, ratio_z, regime_labels, regime_scores)

print(f"Final Equity: ${results['total_equity_curve'][-1]:,.2f}")
```

## Technical Documentation

### Statistical Tests

1. **Hurst Exponent** (R/S method, R² >= 0.80 quality gate)
   - H < 0.5: Mean reverting
   - H ~= 0.5: Random walk
   - H > 0.5: Trending
   - Clamped to [0, 1] range when out-of-bounds

2. **ADF Test** (Augmented Dickey-Fuller)
   - p < 0.05: Stationary (mean reverting)
   - Returns dict with {adf_stat, p_value, is_stationary, note}

3. **Half-Life** (OU process estimation)
   - lambda from dy[t] = lambda*(mu - y[t-1]) + epsilon
   - Half-life = ln(2) / lambda days

4. **Variance Ratio** (Lo-MacKinlay test)
   - VR(q) = Var[q-day] / (q x Var[1-day])
   - VR < 1: Mean reversion
   - VR > 1: Momentum

### Adaptive Core Deployment States

```
CALM_UPTREND:      trend=1, dd<10%, vol<target  ->  deploy C/slow_days
PULLBACK:          trend=0, 5%<=dd<=20%         ->  deploy C/fast_days
HIGH_VOL_DRAWDOWN: dd>=15% or vol>=target       ->  deploy C/fast_days x vol_scale
NEUTRAL:           everything else              ->  deploy C/base_days
WAITING:           before adaptive_start        ->  no deployment
DEPLETED:          cash exhausted               ->  no deployment
```

### Regime Classification Logic

Regime score = weighted average of:
- Hurst in [0, 0.45]: +1 (mean reversion)
- ADF p-value < 0.05: +1 (stationary)
- Half-life in [5, 60] days: +1 (tractable)
- Variance ratio < 0.80: +1 (mean reversion)

Adaptive thresholds (10th/90th percentiles of rolling scores):
- Score > p90: MEAN_REVERTING
- p10 < Score < p90: SIDEWAYS
- Score < p10: TRENDING
- Insufficient data: AMBIGUOUS

## Known Limitations

1. **Long-Only**: No short selling or short strategies
2. **Single Asset**: Designed for single-ticker analysis (portfolio mode uses sequential single-ticker runs)
3. **Daily Bars Only**: No intraday support (intervals < 1d not tested)
4. **No Order Book**: Assumes perfect liquidity at open/close
5. **Simplified Costs**: Fixed slippage + flat commission (no bid-ask spread model)

## Citation

If you use this backtester in your research, please cite:

```bibtex
@software{mean_reversion_backtester_2026,
  author = {Le, Chau},
  title = {Mean Reversion Backtester v3.3: Two-Layer Portfolio Engine},
  year = {2026},
  url = {https://github.com/chaulemuoichin/quant-mean-reversion-engine-v3.3}
}
```

## Contributing

Contributions welcome! Areas for enhancement:
- Multi-asset portfolio mode with correlation analysis
- Machine learning regime classifiers (HMM, LSTM)
- Kalman filter for dynamic beta estimation
- Options overlay strategies (covered calls, protective puts)
- Walk-forward optimization framework

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- **statsmodels**: ADF test implementation
- **yfinance**: Historical market data API
- **pandas/numpy**: Core data processing
- **matplotlib**: Visualization engine

## Contact

**Chau Le**
chau.le@marquette.edu
[LinkedIn](https://www.linkedin.com/in/lechau1801/) | [GitHub](https://github.com/chaulemuoichin)

---

*Disclaimer: This software is for educational and research purposes only. Past performance does not guarantee future results. Trading involves substantial risk of loss. Always conduct your own due diligence before investing.*
