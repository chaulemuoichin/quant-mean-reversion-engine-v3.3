# Mean Reversion Backtester v3.3

A sophisticated **long-only mean reversion trading strategy backtester** featuring a two-layer portfolio architecture (core + tactical sleeves), adaptive regime classification, volatility-adjusted position sizing, and comprehensive statistical validation.

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub release](https://img.shields.io/github/v/release/chaulemuoichin/quant-mean-reversion-engine-v3.3)](https://github.com/chaulemuoichin/quant-mean-reversion-engine-v3.3/releases)

## 🎯 Overview

This backtester implements an advanced mean reversion trading system that combines statistical testing, machine learning-inspired regime detection, and quantitative risk management. Built for equity pairs trading and long-only strategies, it features real-time regime adaptation, multi-level exit strategies, and production-ready performance analytics.

**Key Innovation:** Two-layer portfolio architecture separating core (buy-and-hold) and tactical (mean reversion) allocations with optional DCA or adaptive market-state deployment strategies.

## ✨ Key Features

### 📊 Portfolio Architecture
- **Two-Layer Design**: Independent core (70-80%) and tactical (20-30%) sleeves with automated rebalancing
- **Adaptive Core Deployment**: State-based capital deployment (CALM_UPTREND, PULLBACK, HIGH_VOL_DRAWDOWN, NEUTRAL)
- **DCA Support**: Dollar-cost averaging with configurable slippage and commission modeling
- **Cash Yield**: Daily compounding interest on idle capital

### 🧠 Regime Classification
- **Multi-Factor Analysis**: Combines Hurst exponent, ADF stationarity test, half-life, and variance ratio
- **Adaptive Thresholds**: Dynamic regime boundaries based on historical distribution
- **Per-Bar Labeling**: Real-time regime assignment (MEAN_REVERTING, SIDEWAYS, TRENDING, AMBIGUOUS)
- **Configurable Frequency**: Computationally expensive tests run every N bars (default: 5)

### 📈 Signal Generation & Execution
- **Z-Score Based Entry/Exit**: Price-to-SMA200 ratio with rolling lookback normalization
- **Staged Exits**: Multi-level trim thresholds [(z≥0.5: 25%), (z≥1.0: 50%), (z≥2.0: 100%)]
- **Reversal Confirmation**: RSI turning points + close-above-prior validation
- **Quality Filters**: Minimum trade notional ($1,000) and minimum shares (5)

### 🎯 Risk Management
- **Volatility-Adjusted Sizing**: Risk-parity inspired position scaling (0.5x-1.5x multipliers)
- **ATR-Based Stops**: Dynamic stop-loss at 3x ATR with intraday gap-down handling
- **Thesis Break Detection**: Consecutive negative SMA slope triggers early exit
- **Max Holding Period**: 30-day automatic liquidation timer

### 🔬 Advanced Features
- **Confidence-Weighted Sizing**: Position size scales with regime classification confidence (60-100% range)
- **Cost-Aware Entry Gate**: Expected return vs. transaction cost hurdle rate
- **Better Exits**: Time-decay and adverse-z exit logic (min 3 days, max 30 days)
- **Tactical Vol Targeting**: Realized volatility targeting at 15% annualized

### 📉 Performance Analytics
- **Comprehensive Metrics**: Sharpe, Sortino, Calmar, profit factor, max drawdown
- **Capture Ratios**: Upside/downside capture vs buy-and-hold benchmark
- **Return Attribution**: Decompose total return into core (beta) vs tactical (alpha) contributions
- **Regime Performance**: Breakdown of P&L by regime state
- **Exit Reason Analysis**: Track stop-loss, target, time-decay, and thesis-break exits

### 📊 Visualization & Reporting
- **Equity Curves**: Total, core, tactical, and baseline comparison plots
- **Drawdown Analysis**: Underwater plots with peak-to-trough annotations
- **Exposure Tracking**: Position sizing over time scatter plots
- **Confidence Bins**: Trade performance stratified by entry confidence (0.02 intervals)
- **CSV Export**: Full trade ledger with entry confidence, realized P&L, hold days

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/mean-reversion-backtester.git
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

## 📁 Project Structure

```
mean-reversion-backtester/
├── mean_reversion_standalone.py    # Main backtester (6600+ lines)
├── test_backtester.py              # Comprehensive test suite (4400+ lines)
├── aggregate_confidence_bins.py    # Post-run aggregation utility
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── reports/                        # Output directory (auto-created)
│   ├── TICKER_DATE_TRADES_*.csv   # Trade ledgers
│   ├── TICKER_DATE_curves.csv     # Equity curves
│   ├── TICKER_DATE_benchmark.png  # Comparative charts
│   └── aggregated_bins_*.csv      # Pooled confidence analysis
└── tests/
    └── fixtures/                   # Synthetic data generators
```

## 🧪 Testing

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

## ⚙️ Configuration

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

## 📊 Output Examples

### Trade Ledger (CSV)
```
date,action,regime,entry_confidence,shares,price,realized_pnl,hold_days,reason
2024-01-15,BUY,MEAN_REVERTING,0.73,50,145.23,0.00,0,
2024-01-22,REDUCE,SIDEWAYS,0.73,12,148.90,44.04,7,TRIM_Z_0.5
2024-01-29,SELL,TRENDING,0.73,38,151.45,236.36,14,EXIT_Z_2.0
```

### Performance Report
```
===============================================================================
BACKTEST PERFORMANCE (Two-Layer: Core 80% + Tactical 20%)
===============================================================================
Starting Capital     : $  100,000.00
Ending Capital       : $  112,450.30
Total Return         :     +12.45%
Annualized Return    :     +14.32%
Sharpe Ratio         :       1.42
Sortino Ratio        :       1.89
Calmar Ratio         :       2.31
Max Drawdown         :      -6.20%
Profit Factor        :       2.15
Win Rate             :      64.3% (27/42 trades)
Avg Hold Days        :      8.5 days

RETURN ATTRIBUTION
Core (Beta)          :  +8,450.20 (68.9%)
Tactical (Alpha)     :  +3,800.10 (31.1%)

CAPTURE RATIOS
Upside Capture       :     112.3%
Downside Capture     :      68.7%
Capture Ratio        :       1.63

TACTICAL DIAGNOSTICS
Time in Market       :      34.2%
Avg Exposure         :       8.7%
Blocked Signal Rate  :      18.5%
Accrued Cash Yield   :    $127.45
```

## 🔧 Advanced Usage

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

## 📚 Technical Documentation

### Statistical Tests

1. **Hurst Exponent** (R/S method, R² ≥ 0.80 quality gate)
   - H < 0.5: Mean reverting
   - H ≈ 0.5: Random walk
   - H > 0.5: Trending
   - Clamped to [0, 1] range when out-of-bounds

2. **ADF Test** (Augmented Dickey-Fuller)
   - p < 0.05: Stationary (mean reverting)
   - Returns dict with {adf_stat, p_value, is_stationary, note}

3. **Half-Life** (OU process estimation)
   - λ from Δy[t] = λ·(μ - y[t-1]) + ε
   - Half-life = ln(2) / λ days

4. **Variance Ratio** (Lo-MacKinlay test)
   - VR(q) = Var[q-day] / (q × Var[1-day])
   - VR < 1: Mean reversion
   - VR > 1: Momentum

### Adaptive Core Deployment States

```
CALM_UPTREND:      trend=1, dd<10%, σ<target  →  deploy C/slow_days
PULLBACK:          trend=0, 5%≤dd≤20%         →  deploy C/fast_days
HIGH_VOL_DRAWDOWN: dd≥15% or σ≥target         →  deploy C/fast_days × vol_scale
NEUTRAL:           everything else             →  deploy C/base_days
WAITING:           before adaptive_start       →  no deployment
DEPLETED:          cash exhausted              →  no deployment
```

### Regime Classification Logic

Regime score = weighted average of:
- Hurst ∈ [0, 0.45]: +1 (mean reversion)
- ADF p-value < 0.05: +1 (stationary)
- Half-life ∈ [5, 60] days: +1 (tractable)
- Variance ratio < 0.80: +1 (mean reversion)

Adaptive thresholds (10th/90th percentiles of rolling scores):
- Score > p90: MEAN_REVERTING
- p10 < Score < p90: SIDEWAYS
- Score < p10: TRENDING
- Insufficient data: AMBIGUOUS

## 🐛 Known Limitations

1. **Long-Only**: No short selling or short strategies
2. **Single Asset**: Designed for single-ticker analysis (portfolio mode uses sequential single-ticker runs)
3. **Daily Bars Only**: No intraday support (intervals < 1d not tested)
4. **No Order Book**: Assumes perfect liquidity at open/close
5. **Simplified Costs**: Fixed slippage + flat commission (no bid-ask spread model)

## 📝 Citation

If you use this backtester in your research, please cite:

```bibtex
@software{mean_reversion_backtester_2026,
  author = {Le, Chau},
  title = {Mean Reversion Backtester v3.3: Two-Layer Portfolio Engine},
  year = {2026},
  url = {https://github.com/chaulemuoichin/quant-mean-reversion-engine-v3.3}
}
```

## 🤝 Contributing

Contributions welcome! Areas for enhancement:
- Multi-asset portfolio mode with correlation analysis
- Machine learning regime classifiers (HMM, LSTM)
- Kalman filter for dynamic beta estimation
- Options overlay strategies (covered calls, protective puts)
- Walk-forward optimization framework

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- **statsmodels**: ADF test implementation
- **yfinance**: Historical market data API
- **pandas/numpy**: Core data processing
- **matplotlib**: Visualization engine

## 📧 Contact

**Chau Le**  
📧 chau.le@marquette.edu  
🔗 [LinkedIn](https://www.linkedin.com/in/lechau1801/) | [GitHub](https://github.com/chaulemuoichin)

---

**Disclaimer**: This software is for educational and research purposes only. Past performance does not guarantee future results. Trading involves substantial risk of loss. Always conduct your own due diligence before investing.
