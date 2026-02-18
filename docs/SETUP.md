# Setup & Deployment Guide

## Prerequisites

- **Python**: 3.8 or higher
- **Operating System**: Linux, macOS, or Windows
- **Memory**: Minimum 4GB RAM
- **Storage**: 500MB for code and dependencies

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/mean-reversion-backtester.git
cd mean-reversion-backtester
```

### 2. Create Virtual Environment

**Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Verify Installation:**
```bash
python -c "import numpy, pandas, yfinance, statsmodels, matplotlib; print('All dependencies OK')"
```

## Quick Start

### Single-Ticker Backtest

```bash
# Basic usage
python mean_reversion_standalone.py AAPL

# With custom date range
python mean_reversion_standalone.py AAPL --start-date 2022-01-01 --end-date 2024-12-31

# Debug mode
python mean_reversion_standalone.py AAPL --debug
```

### Two-Layer Portfolio Mode

```bash
# Default 80/20 allocation
python mean_reversion_standalone.py AAPL --two-layer --plot

# Custom allocation
python mean_reversion_standalone.py NVDA --two-layer --core-pct 0.70 --tactical-pct 0.30 --plot

# Adaptive core deployment
python mean_reversion_standalone.py GOOG --two-layer --core-entry adaptive --plot

# DCA deployment over 60 days
python mean_reversion_standalone.py MSFT --two-layer --core-entry dca --dca-days 60 --plot
```

## Configuration

### Command-Line Arguments

```bash
# Core portfolio settings
--starting-capital 100000        # Starting capital in dollars
--core-pct 0.80                 # Core allocation percentage
--tactical-pct 0.20             # Tactical allocation percentage
--core-entry {instant|dca|adaptive}  # Core entry mode
--cash-yield 4.5                # Annual cash yield percentage

# DCA settings (if --core-entry dca)
--dca-start 0                   # Start DCA at bar N
--dca-days 60                   # DCA over N days
--dca-slippage 0.0005           # DCA slippage percentage
--dca-commission 1.0            # DCA commission per trade

# Adaptive settings (if --core-entry adaptive)
--adaptive-start 0              # Start adaptive at bar N
--adaptive-base-days 60         # NEUTRAL state: C/base_days
--adaptive-slow-days 120        # CALM_UPTREND: C/slow_days
--adaptive-fast-days 40         # PULLBACK/HIGH_VOL: C/fast_days
--adaptive-vol-target 0.12      # Volatility target (12%)
--adaptive-vol-floor 0.05       # Vol floor for calculations
--adaptive-max-deploy-pct 0.10  # Max deploy per bar (10% of remaining)
--adaptive-min-deploy-pct 0.002 # Min deploy per bar (0.2% of remaining)

# Signal thresholds
--entry-z -1.5                  # Entry z-score threshold
--add-z -2.0                    # Add z-score threshold
--trim-z 1.0                    # Trim z-score threshold (legacy)
--sell-z 2.0                    # Sell z-score threshold (legacy)

# Position sizing
--max-position-pct 0.15         # Max position size (15% of capital)
--vol-adjust                    # Enable volatility-adjusted sizing
--confidence-sizing             # Enable confidence-weighted sizing

# Risk management
--stop-atr-multiple 3.0         # Stop at entry - (3.0 × ATR)
--max-holding-days 30           # Force exit after 30 days
--thesis-break-bars 10          # SMA slope check over N bars

# Regime classification
--no-regime                     # Disable regime filter (baseline)
--regime-adaptive               # Use adaptive thresholds
--regime-update-freq 5          # Run regime tests every N bars

# Execution
--entry-at {next_open|same_close}  # Entry timing
--slippage-pct 0.0005           # Slippage (5 bps)
--commission 1.0                # Commission per trade

# Rebalancing (two-layer only)
--rebalance-freq {M|Q|A}        # Monthly/Quarterly/Annual
--rebalance-drift-threshold 0.05 # Trigger rebalance if drift > 5%

# Output
--plot                          # Generate PNG charts
--reports-dir reports           # Output directory
--debug                         # Enable debug logging
```

### Programmatic Configuration

```python
from mean_reversion_standalone import MeanReversionConfig

config = MeanReversionConfig(
    # Portfolio
    starting_capital=100_000.0,
    two_layer_mode=True,
    core_allocation_pct=0.80,
    tactical_allocation_pct=0.20,
    core_entry_mode="adaptive",
    
    # Adaptive core parameters
    core_adaptive_start=0,
    core_adaptive_base_days=60,
    core_adaptive_slow_days=120,
    core_adaptive_fast_days=40,
    core_adaptive_vol_target=0.12,
    core_adaptive_vol_floor=0.05,
    core_adaptive_max_deploy_pct=0.10,
    core_adaptive_min_deploy_pct=0.002,
    
    # Signal thresholds
    entry_z=-1.5,
    add_z=-2.0,
    trim_levels=[
        (0.5, 0.25),   # Trim 25% at z≥0.5
        (1.0, 0.50),   # Trim 50% at z≥1.0
        (2.0, 1.00),   # Exit 100% at z≥2.0
    ],
    
    # Position sizing
    max_position_pct=0.15,
    vol_adjust_sizing=True,
    confidence_sizing_enabled=True,
    confidence_c0=0.60,
    confidence_gamma=1.0,
    
    # Risk management
    stop_atr_multiple=3.0,
    max_holding_days=30,
    thesis_break_sma_bars=10,
    
    # Regime
    regime_filter_enabled=True,
    regime_adaptive_thresholds=True,
    regime_update_freq=5,
    allowed_regimes=["MEAN_REVERTING", "SIDEWAYS"],
    
    # Execution
    entry_at="next_open",
    slippage_pct=0.0005,
    commission_per_trade=1.0,
    cash_yield_annual_pct=4.5,
    
    # Rebalancing
    rebalance_freq="Q",  # Quarterly
    rebalance_drift_threshold=0.05,
)

config.validate()  # Check for errors
```

## Running Tests

### All Tests
```bash
python -m unittest test_backtester -v
```

### Specific Test Suite
```bash
# Statistical tests
python -m unittest test_backtester.TestStatisticalTests -v

# Regime classification
python -m unittest test_backtester.TestRegimeClassification -v

# Two-layer portfolio
python -m unittest test_backtester.TestTwoLayerPortfolio -v

# Adaptive deployment
python -m unittest test_backtester.TestAdaptiveCoreDeployment -v
```

### Coverage Report
```bash
pip install pytest pytest-cov
pytest test_backtester.py --cov=mean_reversion_standalone --cov-report=html
```

## Multi-Ticker Workflows

### Batch Processing

**Shell Script** (`backtest_batch.sh`):
```bash
#!/bin/bash

TICKERS=("AAPL" "NVDA" "GOOG" "MSFT" "TSLA" "META")

for ticker in "${TICKERS[@]}"; do
    echo "Processing $ticker..."
    python mean_reversion_standalone.py $ticker \
        --two-layer \
        --core-entry adaptive \
        --plot \
        --reports-dir reports
done

echo "Aggregating results..."
python aggregate_confidence_bins.py --reports-dir reports
```

### Python Wrapper

```python
import subprocess
from pathlib import Path

tickers = ["AAPL", "NVDA", "GOOG", "MSFT", "TSLA", "META"]
reports_dir = Path("reports")

for ticker in tickers:
    print(f"Processing {ticker}...")
    cmd = [
        "python", "mean_reversion_standalone.py", ticker,
        "--two-layer",
        "--core-entry", "adaptive",
        "--plot",
        "--reports-dir", str(reports_dir),
    ]
    subprocess.run(cmd, check=True)

print("Aggregating results...")
subprocess.run([
    "python", "aggregate_confidence_bins.py",
    "--reports-dir", str(reports_dir),
], check=True)
```

## Output Files

### Reports Directory Structure

```
reports/
├── AAPL_20260215_123045_TRADES_v3.3.csv
├── AAPL_20260215_123045_curves.csv
├── AAPL_20260215_123045_benchmark.png
├── AAPL_20260215_123045_drawdown.png
├── NVDA_20260215_130122_TRADES_v3.3.csv
├── NVDA_20260215_130122_curves.csv
├── NVDA_20260215_130122_benchmark.png
├── NVDA_20260215_130122_drawdown.png
└── aggregated_bins_20260215_131500.csv
```

### Trade Ledger Format

**File**: `TICKER_DATE_TRADES_v3.3.csv`

```csv
date,action,regime,entry_confidence,shares,price,realized_pnl,hold_days,reason
2024-01-15,BUY,MEAN_REVERTING,0.73,50,145.23,0.00,0,
2024-01-22,REDUCE,SIDEWAYS,0.73,12,148.90,44.04,7,TRIM_Z_0.5
2024-01-29,SELL,TRENDING,0.73,38,151.45,236.36,14,EXIT_Z_2.0
```

### Equity Curves Format

**File**: `TICKER_DATE_curves.csv`

```csv
date,total_equity,core_equity,tactical_equity,baseline_static,baseline_matched
2024-01-01,100000.00,80000.00,20000.00,100000.00,100000.00
2024-01-02,100123.45,80098.76,20024.69,100087.32,100089.12
2024-01-03,100234.56,80187.65,20046.91,100165.43,100178.23
```

### Aggregated Bins Format

**File**: `aggregated_bins_DATE.csv`

```csv
range,bin,bin_lo,bin_hi,n_trades,win_rate,avg_pnl,median_pnl,avg_hold_days,low_sample
0.50_to_1.00,[0.50,0.52),0.50,0.52,3,33.33,-12.45,-8.23,6.7,TRUE
0.50_to_1.00,[0.52,0.54),0.52,0.54,8,62.50,18.76,15.34,7.2,FALSE
0.50_to_1.00,[0.54,0.56),0.54,0.56,12,75.00,24.89,22.11,6.9,FALSE
```

## Troubleshooting

### Issue: yfinance download fails

**Symptoms**:
```
Error: Failed to download AAPL: HTTPError 429
```

**Solution**:
```python
# Add retry logic or delay between downloads
import time
for ticker in tickers:
    time.sleep(1)  # Rate limit: 1 request/second
    subprocess.run(["python", "mean_reversion_standalone.py", ticker])
```

### Issue: statsmodels not found

**Symptoms**:
```
Warning: statsmodels not installed
ADF p-value: N/A [statsmodels_not_installed]
```

**Solution**:
```bash
pip install statsmodels
```

### Issue: matplotlib backend error

**Symptoms**:
```
RuntimeError: Invalid DISPLAY variable
```

**Solution** (headless server):
```python
# Add before importing matplotlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
```

### Issue: memory error on long backtests

**Symptoms**:
```
MemoryError: Unable to allocate array
```

**Solution**:
- Reduce data range: `--start-date 2022-01-01`
- Use 1d interval only (no intraday)
- Disable plots: remove `--plot` flag

### Issue: accounting identity failed

**Symptoms**:
```
Test failed: Accounting identity failed at bar 142
```

**Solution**:
- This is a serious bug. Please file an issue with:
  - Full error traceback
  - Synthetic data parameters
  - Config settings used

## Performance Tips

### Speed Optimization

1. **Reduce Regime Update Frequency**:
```bash
--regime-update-freq 10  # Run expensive tests every 10 bars (default: 5)
```

2. **Disable Plots**:
```bash
# Remove --plot flag (saves 2-3 seconds per run)
python mean_reversion_standalone.py AAPL --two-layer
```

3. **Use Instant Core Entry**:
```bash
# Adaptive/DCA modes iterate bar-by-bar; instant is faster
--core-entry instant
```

### Memory Optimization

1. **Limit Data Range**:
```bash
--start-date 2023-01-01 --end-date 2024-12-31  # 2 years max
```

2. **Use CSV Output Only**:
```bash
# Avoid matplotlib memory overhead
python mean_reversion_standalone.py AAPL --two-layer > output.txt
```

## Production Deployment

### Scheduled Backtests

**Cron Job** (daily 6 AM):
```cron
0 6 * * * cd /path/to/repo && /path/to/venv/bin/python mean_reversion_standalone.py AAPL --two-layer --plot >> logs/backtest.log 2>&1
```

### Docker Container

**Dockerfile**:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "mean_reversion_standalone.py", "AAPL", "--two-layer", "--plot"]
```

**Build & Run**:
```bash
docker build -t mean-reversion-backtester .
docker run -v $(pwd)/reports:/app/reports mean-reversion-backtester
```

## Next Steps

1. **Explore Examples**: See `test_backtester.py` for usage patterns
2. **Read Architecture**: `ARCHITECTURE.md` for technical deep-dive
3. **Contribute**: `CONTRIBUTING.md` for development guidelines
4. **Ask Questions**: Open an issue on GitHub

## Support

- **Issues**: https://github.com/yourusername/mean-reversion-backtester/issues
- **Email**: chau.le@marquette.edu
- **LinkedIn**: https://www.linkedin.com/in/your-profile
