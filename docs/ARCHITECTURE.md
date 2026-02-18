# Architecture Documentation

## System Overview

The Mean Reversion Backtester v3.3 is a production-grade event-driven backtesting system implementing a two-layer portfolio architecture with statistical regime classification and adaptive position sizing.

## Core Components

### 1. Configuration System (`MeanReversionConfig`)

**Purpose**: Centralized parameter management with validation and immutability.

**Key Features**:
- 60+ configurable parameters
- Built-in validation in `validate()` method
- Immutable copy-with-override via `copy_with(**overrides)`
- Grouped parameters: portfolio, signals, regime, execution, risk

**Design Pattern**: Configuration object pattern with builder capabilities.

### 2. Data Pipeline

```
yfinance → normalize_dataframe() → price/volume scaling → feature engineering
```

**Steps**:
1. **Download**: `fetch_data()` pulls OHLCV from yfinance
2. **Normalization**: Timezone-aware index, sorted chronologically
3. **Adj Close Scaling**: OHLC scaled to adjusted close basis, volume inverse-scaled
4. **Feature Engineering**: Calculate returns, log returns, volume ratios

**Quality Gates**:
- Duplicate timestamp detection
- Gap analysis (trading day continuity)
- NaN percentage calculation
- Zero-volume day counting

### 3. Statistical Testing Module

#### Hurst Exponent (`calculate_hurst_exponent`)
```python
Method: Rescaled Range (R/S)
Formula: H from log(R/S) ~ H * log(lag)
Quality: R² ≥ 0.80 required
Output: (hurst, r_squared) or (None, None)
```

**Interpretation**:
- H < 0.5: Anti-persistent (mean reverting)
- H ≈ 0.5: Random walk (efficient market)
- H > 0.5: Persistent (trending)

**Edge Cases**:
- Constant series → Return (0.5, 1.0) as baseline
- Out-of-range H → Clamp to [0, 1]
- Noisy short windows → Return None

#### ADF Test (`adf_test`)
```python
Method: Augmented Dickey-Fuller
Null Hypothesis: Unit root exists (non-stationary)
Threshold: p < 0.05 → Stationary
Returns: {adf_stat, p_value, is_stationary, note}
```

**Fallback Logic**:
- statsmodels unavailable → Return dict with note="statsmodels_not_installed"
- Insufficient data → Return dict with note="insufficient_data"
- Constant series → Return dict with note="constant_series"

#### Half-Life (`calculate_half_life`)
```python
Model: Ornstein-Uhlenbeck process
Equation: Δy[t] = λ(μ - y[t-1]) + ε
Half-Life: ln(2) / λ trading days
```

**Validity Range**: 5-60 days considered tractable for trading.

#### Variance Ratio (`calculate_variance_ratio`)
```python
Formula: VR(q) = Var[q-day returns] / (q × Var[1-day returns])
Hypothesis: VR = 1 under random walk
Interpretation: VR < 1 → Mean reversion, VR > 1 → Momentum
```

### 4. Regime Classification (`classify_regime`)

**Architecture**: Multi-factor scoring system with adaptive thresholds.

**Scoring Logic**:
```python
score = 0
if hurst ∈ [0, 0.45]: score += weight_hurst
if adf_p < 0.05: score += weight_adf
if half_life ∈ [5, 60]: score += weight_halflife
if variance_ratio < 0.80: score += weight_vr
```

**Adaptive Thresholds**:
```python
# Compute 252-bar rolling percentiles
p10 = score.rolling(252).quantile(0.10)
p90 = score.rolling(252).quantile(0.90)

# Label assignment
if score > p90: regime = MEAN_REVERTING
elif score > p10: regime = SIDEWAYS
else: regime = TRENDING
```

**Performance Optimization**:
- Regime tests run every N bars (default: 5)
- Cached scores interpolated for skipped bars
- Vectorized percentile computation

**Verdict Derivation** (P1.2):
```python
# Dominant regime = mode of per-bar labels
verdict = max(regime_counts.items(), key=lambda x: x[1])[0]
```

### 5. Signal Generation

#### Ratio Series (`compute_ratio_series`)
```python
if ratio_mode == "price_to_sma":
    ratio = close / SMA(close, anchor_window)
elif ratio_mode == "price_to_ema":
    ratio = close / EMA(close, anchor_window)
```

#### Z-Score Normalization (`compute_ratio_z`)
```python
z[i] = (ratio[i] - mean(ratio[i-lookback:i])) / std(ratio[i-lookback:i])
```

#### Action Determination (`determine_action`)
```python
if no_position:
    if z < entry_z and regime_ok and reversal_ok and quality_ok:
        return BUY
elif position_pct < max_position_pct:
    if z < add_z:
        return ADD
elif position_pct > 0:
    for (z_thresh, trim_frac) in sorted(trim_levels):
        if z >= z_thresh:
            return REDUCE  # trim by trim_frac
return HOLD
```

### 6. Position Sizing

#### Base Sizing
```python
notional = capital * max_position_pct
shares = notional / price
```

#### Volatility Adjustment (P2.1)
```python
sigma_annual = std(log_returns) * sqrt(252)
vol_multiplier = clip(1.0 / sigma_annual, vol_floor, vol_cap)
adjusted_shares = shares * vol_multiplier
```

#### Confidence Weighting (Feature D)
```python
confidence = share_of_bars_in(MEAN_REVERTING, SIDEWAYS) over last 60 bars
m_conf = clip((confidence - c0) / (1.0 - c0), 0, 1) ** gamma
adjusted_shares = shares * m_conf
```

#### Realized Vol Targeting (Feature A)
```python
realized_vol = std(returns[-20:]) * sqrt(252)
m_vol = clip(target_vol / max(realized_vol, floor), 0, cap)
adjusted_shares = shares * m_vol
```

### 7. Risk Management

#### ATR Stop Loss
```python
stop_price = entry_price - (ATR * stop_atr_multiple)

# Intraday execution simulation
if open < stop_price:
    fill_price = open  # Gap down
else:
    fill_price = max(low, min(stop_price, high))
```

#### Thesis Break Detection (P0.2)
```python
sma_slope = (SMA[-1] - SMA[-thesis_break_sma_bars]) / thesis_break_sma_bars
consecutive_neg = all(SMA[i] < SMA[i-1] for i in range(-thesis_break_sma_bars, 0))
below_sma = close < SMA[-1]

if consecutive_neg and abs(sma_slope) > min_slope and below_sma:
    trigger_exit()
```

#### Time Decay (Feature C)
```python
if hold_days >= tactical_min_hold_days:
    if z < tactical_exit_z or hold_days >= tactical_max_hold_days:
        trigger_exit(reason="TIME_DECAY")
```

### 8. Two-Layer Portfolio Engine (`TwoLayerPortfolioEngine`)

**Architecture**: Isolated sleeves with independent accounting.

```
Total Capital
├── Core Sleeve (70-80%)
│   ├── Entry Mode: instant | dca | adaptive
│   ├── Cash Component (uninvested + yield)
│   └── Equity Component (shares × price)
└── Tactical Sleeve (20-30%)
    ├── BacktestEngine (existing MR strategy)
    ├── Cash Component (uninvested + yield)
    └── Equity Component (position)
```

#### Core Entry Modes

**Instant** (Legacy):
```python
shares = core_capital / close[0]
equity[i] = shares * close[i]
```

**DCA** (Dollar-Cost Averaging):
```python
invest_per_day = core_capital / dca_days
for i in range(dca_start, dca_start + dca_days):
    # Schedule buy at decision time
    pending_spend = invest_per_day
    # Execute at next bar's open
    shares += pending_spend / open[i+1]
```

**Adaptive** (State-Based Deployment):
```python
for i in range(adaptive_start, n):
    deploy_amount, state = _adaptive_core_deploy_amount(
        close_i, sma200_i, dd_i, sigma_i, remaining_cash
    )
    shares += deploy_amount / close[i]  # Buy at same close
```

#### Adaptive State Machine

```
States → [CALM_UPTREND, PULLBACK, HIGH_VOL_DRAWDOWN, NEUTRAL, WAITING, DEPLETED]

CALM_UPTREND:
  Condition: trend=1, dd<10%, σ<vol_target
  Action: deploy = C / slow_days (patient accumulation)

PULLBACK:
  Condition: trend=0, 5%≤dd≤20%
  Action: deploy = C / fast_days (buy the dip)

HIGH_VOL_DRAWDOWN:
  Condition: dd≥15% or σ≥vol_target
  Action: deploy = (C / fast_days) * vol_scale  (risk-adjusted acceleration)
  vol_scale = clip(vol_target / max(sigma, floor), 0, 1)

NEUTRAL:
  Condition: everything else
  Action: deploy = C / base_days (standard pace)

WAITING:
  Condition: i < adaptive_start
  Action: no deployment

DEPLETED:
  Condition: remaining_cash < $0.01
  Action: no deployment
```

**Design Properties**:
1. **No Lookahead**: Uses only data through bar i
2. **Monotonic Risk**: Higher volatility → less deployed
3. **Drawdown Acceleration**: Deeper pullbacks → faster deployment (bounded by vol adjustment)
4. **Capital Conservation**: shares[i] × close[i] + cash[i] = starting_capital (accounting identity)

#### Rebalancing Logic

```python
if calendar_event(rebalance_freq) and drift > threshold:
    # Compute target allocations
    target_core = total_equity × core_pct
    target_tactical = total_equity × tactical_pct
    
    # Calculate turnover and costs
    turnover = abs(current_core - target_core)
    transaction_cost = turnover × slippage + 2 × commission
    
    # Rebalance
    total_after_cost = total_equity - transaction_cost
    core = total_after_cost × core_pct
    tactical = total_after_cost × tactical_pct
    
    rebalancing_events += 1
```

### 9. Execution Simulation (`BacktestEngine.run`)

**Event Loop**:
```python
for i, (timestamp, bar) in enumerate(df.iterrows()):
    # 1. Update state
    current_price = get_execution_price(bar, entry_at)
    
    # 2. Generate signal
    action = strategy.on_bar(
        history=df.iloc[:i+1],
        current_bar=bar,
        has_position=(state.shares > 0),
        position_pct=(state.shares * current_price / capital),
        reversal_ok=check_reversal_confirmation(...),
        quality_ok=passes_quality_filter(...),
    )
    
    # 3. Apply regime filter
    if regime_filter_enabled and not regime_allows_action(action, regime, policy):
        action = BLOCKED
        blocked_count += 1
        blocked_reason = "REGIME" if regime == TRENDING else "CONFIDENCE"
    
    # 4. Apply cost-aware gate (Feature B)
    if cost_aware_entry_enabled and action in (BUY, ADD):
        e_ret = _expected_return_spec_v1(ratio_now, ratio_mu, half_life)
        notional = compute_notional(...)
        if not _is_cost_effective_spec_v1(e_ret, cost_k, slippage, commission, notional):
            action = BLOCKED
            blocked_by_cost += 1
    
    # 5. Execute trade
    if action == BUY:
        shares_to_buy = compute_shares(capital, max_pct, price, vol_adj, conf_adj)
        if shares_to_buy >= min_shares and value >= min_notional:
            execute_buy(shares_to_buy, price, slippage, commission)
            state.shares += shares_to_buy
            state.entry_price = compute_avg_entry()
    
    elif action == REDUCE:
        trim_action = _get_trim_action(z, trim_levels)
        shares_to_sell = state.shares * trim_action.trim_fraction
        execute_sell(shares_to_sell, price, slippage, commission)
        state.shares -= shares_to_sell
    
    # 6. Check stops
    if has_position and current_price <= state.stop_price:
        fill_price = _stop_execution_base(open, high, low, stop_price)
        execute_sell(state.shares, fill_price, slippage, commission)
        state.shares = 0
        exit_reason = "STOP_LOSS"
    
    # 7. Time decay (Feature C)
    if better_exits_enabled and hold_days >= min_hold and z < exit_z:
        execute_sell(state.shares, price, slippage, commission)
        state.shares = 0
        exit_reason = "TIME_DECAY"
    
    # 8. Record state
    equity_curve.append(state.shares * current_price + state.cash)
    exposure_pct_curve.append((state.shares * current_price / capital) * 100)
    actions_log.append({
        "date": timestamp,
        "action": action,
        "regime": regime,
        "z_score": z,
        "confidence": confidence,
    })
```

### 10. Performance Metrics

#### Sharpe Ratio
```python
returns = pct_change(equity_curve)
sharpe = mean(returns) / std(returns) * sqrt(252)
```

#### Sortino Ratio
```python
downside_returns = returns[returns < 0]
sortino = mean(returns) / std(downside_returns) * sqrt(252)
```

#### Calmar Ratio
```python
ann_return = (1 + total_return) ** (252 / n_bars) - 1
max_dd = abs(min((equity - running_max) / running_max))
calmar = ann_return / max_dd
```

#### Capture Ratios
```python
up_days = benchmark_returns > 0
dn_days = benchmark_returns < 0

upside_capture = mean(portfolio_returns[up_days]) / mean(benchmark_returns[up_days]) * 100
downside_capture = mean(portfolio_returns[dn_days]) / mean(benchmark_returns[dn_days]) * 100
capture_ratio = upside_capture / downside_capture
```

#### Return Attribution
```python
core_contribution = (core_equity[-1] - core_equity[0]) / total_return * 100
tactical_contribution = (tactical_equity[-1] - tactical_equity[0]) / total_return * 100
```

## Design Patterns

1. **Strategy Pattern**: `MeanReversionStrategy` encapsulates signal logic
2. **Builder Pattern**: `MeanReversionConfig.copy_with()` for safe overrides
3. **Template Method**: `BacktestEngine.run()` defines execution skeleton
4. **Observer Pattern**: `actions_log` records all state transitions
5. **Factory Pattern**: Synthetic data generators in test suite

## Testing Strategy

### Unit Tests (50+ tests)
- **Pure Functions**: Test statistical calculations with known inputs/outputs
- **Edge Cases**: NaN handling, zero-length series, constant prices
- **Regression Tests**: Lock in v3.0 → v3.3 behavior preservation

### Integration Tests
- **End-to-End**: Full backtest on synthetic mean-reverting series
- **A/B Comparison**: Regime-filtered vs. baseline (no filter)
- **Accounting**: Capital conservation, no lookahead

### Property-Based Tests
- **Monotonicity**: Higher vol → less deployed (adaptive core)
- **Bounded Deployment**: min_pct ≤ deploy_pct ≤ max_pct
- **Stability**: Small price perturbations → stable schedule (correlation > 0.95)

## Performance Considerations

1. **Vectorization**: Regime scores computed in bulk every N bars
2. **Lazy Evaluation**: Expensive tests (ADF, Hurst) skipped when regime update not due
3. **Memory**: Single-pass equity curve construction (no intermediate copies)
4. **Caching**: Regime labels cached for interpolation between update points

## Extension Points

1. **Custom Strategies**: Subclass `MeanReversionStrategy`, override `on_bar()`
2. **Custom Regime Signals**: Implement `RegimeSignal` interface
3. **Custom Filters**: Add to `passes_quality_filter()` logic
4. **Custom Metrics**: Extend `calculate_performance_metrics()`

## Future Architecture

**Phase 2 (Planned)**: Probabilistic regime beliefs

```python
class RegimeBeliefs:
    """HMM-lite: maintain probability distribution over regimes."""
    def __init__(self):
        self.beliefs = {
            Regime.MEAN_REVERTING: 0.25,
            Regime.SIDEWAYS: 0.25,
            Regime.TRENDING: 0.25,
            Regime.AMBIGUOUS: 0.25,
        }
    
    def update(self, observation: Dict):
        """Bayesian update: P(R|obs) ∝ P(obs|R) × P(R)"""
        # Emission probabilities from statistical tests
        # Transition model from historical regime matrix
        pass
    
    def entropy(self) -> float:
        """Shannon entropy: -Σ p log(p)"""
        return -sum(p * np.log(p) for p in self.beliefs.values() if p > 0)
```

This would enable:
- **Soft Filtering**: Weight trades by regime probability instead of hard block
- **Uncertainty Tracking**: High entropy → reduce position size
- **Adaptive Thresholds**: Transition probabilities inform entry/exit timing
