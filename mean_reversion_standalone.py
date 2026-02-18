#!/usr/bin/env python3
"""
================================================================================
MEAN REVERSION BACKTESTER v3.3 — LONG-ONLY ACTION ENGINE
================================================================================
v3.3 additions vs v3.2:
  PH1.1  Cash yield on idle tactical capital (--cash-yield, daily compounding)
  PH1.2  Exposure vs Price scatter, Return Contribution, Enhanced Underwater plots
  PH1.3  Diagnostics: capture ratios, Calmar, attribution, tactical time-in-market
  PH1.4  Hurst exponent clamped to [0, 1] when out-of-range
  PH2    RegimeSignal abstraction (no probabilistic math yet)

v3.2 additions vs v3.1:
  TL.A  Two-Layer Core/Tactical portfolio mode (--two-layer)
  TL.B  CorePosition (buy-and-hold sleeve) + TwoLayerPortfolioEngine
  TL.C  Static 80/20 baseline comparison (constant-weight, no rebalancing)
  TL.D  Visualization: benchmark PNG, drawdown PNG, curves CSV (--plot)
  TL.E  config.copy_with() helper — eliminates manual field-by-field copies
  TL.F  Phase 2 design notes (probabilistic HMM-lite regime beliefs)

v3.1 patch notes vs v3.0:
  P0.1  ADF test always returns {adf_stat, p_value, is_stationary, note}
  P0.2  Thesis-break requires consecutive neg slope + magnitude + guard
  P0.3  A/B comparison enforced: baseline truly unfiltered
  P0.4  Dust-fill filter: min_trade_notional / min_shares
  P1.1  Adaptive regime thresholds -> SIDEWAYS actually appears
  P1.2  Verdict derived from per-bar regime label distribution
  P2.1  Volatility-adjusted sizing (risk-parity lite)
  P2.2  Staged exits: 3-level trim at configurable ratio_z thresholds
  Diagnostics: metrics-by-regime, exit-reason breakdown, regime transitions

Install:
    pip install yfinance pandas numpy statsmodels matplotlib

Usage (backward-compatible):
    python mean_reversion_standalone.py AAPL
    python mean_reversion_standalone.py NVDA --window 30 --debug
    python mean_reversion_standalone.py GOOG --no-regime --entry-at same_close

Two-Layer mode:
    python mean_reversion_standalone.py AAPL --two-layer --plot
    python mean_reversion_standalone.py AAPL --two-layer --core-pct 0.7 --tactical-pct 0.3
================================================================================
"""

import logging
import sys
import argparse
import csv
import json
import hashlib
import re
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

logger = logging.getLogger("mr_backtest")

_PERIOD_PATTERN = re.compile(r"^\d+(d|wk|mo|y)$")
_VALID_INTERVALS = {
    "1m", "2m", "5m", "15m", "30m", "60m", "90m",
    "1h", "1d", "5d", "1wk", "1mo", "3mo",
}
_VALID_PRICE_FIELDS = {"adjclose", "close"}

# ---------------------------------------------------------------------------
# Optional imports
# ---------------------------------------------------------------------------
try:
    import yfinance as yf
    YF_OK = True
except ImportError:
    YF_OK = False
    print("Warning: yfinance not installed - run: pip install yfinance")

try:
    from statsmodels.tsa.stattools import adfuller  # type: ignore[reportMissingImports]
    SM_OK = True
except ImportError:
    SM_OK = False

try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend; safe for CLI / CI
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.ticker import FuncFormatter
    MPL_OK = True
except ImportError:
    MPL_OK = False


# =============================================================================
# ENUMS
# =============================================================================

class Regime(str, Enum):
    MEAN_REVERTING = "MEAN_REVERTING"
    SIDEWAYS = "SIDEWAYS"
    TRENDING = "TRENDING"
    AMBIGUOUS = "AMBIGUOUS"


class Action(str, Enum):
    BUY = "BUY"
    ADD = "ADD"
    HOLD = "HOLD"
    REDUCE = "REDUCE"
    SELL = "SELL"
    BLOCKED = "BLOCKED"


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class MeanReversionConfig:
    """Consolidated configuration for the long-only MR backtester."""

    # --- Ratio anchor ---
    ratio_mode: str = "price_to_sma"      # "price_to_sma" | "price_to_ema"
    ratio_anchor_window: int = 200
    ratio_lookback: int = 60

    # --- Signal thresholds (ratio z-score) ---
    entry_z: float = -1.5
    add_z: float = -2.0
    # Staged trim thresholds (P2.2): list of (z_threshold, trim_fraction)
    trim_levels: List[Tuple[float, float]] = field(
        default_factory=lambda: [(0.5, 0.25), (1.0, 0.50), (2.0, 1.0)]
    )
    # Legacy single thresholds (used if trim_levels empty)
    trim_z: float = 1.0
    sell_z: float = 2.0
    exit_threshold: float = 0.0
    lookback_window: int = 20

    # --- Reversal confirmation ---
    require_reversal_confirmation: bool = True
    confirmation_methods: List[str] = field(
        default_factory=lambda: ["rsi_turning", "close_above_prior"]
    )

    # --- Regime ---
    regime_filter_enabled: bool = True
    allowed_regimes: List[str] = field(
        default_factory=lambda: ["MEAN_REVERTING", "SIDEWAYS"]
    )
    ambiguous_policy: str = "tighten"
    regime_lookback: int = 252
    regime_adaptive_thresholds: bool = True  # P1.1
    regime_update_freq: int = 5              # run expensive regime tests every N bars

    # --- Position sizing & scaling ---
    starting_capital: float = 100_000.0
    max_position_pct: float = 0.15
    add_step_pct: float = 0.03
    trim_step_pct: float = 0.03
    min_cash_pct: float = 0.05
    # P0.4: dust fill filters
    min_trade_notional: float = 1000.0
    min_shares: int = 5
    # P2.1: vol-adjusted sizing
    vol_adjust_sizing: bool = True
    vol_sizing_floor: float = 0.5    # min multiplier
    vol_sizing_cap: float = 1.5      # max multiplier
    # Feature A (Part 1): tactical realized-vol targeting multiplier
    tactical_vol_targeting_enabled: bool = False
    tactical_vol_target: float = 0.15          # Spec v1 default: 0.15
    tactical_vol_window: int = 20
    tactical_vol_floor: float = 0.05
    tactical_vol_cap: float = 1.50
    # Feature B (Part 2): cost-aware entry gate
    cost_aware_entry_enabled: bool = False
    cost_bps_est: float = 15.0
    cost_k: float = 1.0                        # Spec v1 default: 1.0
    # Feature C (Part 2): better exits (optional; additive)
    better_exits_enabled: bool = False
    tactical_exit_z: float = -0.20
    tactical_min_hold_days: int = 3
    tactical_max_hold_days: int = 30
    # Feature D (Part 3): confidence-weighted sizing (optional; additive)
    confidence_sizing_enabled: bool = False
    confidence_c0: float = 0.60
    confidence_gamma: float = 1.0
    # Feature E (Part 3): tactical sleeve mode
    tactical_mode: str = "single"            # "single" | "portfolio"
    tactical_max_positions: int = 5
    tactical_entry_z: float = -1.25
    tactical_weighting: str = "equal"        # "equal" | "inv_vol"

    # --- Risk management ---
    stop_atr_multiple: float = 3.0
    target_atr_multiple: float = 3.0
    max_holding_days: int = 30
    # P0.2: thesis break fix
    thesis_break_sma_bars: int = 10
    thesis_break_min_slope: float = 0.0   # auto-calibrated if 0
    thesis_break_require_below_sma: bool = True

    # --- Execution ---
    entry_at: str = "next_open"
    lag_signals_for_same_close: bool = True  # avoid same-close look-ahead by using lagged signals
    slippage_pct: float = 0.0005
    commission_per_trade: float = 1.0

    # --- Quality filter (proxy) ---
    require_above_sma200: bool = False
    require_positive_12m_momentum: bool = False

    # --- Legacy compat ---
    require_volume_confirmation: bool = True
    volume_confirmation_ratio: float = 1.5
    require_rsi_confirmation: bool = True
    rsi_oversold: float = 35.0
    min_confidence: float = 0.0
    recent_signal_days: int = 120
    min_days_between_signals: int = 2

    # --- Debug ---
    debug: bool = False

    # --- Two-Layer Core/Tactical (v3.2) ---
    two_layer_mode: bool = False
    core_allocation_pct: float = 0.80
    tactical_allocation_pct: float = 0.20
    cash_yield_annual_pct: float = 0.0  # T-bill proxy for tactical cash benchmark
    plot: bool = False  # save benchmark PNG / drawdown PNG / curves CSV
    rebalance_freq: Optional[str] = "Q"  # None, "M", "Q", "A"
    rebalance_drift_threshold: float = 0.05

    # --- Core DCA Entry (v3.4) ---
    core_entry_mode: str = "instant"      # "instant" | "dca" | "adaptive"
    core_dca_days: int = 40               # number of bars to spread core entry over
    core_dca_start: int = 0               # bar offset to begin DCA
    core_dca_commission: float = 0.0      # per-fill commission for DCA buys
    core_dca_slippage_pct: float = 0.0    # slippage for DCA buys

    # --- Core Adaptive Entry ---
    core_adaptive_base_days: int = 60     # NEUTRAL state deploy pace
    core_adaptive_slow_days: int = 120    # CALM_UPTREND deploy pace
    core_adaptive_fast_days: int = 40     # PULLBACK / HIGH_VOL_DRAWDOWN base pace
    core_adaptive_vol_window: int = 20    # rolling window for realized vol
    core_adaptive_dd_window: int = 252    # rolling window for peak drawdown
    core_adaptive_vol_target: float = 0.12  # 12% annualized vol threshold
    core_adaptive_vol_floor: float = 0.05   # floor for vol in HIGH_VOL_DRAWDOWN scaling
    core_adaptive_max_deploy_pct: float = 0.10  # max fraction of remaining cash per bar
    core_adaptive_min_deploy_pct: float = 0.002  # min fraction of remaining cash per bar
    core_adaptive_start: int = 0          # bar offset to begin adaptive deployment

    # --- Spec v1: Tactical Overlay Upgrade (gating flag) ---
    enable_spec_v1_upgrades: bool = False

    # --- Universe Filter (v3.4) ---
    universe_filter_enabled: bool = False
    min_dollar_vol: float = 20_000_000.0  # median daily dollar volume threshold
    min_price: float = 5.0                # median close price threshold
    max_gaps: int = 10                     # max detected large date gaps
    gap_days_threshold: int = 7            # for interval=1d, count gaps >= this many calendar days
    max_nan_frac: float = 0.01             # max allowed NaN fraction in close/volume rows
    max_zero_vol_days: int = 5             # hard fail when too many zero-volume bars
    min_price_tail: float = 1.0            # tail median close threshold on last 30 bars

    def __post_init__(self):
        assert self.entry_at in ("next_open", "same_close")
        assert 0 < self.max_position_pct <= 1.0
        assert self.starting_capital > 0
        assert self.core_entry_mode in ("instant", "dca", "adaptive")
        if self.core_entry_mode == "dca":
            assert self.core_dca_days >= 1, "core_dca_days must be >= 1"
            assert self.core_dca_start >= 0, "core_dca_start must be >= 0"
        if self.core_entry_mode == "adaptive":
            assert self.core_adaptive_base_days >= 1, "core_adaptive_base_days must be >= 1"
            assert self.core_adaptive_slow_days >= 1, "core_adaptive_slow_days must be >= 1"
            assert self.core_adaptive_fast_days >= 1, "core_adaptive_fast_days must be >= 1"
            assert self.core_adaptive_vol_window >= 2, "core_adaptive_vol_window must be >= 2"
            assert self.core_adaptive_dd_window >= 1, "core_adaptive_dd_window must be >= 1"
            assert self.core_adaptive_vol_target > 0, "core_adaptive_vol_target must be > 0"
            assert self.core_adaptive_vol_floor > 0, "core_adaptive_vol_floor must be > 0"
            assert 0 < self.core_adaptive_max_deploy_pct <= 1.0
            assert 0 < self.core_adaptive_min_deploy_pct <= self.core_adaptive_max_deploy_pct
            assert self.core_adaptive_start >= 0
        if self.two_layer_mode:
            total = self.core_allocation_pct + self.tactical_allocation_pct
            assert abs(total - 1.0) < 1e-9, (
                f"core + tactical must sum to 1.0, got {total:.4f}"
            )
        assert int(self.regime_update_freq) >= 1, "regime_update_freq must be >= 1"
        assert self.rebalance_freq in (None, "M", "Q", "A"), (
            "rebalance_freq must be one of: None, 'M', 'Q', 'A'"
        )
        assert float(self.rebalance_drift_threshold) >= 0.0, (
            "rebalance_drift_threshold must be >= 0"
        )
        assert 0.0 <= float(self.max_nan_frac) <= 1.0, "max_nan_frac must be in [0, 1]"
        assert int(self.gap_days_threshold) >= 1, "gap_days_threshold must be >= 1"
        assert int(self.max_zero_vol_days) >= 0, "max_zero_vol_days must be >= 0"
        assert float(self.min_price_tail) >= 0.0, "min_price_tail must be >= 0"
        assert float(self.tactical_vol_target) >= 0.0, "tactical_vol_target must be >= 0"
        assert int(self.tactical_vol_window) >= 2, "tactical_vol_window must be >= 2"
        assert float(self.tactical_vol_floor) > 0.0, "tactical_vol_floor must be > 0"
        assert float(self.tactical_vol_cap) > 0.0, "tactical_vol_cap must be > 0"
        assert float(self.cost_bps_est) >= 0.0, "cost_bps_est must be >= 0"
        assert float(self.cost_k) >= 1.0, "cost_k must be >= 1.0"
        assert int(self.tactical_min_hold_days) >= 0, "tactical_min_hold_days must be >= 0"
        assert int(self.tactical_max_hold_days) >= 1, "tactical_max_hold_days must be >= 1"
        assert 0.0 <= float(self.confidence_c0) < 1.0, "confidence_c0 must be in [0, 1)"
        assert float(self.confidence_gamma) >= 1.0, "confidence_gamma must be >= 1"
        assert self.tactical_mode in ("single", "portfolio"), "tactical_mode must be 'single' or 'portfolio'"
        assert int(self.tactical_max_positions) >= 1, "tactical_max_positions must be >= 1"
        assert self.tactical_weighting in ("equal", "inv_vol"), "tactical_weighting must be 'equal' or 'inv_vol'"

    def copy_with(self, **overrides) -> "MeanReversionConfig":
        """Return a shallow copy with selected fields overridden.

        This avoids the error-prone pattern of manually copying every field
        and prevents accidental mutation of the shared original.
        """
        import dataclasses as _dc
        vals = {f.name: getattr(self, f.name) for f in _dc.fields(self)}
        vals.update(overrides)
        return MeanReversionConfig(**vals)


# =============================================================================
# UTILITIES
# =============================================================================

def safe_number(value: Any, default: Optional[float] = None) -> Optional[float]:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return default if (pd.isna(value) or np.isinf(value)) else float(value)
    if isinstance(value, str):
        try:
            return float(value.replace(",", "").replace("$", ""))
        except ValueError:
            return default
    return default


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure tz-aware sorted index and required OHLCV columns."""
    df = df.sort_index().copy()
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    df.index = idx
    for col in ("open", "high", "low", "close", "volume"):
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    return df


def is_valid_period(period: str) -> bool:
    """Accept yfinance period values like 500d/4y/10y/max."""
    if not isinstance(period, str):
        return False
    value = period.strip().lower()
    return value == "max" or bool(_PERIOD_PATTERN.match(value))


def is_valid_interval(interval: str) -> bool:
    """Allow common yfinance intervals."""
    if not isinstance(interval, str):
        return False
    return interval.strip().lower() in _VALID_INTERVALS


def is_valid_price_field(price_field: str) -> bool:
    if not isinstance(price_field, str):
        return False
    return price_field.strip().lower() in _VALID_PRICE_FIELDS


def is_valid_iso_date_or_empty(value: str) -> bool:
    if value is None:
        return True
    text = str(value).strip()
    if text == "":
        return True
    try:
        datetime.strptime(text, "%Y-%m-%d")
        return True
    except ValueError:
        return False


def _safe_ticker_for_filename(ticker: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(ticker))
    return safe or "TICKER"


def _stop_execution_base(open_px: float, high_px: float, low_px: float, stop_px: float) -> float:
    """Base stop execution on daily bars: gap below stop fills at open, otherwise at stop."""
    base = float(open_px) if float(open_px) < float(stop_px) else float(stop_px)
    lo = float(low_px)
    hi = float(high_px)
    return max(lo, min(base, hi))


def _should_block_next_open_entry(i: int, n: int, entry_at: str) -> bool:
    return str(entry_at) == "next_open" and int(i) >= int(n) - 1


def _daily_cash_yield_rate(cash_yield_annual_pct: float) -> float:
    """Convert annualized cash yield (%) to daily compounding over calendar days."""
    annual = float(cash_yield_annual_pct)
    if annual == 0:
        return 0.0
    return (1 + annual / 100) ** (1 / 365) - 1


def _vol_target_multiplier(
    sigma_annual: Optional[float],
    sigma_target: float,
    sigma_floor: float,
    cap: float,
) -> float:
    """Spec v1: m_vol = np.clip(target_vol / np.maximum(sigma_annual, vol_floor), 0, vol_cap)."""
    sig_a = safe_number(sigma_annual, None)
    if sig_a is None or not np.isfinite(sig_a):
        return 1.0
    sig_eff = max(float(sig_a), float(sigma_floor))
    raw = float(sigma_target) / sig_eff if sig_eff > 0 else float(cap)
    return float(np.clip(raw, 0.0, float(cap)))


def _expected_return_proxy_from_ratio(
    ratio_now: Optional[float],
    ratio_mu: Optional[float],
    eps: float = 1e-12,
) -> float:
    """Feature B: ERet ~= max(mu - R, 0) / max(R, eps)."""
    r = safe_number(ratio_now, None)
    mu = safe_number(ratio_mu, None)
    if r is None or mu is None:
        return 0.0
    dr = max(float(mu) - float(r), 0.0)
    return float(dr / max(float(r), float(eps)))


def _expected_return_spec_v1(
    ratio_now: Optional[float],
    ratio_mu: Optional[float],
    half_life: Optional[float],
    eps: float = 1e-12,
) -> float:
    """Spec v1: e_ret = dR * (ratio_mu - ratio_now) / ratio_now.

    dR = np.log(2) / half_life if half_life > 0 else 0.0
    """
    r = safe_number(ratio_now, None)
    mu = safe_number(ratio_mu, None)
    hl = safe_number(half_life, None)
    if r is None or mu is None or r <= eps:
        return 0.0
    dR = float(np.log(2) / float(hl)) if (hl is not None and float(hl) > 0) else 0.0
    return float(dR * (float(mu) - float(r)) / float(r))


def _is_cost_effective_spec_v1(
    e_ret: float,
    cost_k: float,
    slippage_bps: float,
    commission: float,
    notional: float,
) -> bool:
    """Spec v1: is_cost_effective = e_ret >= (cost_k * (slippage_bps/10000 + commission/notional))."""
    if notional <= 0:
        return False
    min_req = float(cost_k) * (float(slippage_bps) / 10000.0 + float(commission) / float(notional))
    return float(e_ret) >= min_req


def _entry_confidence_from_labels(
    i: int,
    labels: pd.Series,
    N: int = 60,
) -> Optional[float]:
    """Entry confidence proxy: share of last N bars in {MR, SIDEWAYS}."""
    if labels is None or len(labels) == 0 or int(i) < 0:
        return None
    end = min(int(i), len(labels) - 1)
    start = max(0, end - int(N) + 1)
    window = labels.iloc[start:end + 1]
    if len(window) == 0:
        return None
    good = window.isin([Regime.MEAN_REVERTING.value, Regime.SIDEWAYS.value])
    return float(good.mean())


def _confidence_sizing_multiplier(
    conf: Optional[float],
    c0: float = 0.60,
    gamma: float = 1.0,
) -> float:
    """Feature D / Spec v1: m_conf = np.clip((confidence - c0) / (1.0 - c0), 0, 1.0).

    When gamma=1.0 (Spec v1 default), this is a pure linear ramp with no power/gamma.
    """
    c = safe_number(conf, None)
    if c is None:
        return 0.0
    c_f = float(np.clip(float(c), 0.0, 1.0))
    c0_f = float(np.clip(float(c0), 0.0, 0.999999))
    if c_f <= c0_f:
        m_conf = 0.0
    elif c_f >= 1.0:
        m_conf = 1.0
    else:
        denom = max(1.0 - c0_f, 1e-12)
        m_conf = (c_f - c0_f) / denom
    g = max(float(gamma), 1.0)
    return float(np.clip(m_conf, 0.0, 1.0) ** g)


def _apply_adjclose_price_volume_scaling(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Scale OHLC to adjclose basis and inverse-scale volume for split-consistent volume ratios."""
    out = df.copy()
    close_raw = out["close"].replace(0, np.nan)
    scale = out["adjclose"] / close_raw
    scale_safe = scale.replace([np.inf, -np.inf, 0, -0.0], np.nan).fillna(1.0)
    for col in ("open", "high", "low", "close"):
        if col in out.columns:
            out[col] = out[col] * scale_safe
    if "volume" in out.columns:
        out["volume"] = out["volume"] / scale_safe
    return out, scale_safe


def _adaptive_core_deploy_amount(
    close_i: float,
    sma200_i: float,
    dd_i: float,
    sigma_i: float,
    remaining_cash: float,
    *,
    base_days: int = 60,
    slow_days: int = 120,
    fast_days: int = 40,
    vol_target: float = 0.12,
    vol_floor: float = 0.05,
    max_deploy_pct: float = 0.10,
    min_deploy_pct: float = 0.002,
) -> Tuple[float, str]:
    """Compute adaptive core deployment amount for one bar.

    Pure function — uses only current-bar data (no lookahead).

    State classification (simple, deterministic):
        CALM_UPTREND      : trend=1 and dd < 0.10 and sigma < vol_target
        PULLBACK          : trend=0 and dd in [0.05, 0.20]
        HIGH_VOL_DRAWDOWN : dd >= 0.15 or sigma >= vol_target
        NEUTRAL           : everything else

    Returns (deploy_amount, state_label).
    """
    if remaining_cash <= 0.01:
        return 0.0, "DEPLETED"

    # --- indicators (NaN-safe) ---
    sma_ok = not (sma200_i is None or np.isnan(sma200_i))
    sigma_ok = not (sigma_i is None or np.isnan(sigma_i))
    trend = 1 if (sma_ok and close_i >= sma200_i) else 0
    dd = float(dd_i) if not (dd_i is None or np.isnan(dd_i)) else 0.0
    sigma = float(sigma_i) if sigma_ok else 0.0

    # --- state classification ---
    is_high_vol_dd = (dd >= 0.15) or (sigma_ok and sigma >= vol_target)
    is_calm_uptrend = (trend == 1 and dd < 0.10
                       and (not sigma_ok or sigma < vol_target))
    is_pullback = (trend == 0 and 0.05 <= dd <= 0.20)

    if is_high_vol_dd:
        state = "HIGH_VOL_DRAWDOWN"
        vol_scale = float(np.clip(
            vol_target / max(sigma, vol_floor), 0.0, 1.0,
        ))
        raw = (remaining_cash / float(fast_days)) * vol_scale
    elif is_pullback:
        state = "PULLBACK"
        raw = remaining_cash / float(fast_days)
    elif is_calm_uptrend:
        state = "CALM_UPTREND"
        raw = remaining_cash / float(slow_days)
    else:
        state = "NEUTRAL"
        raw = remaining_cash / float(base_days)

    # --- clamp ---
    lo = min_deploy_pct * remaining_cash
    hi = max_deploy_pct * remaining_cash
    deploy = float(np.clip(raw, lo, hi))
    return deploy, state


def _safe_iso_date(ts: Any) -> str:
    try:
        if ts is None or pd.isna(ts):
            return ""
        return pd.Timestamp(ts).strftime("%Y-%m-%d")
    except Exception:
        return ""


def compute_data_quality_diagnostics(
    df: Optional[pd.DataFrame],
    *,
    requested_price_field: str = "adjclose",
    price_field_used: str = "close",
    price_field_warning: str = "",
    raw_duplicate_timestamps_count: Optional[int] = None,
    raw_is_monotonic_index: Optional[bool] = None,
    has_splits: bool = False,
    has_dividends: bool = False,
    has_split_events: bool = False,
    has_dividend_events: bool = False,
    ohlc_scaled_by_adjclose: bool = False,
    volume_inverse_scaled: bool = False,
    interval: str = "1d",
    gap_days_threshold_1d: int = 7,
) -> Dict[str, Any]:
    """Compute lightweight data-quality diagnostics from downloaded bars."""
    gap_days_threshold_used = (
        int(gap_days_threshold_1d) if str(interval).strip().lower() == "1d" else 3
    )
    gap_days_threshold_used = max(1, int(gap_days_threshold_used))
    if df is None or len(df) == 0:
        return {
            "requested_price_field": requested_price_field,
            "price_field_used": price_field_used,
            "price_field_warning": price_field_warning or "",
            "ohlc_scaled_by_adjclose": bool(ohlc_scaled_by_adjclose),
            "volume_inverse_scaled": bool(volume_inverse_scaled),
            "data_first_dt": "",
            "data_last_dt": "",
            "bars_downloaded": 0,
            "calendar_span_days": 0,
            "duplicate_timestamps_count": int(raw_duplicate_timestamps_count or 0),
            "is_monotonic_index": (
                bool(raw_is_monotonic_index)
                if raw_is_monotonic_index is not None else True
            ),
            "gap_stats": {"max_gap_days": 0, "n_gaps_over_3d": 0},
            "gap_days_threshold_used": gap_days_threshold_used,
            "pct_missing_est": 0.0,
            "has_splits": bool(has_splits),
            "has_dividends": bool(has_dividends),
            "has_split_events": bool(has_split_events),
            "has_dividend_events": bool(has_dividend_events),
            "any_nonpositive_prices": 0,
            "any_nan_prices": 0,
            "zero_volume_days": 0,
        }

    idx = pd.DatetimeIndex(df.index)
    bars = int(len(df))
    first = idx.min()
    last = idx.max()
    data_first_dt = _safe_iso_date(first)
    data_last_dt = _safe_iso_date(last)
    calendar_span_days = (
        int((pd.Timestamp(last).normalize() - pd.Timestamp(first).normalize()).days) + 1
        if bars > 0 else 0
    )

    duplicate_timestamps_count = int(
        raw_duplicate_timestamps_count
        if raw_duplicate_timestamps_count is not None
        else idx.duplicated().sum()
    )
    is_monotonic_index = (
        bool(raw_is_monotonic_index)
        if raw_is_monotonic_index is not None
        else bool(idx.is_monotonic_increasing)
    )

    unique_sorted = pd.DatetimeIndex(sorted(idx.unique()))
    max_gap_days = 0
    n_gaps_over_3d = 0
    if len(unique_sorted) > 1:
        deltas = np.diff(unique_sorted.values).astype("timedelta64[D]").astype(int)
        if len(deltas) > 0:
            max_gap_days = int(np.max(deltas))
            n_gaps_over_3d = int(np.sum(deltas >= gap_days_threshold_used))

    expected_weekdays = 0
    if bars > 0:
        start_d = pd.Timestamp(first).date()
        end_exclusive = (pd.Timestamp(last) + pd.Timedelta(days=1)).date()
        expected_weekdays = int(np.busday_count(start_d, end_exclusive))
    observed_trading_days = int(len(unique_sorted))
    missing_est = max(expected_weekdays - observed_trading_days, 0)
    pct_missing_est = round((missing_est / max(expected_weekdays, 1) * 100.0), 2)

    close_series = df["close"] if "close" in df.columns else pd.Series([], dtype=float)
    any_nan_prices = int(close_series.isna().sum()) if len(close_series) else 0
    any_nonpositive_prices = (
        int((close_series <= 0).sum()) if len(close_series) else 0
    )
    volume_series = df["volume"] if "volume" in df.columns else pd.Series([], dtype=float)
    zero_volume_days = int((pd.to_numeric(volume_series, errors="coerce").fillna(0) == 0).sum()) if len(volume_series) else 0

    return {
        "requested_price_field": requested_price_field,
        "price_field_used": price_field_used,
        "price_field_warning": price_field_warning or "",
        "ohlc_scaled_by_adjclose": bool(ohlc_scaled_by_adjclose),
        "volume_inverse_scaled": bool(volume_inverse_scaled),
        "data_first_dt": data_first_dt,
        "data_last_dt": data_last_dt,
        "bars_downloaded": bars,
        "calendar_span_days": calendar_span_days,
        "duplicate_timestamps_count": duplicate_timestamps_count,
        "is_monotonic_index": is_monotonic_index,
        "gap_stats": {
            "max_gap_days": max_gap_days,
            "n_gaps_over_3d": n_gaps_over_3d,
        },
        "gap_days_threshold_used": gap_days_threshold_used,
        "pct_missing_est": pct_missing_est,
        "has_splits": bool(has_splits),
        "has_dividends": bool(has_dividends),
        "has_split_events": bool(has_split_events),
        "has_dividend_events": bool(has_dividend_events),
        "any_nonpositive_prices": any_nonpositive_prices,
        "any_nan_prices": any_nan_prices,
        "zero_volume_days": zero_volume_days,
    }


def _append_data_quality_section(lines: List[str], data_quality: Optional[Dict[str, Any]]) -> None:
    if not data_quality:
        return
    gap_stats = data_quality.get("gap_stats", {}) if isinstance(data_quality, dict) else {}
    lines.append("=" * 74)
    lines.append("DATA QUALITY")
    lines.append("=" * 74)
    lines.append(
        f"  Price Field     : requested={data_quality.get('requested_price_field', '')} "
        f"used={data_quality.get('price_field_used', '')}"
    )
    lines.append(
        f"  Scaling         : ohlc_scaled_by_adjclose={bool(data_quality.get('ohlc_scaled_by_adjclose', False))}, "
        f"volume_inverse_scaled={bool(data_quality.get('volume_inverse_scaled', False))}"
    )
    warning = str(data_quality.get("price_field_warning", "") or "")
    if warning:
        lines.append(f"  WARNING         : {warning}")
    lines.append(f"  Data First Dt   : {data_quality.get('data_first_dt', '')}")
    lines.append(f"  Data Last Dt    : {data_quality.get('data_last_dt', '')}")
    lines.append(f"  Bars Downloaded : {int(data_quality.get('bars_downloaded', 0))}")
    lines.append(f"  Calendar Span   : {int(data_quality.get('calendar_span_days', 0))} days")
    lines.append(
        f"  Duplicate Timestamps : {int(data_quality.get('duplicate_timestamps_count', 0))}"
    )
    lines.append(
        f"  Monotonic Index : {bool(data_quality.get('is_monotonic_index', True))}"
    )
    lines.append(
        f"  Gap Stats       : max_gap_days={int(gap_stats.get('max_gap_days', 0))}, "
        f"n_gaps_over_3d={int(gap_stats.get('n_gaps_over_3d', 0))}"
    )
    lines.append(f"  Missing Est     : {float(data_quality.get('pct_missing_est', 0.0)):.2f}%")
    lines.append(
        f"  Actions Present : splits={bool(data_quality.get('has_splits', False))}, "
        f"dividends={bool(data_quality.get('has_dividends', False))}"
    )
    lines.append(
        f"  Action Events   : split_events={bool(data_quality.get('has_split_events', False))}, "
        f"dividend_events={bool(data_quality.get('has_dividend_events', False))}"
    )
    lines.append(
        f"  Price Issues    : nonpositive={int(data_quality.get('any_nonpositive_prices', 0))}, "
        f"nan={int(data_quality.get('any_nan_prices', 0))}"
    )
    lines.append(
        f"  Volume Issues   : zero_volume_days={int(data_quality.get('zero_volume_days', 0))}"
    )
    lines.append("")


def _build_bias_audit_payload(
    cfg: MeanReversionConfig,
    data_quality: Optional[Dict[str, Any]] = None,
    *,
    universe_name: str = "",
    universe_asof: str = "",
    universe_source: str = "",
) -> Dict[str, Any]:
    dq = data_quality if isinstance(data_quality, dict) else {}
    requested = str(dq.get("requested_price_field", "") or "").strip().lower()
    used = str(dq.get("price_field_used", requested or "close") or "close").strip().lower()
    scaled = bool(dq.get("ohlc_scaled_by_adjclose", False))
    if used == "adjclose" or scaled:
        dividend_assumption = (
            "assumes frictionless dividend reinvestment (total return proxy)"
        )
    else:
        dividend_assumption = "price-only; dividends not explicitly modeled"
    return {
        "data_source": "yfinance",
        "universe": "user-supplied ticker list (may be survivorship biased)",
        "universe_name": str(universe_name or ""),
        "universe_asof": str(universe_asof or ""),
        "universe_source": str(universe_source or ""),
        "price_series_mode": (
            f"requested={requested or used}, used={used}, "
            f"ohlc_scaled_by_adjclose={scaled}"
        ),
        "price_field_used": used,
        "ohlc_scaled_by_adjclose": scaled,
        "dividend_assumption": dividend_assumption,
        "taxes": "assumed 0%",
        "borrowing_leverage": "none",
        "slippage_commission_model": (
            f"slippage={cfg.slippage_pct*10000:.1f}bps, "
            f"commission=${cfg.commission_per_trade:.2f}/fill"
        ),
        "stop_loss_execution_realism": (
            "gap-stop aware: low<=stop triggers; if open<stop fill at open else at stop; "
            "clamped to [low, high], then slippage/commission applied"
        ),
        "cash_yield_annual_pct": float(cfg.cash_yield_annual_pct),
    }


def _curve_cagr_pct(curve: Any, bars_per_year: float = 252.0) -> Optional[float]:
    if curve is None:
        return None
    try:
        arr = list(curve)
    except TypeError:
        return None
    if len(arr) < 2:
        return None
    start = safe_number(arr[0], None)
    end = safe_number(arr[-1], None)
    if start is None or end is None or start <= 0:
        return None
    years = (len(arr) - 1) / float(bars_per_year)
    if years <= 0:
        return None
    cagr = (float(end) / float(start)) ** (1.0 / years) - 1.0
    return float(cagr * 100.0)


def _build_survivorship_sensitivity_payload(
    df: pd.DataFrame,
    cfg: MeanReversionConfig,
    tl_result: Optional[Dict[str, Any]],
    survivorship_drag_ann: float = 0.0,
) -> Dict[str, Any]:
    drag = float(safe_number(survivorship_drag_ann, 0.0) or 0.0)
    if tl_result is not None and cfg.two_layer_mode:
        core_weight = float(cfg.core_allocation_pct)
        core_curve = tl_result.get("core_equity_curve", [])
        baseline_curve = tl_result.get("baseline_equity_curve", [])
    else:
        core_weight = 1.0
        if len(df) >= 2 and float(df["close"].iloc[0]) > 0:
            rel = df["close"].astype(float) / float(df["close"].iloc[0])
            baseline_curve = (rel * float(cfg.starting_capital)).tolist()
            core_curve = baseline_curve
        else:
            core_curve = []
            baseline_curve = []

    core_cagr = _curve_cagr_pct(core_curve)
    static_baseline_cagr = _curve_cagr_pct(baseline_curve)
    core_cagr_adj = None if core_cagr is None else float(core_cagr) - drag
    static_baseline_cagr_adj = (
        None
        if static_baseline_cagr is None
        else float(static_baseline_cagr) - drag * float(core_weight)
    )
    return {
        "heuristic_sensitivity_only": True,
        "survivorship_drag_ann": drag,
        "core_weight": float(core_weight),
        "core_cagr": None if core_cagr is None else round(float(core_cagr), 6),
        "static_baseline_cagr": (
            None if static_baseline_cagr is None else round(float(static_baseline_cagr), 6)
        ),
        "core_cagr_adj": None if core_cagr_adj is None else round(float(core_cagr_adj), 6),
        "static_baseline_cagr_adj": (
            None
            if static_baseline_cagr_adj is None
            else round(float(static_baseline_cagr_adj), 6)
        ),
    }


def _append_bias_audit_section(
    lines: List[str],
    bias_audit: Optional[Dict[str, Any]],
    survivorship_sensitivity: Optional[Dict[str, Any]] = None,
) -> None:
    b = bias_audit if isinstance(bias_audit, dict) else {}
    s = survivorship_sensitivity if isinstance(survivorship_sensitivity, dict) else {}
    lines.append("=" * 74)
    lines.append("BIAS AUDIT / ASSUMPTIONS")
    lines.append("=" * 74)
    lines.append(f"  Data Source      : {b.get('data_source', 'yfinance')}")
    lines.append(f"  Universe         : {b.get('universe', 'user-supplied ticker list (may be survivorship biased)')}")
    lines.append(f"  Universe Name    : {b.get('universe_name', '')}")
    lines.append(f"  Universe As-Of   : {b.get('universe_asof', '')}")
    lines.append(f"  Universe Source  : {b.get('universe_source', '')}")
    lines.append(f"  Price Series     : {b.get('price_series_mode', '')}")
    lines.append(f"  Dividend Assump. : {b.get('dividend_assumption', '')}")
    lines.append(f"  Taxes            : {b.get('taxes', 'assumed 0%')}")
    lines.append(f"  Borrow/Leverage  : {b.get('borrowing_leverage', 'none')}")
    lines.append(f"  Slippage/Comm    : {b.get('slippage_commission_model', '')}")
    lines.append(f"  Stop Execution   : {b.get('stop_loss_execution_realism', '')}")
    lines.append(f"  Cash Yield       : {float(safe_number(b.get('cash_yield_annual_pct'), 0.0) or 0.0):.2f}% annualized")
    if s:
        lines.append(
            f"  Survivorship Drag: {float(safe_number(s.get('survivorship_drag_ann'), 0.0) or 0.0):.2f}% annual "
            "(heuristic sensitivity only)"
        )
        core_adj = safe_number(s.get("core_cagr_adj"), None)
        base_adj = safe_number(s.get("static_baseline_cagr_adj"), None)
        if core_adj is not None:
            lines.append(f"  Core CAGR adj    : {core_adj:+.4f}%")
        if base_adj is not None:
            lines.append(f"  Static CAGR adj  : {base_adj:+.4f}%")
    lines.append("")


def _compute_trade_excursions(
    trade_records: List[Dict[str, Any]],
    price_series: pd.Series,
    ratio_z_series: Optional[pd.Series] = None,
) -> None:
    """Add MAE/MFE + entry/exit z metrics to completed-trade records (analysis only)."""
    if not isinstance(trade_records, list) or len(trade_records) == 0:
        return
    if price_series is None or len(price_series) == 0:
        return

    n_prices = len(price_series)
    for rec in trade_records:
        entry_idx_raw = safe_number(rec.get("entry_bar_index"), None)
        exit_idx_raw = safe_number(rec.get("exit_bar_index"), None)
        entry_idx = int(entry_idx_raw) if entry_idx_raw is not None else -1
        exit_idx = int(exit_idx_raw) if exit_idx_raw is not None else -1
        if entry_idx < 0 or exit_idx < entry_idx or entry_idx >= n_prices:
            rec["mae_pct"] = None
            rec["mfe_pct"] = None
            rec["mae_abs"] = None
            rec["mfe_abs"] = None
            rec["entry_z"] = safe_number(rec.get("entry_ratio_z"), None)
            rec["exit_z"] = safe_number(rec.get("exit_ratio_z"), None)
            continue
        exit_idx = min(exit_idx, n_prices - 1)
        window = price_series.iloc[entry_idx: exit_idx + 1].dropna()
        entry_price = safe_number(rec.get("entry_price"), None)
        shares = safe_number(rec.get("shares"), 0.0) or 0.0
        if entry_price is None or entry_price <= 0 or len(window) == 0:
            rec["mae_pct"] = None
            rec["mfe_pct"] = None
            rec["mae_abs"] = None
            rec["mfe_abs"] = None
        else:
            min_px = float(window.min())
            max_px = float(window.max())
            mae_pct = (min_px / float(entry_price) - 1.0) * 100.0
            mfe_pct = (max_px / float(entry_price) - 1.0) * 100.0
            rec["mae_pct"] = round(float(mae_pct), 4)
            rec["mfe_pct"] = round(float(mfe_pct), 4)
            rec["mae_abs"] = round(float((min_px - float(entry_price)) * shares), 2)
            rec["mfe_abs"] = round(float((max_px - float(entry_price)) * shares), 2)

        entry_z = safe_number(rec.get("entry_ratio_z"), None)
        exit_z = safe_number(rec.get("exit_ratio_z"), None)
        if exit_z is None and ratio_z_series is not None and 0 <= exit_idx < len(ratio_z_series):
            v = ratio_z_series.iloc[exit_idx]
            exit_z = None if pd.isna(v) else float(v)
        rec["entry_z"] = entry_z
        rec["exit_z"] = exit_z


def _tail_diagnostics(trade_records: List[Dict[str, Any]]) -> Dict[str, float]:
    pnls = []
    maes = []
    mfes = []
    stop_hits = 0
    total = 0
    for rec in trade_records or []:
        pnl = safe_number(rec.get("realized_pnl", rec.get("pnl")), None)
        mae = safe_number(rec.get("mae_pct"), None)
        mfe = safe_number(rec.get("mfe_pct"), None)
        reason = str(rec.get("exit_reason", "") or "")
        if pnl is not None:
            pnls.append(float(pnl))
            total += 1
            if "STOP" in reason.upper():
                stop_hits += 1
        if mae is not None:
            maes.append(float(mae))
        if mfe is not None:
            mfes.append(float(mfe))

    def _pctile(arr: List[float], q: float) -> float:
        return float(np.percentile(arr, q)) if arr else 0.0

    return {
        "pnl_trade_mean": round(float(np.mean(pnls)) if pnls else 0.0, 4),
        "pnl_trade_median": round(float(np.median(pnls)) if pnls else 0.0, 4),
        "pnl_trade_p10": round(_pctile(pnls, 10), 4),
        "pnl_trade_p90": round(_pctile(pnls, 90), 4),
        "mae_pct_median": round(float(np.median(maes)) if maes else 0.0, 4),
        "mfe_pct_median": round(float(np.median(mfes)) if mfes else 0.0, 4),
        "stop_hit_rate": round((stop_hits / total) if total > 0 else 0.0, 4),
    }


def _build_config_hash(
    cfg: MeanReversionConfig,
    *,
    period: str,
    interval: str,
    price_field: str,
    require_min_bars: int,
) -> str:
    key_cfg = {
        "period": period,
        "interval": interval,
        "price_field": price_field,
        "require_min_bars": int(require_min_bars),
        "lookback_window": cfg.lookback_window,
        "ratio_anchor_window": cfg.ratio_anchor_window,
        "ratio_lookback": cfg.ratio_lookback,
        "entry_z": cfg.entry_z,
        "lag_signals_for_same_close": bool(cfg.lag_signals_for_same_close),
        "add_z": cfg.add_z,
        "trim_levels": [[float(z), float(f)] for z, f in cfg.trim_levels],
        "sell_z": cfg.sell_z,
        "exit_threshold": cfg.exit_threshold,
        "max_position_pct": cfg.max_position_pct,
        "add_step_pct": cfg.add_step_pct,
        "trim_step_pct": cfg.trim_step_pct,
        "min_cash_pct": cfg.min_cash_pct,
        "stop_atr_multiple": cfg.stop_atr_multiple,
        "target_atr_multiple": cfg.target_atr_multiple,
        "max_holding_days": cfg.max_holding_days,
        "entry_at": cfg.entry_at,
        "slippage_pct": cfg.slippage_pct,
        "commission_per_trade": cfg.commission_per_trade,
        "regime_filter_enabled": cfg.regime_filter_enabled,
        "regime_update_freq": int(cfg.regime_update_freq),
        "allowed_regimes": list(cfg.allowed_regimes),
        "ambiguous_policy": cfg.ambiguous_policy,
        "min_confidence": cfg.min_confidence,
        "two_layer_mode": cfg.two_layer_mode,
        "core_allocation_pct": cfg.core_allocation_pct,
        "tactical_allocation_pct": cfg.tactical_allocation_pct,
        "cash_yield_annual_pct": cfg.cash_yield_annual_pct,
        "rebalance_freq": cfg.rebalance_freq,
        "rebalance_drift_threshold": cfg.rebalance_drift_threshold,
        "max_zero_vol_days": cfg.max_zero_vol_days,
        "min_price_tail": cfg.min_price_tail,
        "tactical_vol_targeting_enabled": bool(cfg.tactical_vol_targeting_enabled),
        "tactical_vol_target": cfg.tactical_vol_target,
        "tactical_vol_window": cfg.tactical_vol_window,
        "tactical_vol_floor": cfg.tactical_vol_floor,
        "tactical_vol_cap": cfg.tactical_vol_cap,
        "cost_aware_entry_enabled": bool(cfg.cost_aware_entry_enabled),
        "cost_bps_est": cfg.cost_bps_est,
        "cost_k": cfg.cost_k,
        "better_exits_enabled": bool(cfg.better_exits_enabled),
        "tactical_exit_z": cfg.tactical_exit_z,
        "tactical_min_hold_days": cfg.tactical_min_hold_days,
        "tactical_max_hold_days": cfg.tactical_max_hold_days,
        "confidence_sizing_enabled": bool(cfg.confidence_sizing_enabled),
        "confidence_c0": cfg.confidence_c0,
        "confidence_gamma": cfg.confidence_gamma,
        "tactical_mode": cfg.tactical_mode,
        "tactical_max_positions": int(cfg.tactical_max_positions),
        "tactical_entry_z": cfg.tactical_entry_z,
        "tactical_weighting": cfg.tactical_weighting,
    }
    payload = json.dumps(key_cfg, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


# =============================================================================
# UNIVERSE FILTER (v3.4)
# =============================================================================

def universe_filter(
    df: pd.DataFrame,
    cfg: MeanReversionConfig,
    require_min_bars: Optional[int] = None,
    interval: str = "1d",
) -> Tuple[bool, List[str], Dict[str, Any]]:
    """Pre-run data quality + liquidity filter.

    Returns (passed, reasons, metrics) where reasons is non-empty on failure.
    """
    reasons: List[str] = []
    metrics: Dict[str, Any] = {
        "median_close": 0.0,
        "median_dollar_vol": 0.0,
        "tail_median_close": 0.0,
        "tail_median_dollar_vol": 0.0,
        "n_gaps_over_3d": 0,
        "gap_days_threshold_used": 0,
        "close_nan_count": 0,
        "volume_nan_count": 0,
        "zero_volume_days": 0,
        "nan_rows": 0,
        "nan_frac": 0.0,
        "usable_rows": 0,
        "min_required": 0,
    }

    if df is None or len(df) == 0:
        reasons.append("no_rows")
        return False, reasons, metrics

    missing_cols = [c for c in ("close", "volume") if c not in df.columns]
    if missing_cols:
        reasons.append(f"missing_columns={','.join(missing_cols)}")
        return False, reasons, metrics

    close_raw = pd.to_numeric(df["close"], errors="coerce")
    volume_raw = pd.to_numeric(df["volume"], errors="coerce")
    close_nan_count = int(close_raw.isna().sum())
    volume_nan_count = int(volume_raw.isna().sum())
    metrics["close_nan_count"] = close_nan_count
    metrics["volume_nan_count"] = volume_nan_count

    n_rows = int(len(df))
    nan_mask = close_raw.isna() | volume_raw.isna()
    nan_rows = int(nan_mask.sum())
    nan_frac = (float(nan_rows) / float(n_rows)) if n_rows > 0 else 0.0
    metrics["nan_rows"] = nan_rows
    metrics["nan_frac"] = round(nan_frac, 6)
    if nan_frac > float(cfg.max_nan_frac):
        reasons.append("too_many_nans")

    valid_mask = ~nan_mask
    close = close_raw[valid_mask]
    volume = volume_raw[valid_mask]
    dollar_vol = close * volume
    zero_volume_days = int((volume_raw.fillna(0) == 0).sum())
    metrics["zero_volume_days"] = zero_volume_days
    usable_rows = int(valid_mask.sum())
    if require_min_bars is not None and int(require_min_bars) > 0:
        min_required = int(require_min_bars)
    else:
        min_required = int(min(252, int(np.ceil(0.9 * n_rows)))) if n_rows > 0 else 0
    metrics["usable_rows"] = usable_rows
    metrics["min_required"] = min_required
    if usable_rows < min_required:
        reasons.append("insufficient_usable_bars")

    if len(close) == 0:
        reasons.append("no_valid_close_values")
    if len(volume) == 0:
        reasons.append("no_valid_volume_values")
    if len(dollar_vol) == 0:
        reasons.append("no_valid_dollar_volume_values")
    if zero_volume_days > int(cfg.max_zero_vol_days):
        reasons.append(
            f"zero_volume_days={zero_volume_days} > max_zero_vol_days={int(cfg.max_zero_vol_days)}"
        )

    median_close = float(close.median()) if len(close) > 0 else 0.0
    median_dollar_vol = float(dollar_vol.median()) if len(dollar_vol) > 0 else 0.0
    tail_window = 30
    if len(close) > 0:
        tail_close = close.iloc[-tail_window:]
        tail_dollar = dollar_vol.iloc[-tail_window:]
        tail_median_close = float(tail_close.median()) if len(tail_close) > 0 else 0.0
        tail_median_dollar_vol = float(tail_dollar.median()) if len(tail_dollar) > 0 else 0.0
    else:
        tail_median_close = 0.0
        tail_median_dollar_vol = 0.0

    # Gap diagnostics (reuse existing logic)
    n_gaps_over_3d = 0
    gap_days_threshold_used = (
        int(cfg.gap_days_threshold) if str(interval).strip().lower() == "1d" else 3
    )
    gap_days_threshold_used = max(1, int(gap_days_threshold_used))
    try:
        idx = pd.DatetimeIndex(df.index)
        unique_sorted = pd.DatetimeIndex(sorted(idx.unique()))
        if len(unique_sorted) > 1:
            deltas = np.diff(unique_sorted.values).astype("timedelta64[D]").astype(int)
            n_gaps_over_3d = int(np.sum(deltas >= gap_days_threshold_used))
    except Exception:
        reasons.append("invalid_datetime_index_for_gap_check")

    if median_close < cfg.min_price:
        reasons.append(
            f"median_close={median_close:.2f} < min_price={cfg.min_price:.2f}"
        )
    if median_dollar_vol < cfg.min_dollar_vol:
        reasons.append(
            f"median_dollar_vol={median_dollar_vol:,.0f} < "
            f"min_dollar_vol={cfg.min_dollar_vol:,.0f}"
        )
    if tail_median_close < float(cfg.min_price_tail):
        reasons.append(
            f"tail_median_close={tail_median_close:.2f} < min_price_tail={float(cfg.min_price_tail):.2f}"
        )
    if cfg.max_gaps > 0 and n_gaps_over_3d > cfg.max_gaps:
        reasons.append(
            f"n_gaps_over_3d={n_gaps_over_3d} > max_gaps={cfg.max_gaps}"
        )

    metrics["median_close"] = round(median_close, 4)
    metrics["median_dollar_vol"] = round(median_dollar_vol, 2)
    metrics["tail_median_close"] = round(tail_median_close, 4)
    metrics["tail_median_dollar_vol"] = round(tail_median_dollar_vol, 2)
    metrics["n_gaps_over_3d"] = n_gaps_over_3d
    metrics["gap_days_threshold_used"] = int(gap_days_threshold_used)

    passed = len(reasons) == 0
    return passed, reasons, metrics


def _generate_skipped_report(
    ticker: str,
    cfg: MeanReversionConfig,
    reasons: List[str],
    filter_metrics: Dict[str, Any],
    data_quality: Optional[Dict[str, Any]] = None,
    bias_audit: Optional[Dict[str, Any]] = None,
) -> str:
    """Generate a report for tickers skipped by the universe filter."""
    lines: List[str] = []
    lines.append("=" * 74)
    lines.append(f"{'SKIPPED BY UNIVERSE FILTER':^74}")
    lines.append(f"{ticker}")
    lines.append("=" * 74)
    lines.append(f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("Status: SKIPPED BY UNIVERSE FILTER")
    lines.append("")
    lines.append("Reasons:")
    for r in reasons:
        lines.append(f"  - {r}")
    lines.append("")
    lines.append("Filter Metrics:")
    for k, v in filter_metrics.items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    _append_data_quality_section(lines, data_quality)
    if bias_audit:
        _append_bias_audit_section(lines, bias_audit)
    lines.append("=" * 74)
    lines.append("DISCLAIMER: Educational only. NOT financial advice.")
    lines.append("=" * 74)
    return "\n".join(lines)


# =============================================================================
# DATA FETCHING
# =============================================================================

def fetch_price_data(
    ticker: str,
    period: str = "500d",
    interval: str = "1d",
    price_field: str = "adjclose",
    use_adjclose_scaling: bool = True,
) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    if not YF_OK:
        return None, {"error": "yfinance_not_available"}
    try:
        raw = yf.Ticker(ticker).history(
            period=period, interval=interval, auto_adjust=False, actions=True,
        )
        if raw.empty:
            return None, {"error": "empty_history"}

        meta: Dict[str, Any] = {
            "requested_price_field": str(price_field).strip().lower(),
            "price_field_used": "close",
            "price_field_warning": "",
            "ohlc_scaled_by_adjclose": False,
            "volume_inverse_scaled": False,
            "raw_duplicate_timestamps_count": int(pd.DatetimeIndex(raw.index).duplicated().sum()),
            "raw_is_monotonic_index": bool(pd.DatetimeIndex(raw.index).is_monotonic_increasing),
        }

        df = raw.copy()
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]
        if "adj_close" in df.columns:
            df["adjclose"] = df["adj_close"]
        elif "adjclose" not in df.columns:
            df["adjclose"] = np.nan

        req_field = meta["requested_price_field"]
        use_adj = (req_field == "adjclose")
        has_adj = ("adjclose" in df.columns) and bool(df["adjclose"].notna().any())
        if use_adj and has_adj and use_adjclose_scaling:
            df, _ = _apply_adjclose_price_volume_scaling(df)
            meta["price_field_used"] = "adjclose"
            meta["ohlc_scaled_by_adjclose"] = True
            meta["volume_inverse_scaled"] = True
        elif use_adj and not has_adj:
            meta["price_field_used"] = "close"
            meta["price_field_warning"] = (
                "Adj Close unavailable from provider; fell back to Close."
            )
        else:
            meta["price_field_used"] = "close"

        if "dividends" in df.columns:
            meta["has_dividends"] = True
            meta["has_dividend_events"] = bool((df["dividends"].fillna(0) != 0).any())
        else:
            meta["has_dividends"] = False
            meta["has_dividend_events"] = False
        if "stock_splits" in df.columns:
            meta["has_splits"] = True
            meta["has_split_events"] = bool((df["stock_splits"].fillna(0) != 0).any())
        else:
            meta["has_splits"] = False
            meta["has_split_events"] = False

        df = normalize_dataframe(df)
        df["return"] = df["close"].pct_change()
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))
        df["volume_sma_20"] = df["volume"].rolling(20).mean()
        df["volume_ratio"] = df["volume"] / df["volume_sma_20"]
        return df, meta
    except Exception as exc:
        return None, {"error": str(exc)}


def fetch_basic_info(ticker: str) -> Dict:
    if not YF_OK:
        return {"ticker": ticker, "name": ticker}
    try:
        info = yf.Ticker(ticker).info
        return {
            "ticker": ticker,
            "name": info.get("longName") or info.get("shortName") or ticker,
            "sector": info.get("sector", "Unknown"),
            "current_price": safe_number(info.get("currentPrice"))
            or safe_number(info.get("regularMarketPrice")),
        }
    except Exception:
        return {"ticker": ticker, "name": ticker}


# =============================================================================
# TECHNICAL INDICATORS
# =============================================================================

def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calculate_bollinger(prices: pd.Series, period: int = 20) -> Dict:
    sma = prices.rolling(period).mean()
    std = prices.rolling(period).std()
    return {"upper": sma + 2 * std, "middle": sma, "lower": sma - 2 * std}


def calculate_moving_averages(prices: pd.Series) -> Dict:
    return {
        "sma_10": prices.rolling(10).mean(),
        "sma_20": prices.rolling(20).mean(),
        "sma_50": prices.rolling(50).mean(),
        "sma_200": prices.rolling(200).mean(),
    }


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"].shift(1)
    tr = pd.concat([h - l, abs(h - c), abs(l - c)], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def calculate_macd(prices: pd.Series) -> Dict:
    ema12 = prices.ewm(span=12, adjust=False).mean()
    ema26 = prices.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return {"macd": macd, "signal": signal, "histogram": macd - signal}


def get_technicals(df: pd.DataFrame) -> Dict:
    if df is None or len(df) < 50:
        return {}
    close = df["close"]
    price = float(close.iloc[-1])
    rsi_s = calculate_rsi(close)
    rsi = float(rsi_s.iloc[-1]) if not pd.isna(rsi_s.iloc[-1]) else None
    macd_d = calculate_macd(close)
    macd_hist = float(macd_d["histogram"].iloc[-1]) if not pd.isna(macd_d["histogram"].iloc[-1]) else None
    bb = calculate_bollinger(close)
    bb_upper = float(bb["upper"].iloc[-1]) if not pd.isna(bb["upper"].iloc[-1]) else None
    bb_lower = float(bb["lower"].iloc[-1]) if not pd.isna(bb["lower"].iloc[-1]) else None
    mas = calculate_moving_averages(close)
    sma_50 = float(mas["sma_50"].iloc[-1]) if not pd.isna(mas["sma_50"].iloc[-1]) else None
    sma_200 = float(mas["sma_200"].iloc[-1]) if not pd.isna(mas["sma_200"].iloc[-1]) else None
    sma_20 = float(mas["sma_20"].iloc[-1]) if not pd.isna(mas["sma_20"].iloc[-1]) else None
    atr_s = calculate_atr(df)
    atr_val = float(atr_s.iloc[-1]) if not pd.isna(atr_s.iloc[-1]) else None
    bb_pos = None
    if bb_upper and bb_lower and bb_upper > bb_lower:
        bb_pos = (price - bb_lower) / (bb_upper - bb_lower) * 100
    return {
        "rsi": round(rsi, 1) if rsi else None,
        "rsi_signal": "OVERSOLD" if rsi and rsi < 30 else ("OVERBOUGHT" if rsi and rsi > 70 else "NEUTRAL"),
        "macd_histogram": round(macd_hist, 3) if macd_hist else None,
        "macd_trend": "BULLISH" if macd_hist and macd_hist > 0 else "BEARISH",
        "bb_upper": round(bb_upper, 2) if bb_upper else None,
        "bb_lower": round(bb_lower, 2) if bb_lower else None,
        "bb_position": round(bb_pos, 1) if bb_pos is not None else None,
        "sma_20": round(sma_20, 2) if sma_20 else None,
        "sma_50": round(sma_50, 2) if sma_50 else None,
        "sma_200": round(sma_200, 2) if sma_200 else None,
        "above_sma_50": price > sma_50 if sma_50 else None,
        "above_sma_200": price > sma_200 if sma_200 else None,
        "atr": round(atr_val, 2) if atr_val else None,
        "atr_pct": round(atr_val / price * 100, 2) if atr_val else None,
    }


# =============================================================================
# STATISTICAL TESTS
# =============================================================================

def calculate_hurst_exponent(
    prices: pd.Series, max_lag: int = 100,
) -> Tuple[Optional[float], Optional[float]]:
    """R/S Hurst exponent with R-squared quality gate."""
    clean = pd.Series(pd.to_numeric(prices, errors="coerce"), dtype=float).dropna()
    if len(clean) < 40:
        return None, None
    diffs = clean.diff().dropna().to_numpy(dtype=float)
    if len(diffs) > 0 and np.allclose(diffs, 0.0):
        # Constant series: treat as random-walk baseline instead of failing.
        return 0.5, 1.0
    ts = np.log(clean.values)
    n = len(ts)
    log_lags, log_rs = [], []
    for lag in range(2, min(max_lag, n // 2)):
        rs_list = []
        for start in range(0, n - lag, lag):
            seg = ts[start: start + lag]
            dev = seg - seg.mean()
            cum = np.cumsum(dev)
            r = cum.max() - cum.min()
            s = seg.std(ddof=1)
            if s > 1e-12:
                rs_list.append(r / s)
        if rs_list:
            log_lags.append(np.log(lag))
            log_rs.append(np.log(np.mean(rs_list)))
    if len(log_lags) < 5:
        return None, None
    try:
        result = np.polyfit(log_lags, log_rs, 1, full=True)
        coeffs, residuals = result[0], result[1]
    except Exception:
        return None, None
    hurst = float(coeffs[0])
    ss_res = residuals[0] if len(residuals) > 0 else 0.0
    y_arr = np.array(log_rs)
    ss_tot = float(np.sum((y_arr - y_arr.mean()) ** 2))
    r_sq = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    if r_sq < 0.80:
        return None, None
    # Clamp to valid range [0, 1]; out-of-range values arise from
    # noisy short windows and would confuse downstream logic.
    if hurst < 0 or hurst > 1:
        logger.debug("Hurst %.3f out of [0,1], clamped", hurst)
        hurst = max(0.0, min(1.0, hurst))
    return round(hurst, 3), round(r_sq, 3)


def calculate_half_life(prices: pd.Series) -> Optional[float]:
    s = np.asarray(prices.dropna().to_numpy(dtype=float), dtype=float)
    if len(s) < 30:
        return None
    y = np.diff(s)
    x = np.column_stack([s[:-1], np.ones(len(s) - 1)])
    try:
        beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    except Exception:
        return None
    if beta[0] >= 0:
        return None
    hl = -np.log(2) / beta[0]
    return float(round(hl, 1)) if 0 < hl < 500 else None


def calculate_variance_ratio(prices: pd.Series, q: int = 5) -> Optional[float]:
    lp = np.log(np.asarray(prices.dropna().to_numpy(dtype=float), dtype=float))
    n = len(lp)
    if n < q * 10:
        return None
    r1 = np.diff(lp)
    rq = lp[q:] - lp[:-q]
    v1 = np.var(r1, ddof=1)
    if v1 < 1e-15:
        return None
    return float(round(np.var(rq, ddof=1) / (q * v1), 4))


# P0.1 — robust ADF with note field + debug logging
def adf_test(prices: pd.Series, debug: bool = False) -> Dict:
    """
    Always returns {adf_stat, p_value, is_stationary, note}.
    Never raises.  ``note`` explains any N/A.
    """
    base = {"adf_stat": None, "p_value": None, "is_stationary": None, "note": ""}

    if not SM_OK:
        base["note"] = "statsmodels not available"
        if debug:
            logger.warning("ADF skipped: statsmodels not installed")
        return base

    s = np.asarray(prices.dropna().to_numpy(dtype=float), dtype=float)
    if len(s) < 30:
        base["note"] = f"insufficient data ({len(s)} < 30)"
        if debug:
            logger.warning("ADF skipped: only %d observations", len(s))
        return base

    if np.std(s) < 1e-12:
        base["note"] = "constant series"
        if debug:
            logger.warning("ADF skipped: constant series")
        return base

    try:
        res = adfuller(s, maxlag=min(20, len(s) // 3 - 1), autolag="AIC")
        p = float(res[1])
        return {
            "adf_stat": round(res[0], 4),
            "p_value": round(p, 4),
            "is_stationary": p < 0.05,
            "note": "",
        }
    except Exception as exc:
        base["note"] = f"exception: {str(exc)[:60]}"
        if debug:
            logger.warning("ADF exception: %s", exc)
        return base


def calculate_mean_reversion_metrics(df: pd.DataFrame, window: int = 20,
                                     debug: bool = False) -> Dict:
    close = df["close"].dropna()
    hurst, hurst_r2 = calculate_hurst_exponent(close)
    half_life = calculate_half_life(close)
    vr = calculate_variance_ratio(close, q=5)
    adf = adf_test(close, debug=debug)

    ev_for = ev_against = total = 0
    tests_detail = []

    for label, test_val, is_for in [
        ("hurst", hurst, hurst is not None and hurst < 0.50),
        ("adf", adf.get("is_stationary"), adf.get("is_stationary") is True),
        ("vr", vr, vr is not None and vr < 1.0),
        ("half_life", half_life, half_life is not None and 1 < half_life < 60),
    ]:
        # Skip if result is None (unless it's a bool False)
        if test_val is None and not isinstance(test_val, bool):
            tests_detail.append((label, "N/A", None))
            continue
        total += 1
        if is_for:
            ev_for += 1
            tests_detail.append((label, "FOR", True))
        else:
            ev_against += 1
            tests_detail.append((label, "AGAINST", False))

    # P1.2: verdict is now a simple label; the real regime verdict
    # comes from classify_regime(). Keep this for stat-test summary.
    if total == 0:
        verdict = "INSUFFICIENT_DATA"
    elif ev_for >= 3:
        verdict = "STRONG_MEAN_REVERTING"
    elif ev_for >= 2:
        verdict = "MODERATE_MEAN_REVERTING"
    elif ev_against >= 3:
        verdict = "TRENDING"
    else:
        verdict = "MIXED"

    def _interp(val, thr, lo, hi):
        return (lo if val < thr else hi) if val is not None else None

    return {
        "hurst_exponent": hurst, "hurst_r_squared": hurst_r2,
        "hurst_interpretation": _interp(hurst, 0.5, "MEAN_REVERTING", "TRENDING"),
        "adf_stat": adf.get("adf_stat"), "adf_p_value": adf.get("p_value"),
        "is_stationary": adf.get("is_stationary"), "adf_note": adf.get("note", ""),
        "half_life_days": half_life, "variance_ratio": vr,
        "variance_ratio_interpretation": _interp(vr, 1.0, "MEAN_REVERTING", "TRENDING"),
        "stat_verdict": verdict, "verdict": verdict,
        "evidence_for": ev_for, "evidence_against": ev_against,
        "total_tests": total, "tests_detail": tests_detail,
    }


# =============================================================================
# RATIO ANCHOR FRAMEWORK
# =============================================================================

def compute_ratio_series(
    df: pd.DataFrame, cfg: MeanReversionConfig,
    custom_ratio: Optional[pd.Series] = None,
) -> pd.Series:
    """Price / anchor ratio. Plug-in: pass custom_ratio for a precomputed series."""
    if custom_ratio is not None:
        return custom_ratio
    close = df["close"]
    if cfg.ratio_mode == "price_to_ema":
        anchor = close.ewm(span=cfg.ratio_anchor_window, adjust=False).mean()
    else:
        anchor = close.rolling(cfg.ratio_anchor_window).mean()
    return close / anchor.replace(0, np.nan)


def compute_ratio_z(ratio: pd.Series, lookback: int = 60) -> pd.Series:
    """Z-score of the ratio relative to its own rolling statistics."""
    mu = ratio.rolling(lookback).mean()
    sigma = ratio.rolling(lookback).std().replace(0, np.nan)
    return (ratio - mu) / sigma


# =============================================================================
# REGIME SIGNAL ABSTRACTION (Phase 2 prep — no probabilistic math yet)
# =============================================================================

@dataclass
class RegimeSignal:
    """Wrapper carrying regime information to the action engine.

    Current implementation: hard labels with confidence = 0.0 or 1.0.
    Phase 2 (future): populate ``beliefs`` with a probability vector from
    an HMM or Bayesian filter — trading decisions can then scale sizing
    by P(MR-friendly) instead of using a binary gate.

    Attributes:
        label:      Hard regime label (backward-compatible).
        confidence: 0.0–1.0 confidence in the label.  Currently binary.
        score:      Raw MR-score from the classifier [0, 1].
    beliefs:    Probabilistic state beliefs in order:
                [MEAN_REVERTING, SIDEWAYS, TRENDING, AMBIGUOUS].
                Defaults to uniform prior.
    """
    label: str = Regime.AMBIGUOUS.value
    confidence: float = 1.0
    score: float = 0.5
    # TODO Phase 2: replace this prior with filtered HMM posterior.
    beliefs: Optional[np.ndarray] = field(
        default_factory=lambda: np.full(4, 0.25, dtype=float),
    )

    @staticmethod
    def from_hard_label(label: str, score: float = 0.5) -> "RegimeSignal":
        """Construct from a hard label (current v3.x behavior)."""
        one_hot = np.zeros(4, dtype=float)
        idx_map = {
            Regime.MEAN_REVERTING.value: 0,
            Regime.SIDEWAYS.value: 1,
            Regime.TRENDING.value: 2,
            Regime.AMBIGUOUS.value: 3,
        }
        one_hot[idx_map.get(label, 3)] = 1.0
        return RegimeSignal(label=label, confidence=1.0, score=score, beliefs=one_hot)

    @property
    def mr_probability(self) -> float:
        """Estimated probability the regime is MR-friendly.

        TODO Phase 2: return beliefs[MEAN_REVERTING] + beliefs[SIDEWAYS]
        when beliefs is populated.
        """
        if self.beliefs is not None:
            if isinstance(self.beliefs, dict):
                return (
                    float(self.beliefs.get(Regime.MEAN_REVERTING.value, 0.0))
                    + float(self.beliefs.get(Regime.SIDEWAYS.value, 0.0))
                )
            arr = np.asarray(self.beliefs, dtype=float).reshape(-1)
            if arr.size >= 2:
                return float(arr[0] + arr[1])
        # Fallback: binary from label
        if self.label in (Regime.MEAN_REVERTING.value, Regime.SIDEWAYS.value):
            return 1.0
        return 0.0


# =============================================================================
# REGIME CLASSIFIER (P1.1 — adaptive thresholds)
# =============================================================================

def _compute_regime_raw_scores(
    df: pd.DataFrame, cfg: MeanReversionConfig,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Compute raw MR-score and trend-strength per bar."""
    close = df["close"]
    n = len(df)
    win = cfg.regime_lookback
    mr_scores = pd.Series(np.nan, index=df.index, dtype=float)
    trend_strengths = pd.Series(np.nan, index=df.index, dtype=float)
    tests_count = pd.Series(0, index=df.index, dtype=int)
    update_freq = max(int(getattr(cfg, "regime_update_freq", 1) or 1), 1)
    compute_idx = list(range(win, n, update_freq))
    if n > win and (n - 1) not in compute_idx:
        compute_idx.append(n - 1)

    # Spec v1 look-ahead fix: if entry_at == "same_close", exclude current bar
    exclude_current = (cfg.entry_at == "same_close" and cfg.enable_spec_v1_upgrades)

    for i in compute_idx:
        end_idx = i if exclude_current else i + 1  # Spec v1: df.iloc[:i] for same_close
        window = close.iloc[max(0, i - win): end_idx]
        if len(window) < 30:
            continue
        ev_for = 0
        tests = 0

        h, _ = calculate_hurst_exponent(window, max_lag=50)
        if h is not None:
            tests += 1
            if h < 0.50:
                ev_for += 1

        vr = calculate_variance_ratio(window, q=5)
        if vr is not None:
            tests += 1
            if vr < 1.0:
                ev_for += 1

        hl = calculate_half_life(window)
        if hl is not None:
            tests += 1
            if 1 < hl < 60:
                ev_for += 1

        score = ev_for / max(tests, 1)

        ret_total = (window.iloc[-1] / window.iloc[0] - 1) if window.iloc[0] > 0 else 0
        vol = window.pct_change().std() * np.sqrt(252) if len(window) > 2 else 0.3
        ts_val = abs(ret_total) / max(vol, 0.01)

        mr_scores.iloc[i] = round(score, 3)
        trend_strengths.iloc[i] = round(ts_val, 3)
        tests_count.iloc[i] = int(tests)

    if update_freq > 1:
        mr_scores = mr_scores.ffill()
        trend_strengths = trend_strengths.ffill()
        tests_count = tests_count.replace(0, np.nan).ffill().fillna(0).astype(int)
        if cfg.debug:
            skipped = max(0, (n - win) - len(compute_idx))
            logger.debug(
                "Regime stats downsampled: update_freq=%d computed=%d skipped=%d",
                update_freq, len(compute_idx), skipped,
            )

    return mr_scores, trend_strengths, tests_count


def classify_regime(
    df: pd.DataFrame, cfg: MeanReversionConfig,
) -> Tuple[pd.Series, pd.Series]:
    """
    Produce per-bar regime labels and regime scores [0, 1].

    P1.1: Uses adaptive percentile thresholds so SIDEWAYS actually appears.
    """
    mr_scores, trend_strengths, tests_count = _compute_regime_raw_scores(df, cfg)
    n = len(df)
    labels = pd.Series(Regime.AMBIGUOUS.value, index=df.index, dtype=str)
    final_scores = pd.Series(0.5, index=df.index, dtype=float)

    valid_mr = mr_scores.dropna()
    valid_ts = trend_strengths.dropna()

    if cfg.regime_adaptive_thresholds and len(valid_mr) > 20 and len(valid_ts) > 20:
        # Adaptive thresholds from data distribution
        mr_high = float(valid_mr.quantile(0.70))
        mr_low = float(valid_mr.quantile(0.30))
        ts_high = float(valid_ts.quantile(0.70))
        ts_low = float(valid_ts.quantile(0.35))
    else:
        # Fallback fixed thresholds
        mr_high, mr_low = 0.65, 0.35
        ts_high, ts_low = 1.2, 0.5

    for i in range(n):
        sc = mr_scores.iloc[i]
        ts = trend_strengths.iloc[i]
        if pd.isna(sc) or pd.isna(ts):
            continue

        # When statistical tests can't discriminate (all ~0.5),
        # lean heavily on trend_strength as the primary signal.
        no_evidence = int(tests_count.iloc[i]) == 0
        tests_informative = (sc != 0.5) and not no_evidence  # 0.5 means 0/0 or equal for/against

        if no_evidence:
            # Preserve gating behavior from prior logic: no-evidence bars block entries.
            adj_score = 0.0
        elif tests_informative:
            adj_score = sc
            if ts > ts_high:
                adj_score *= 0.5  # Strong penalty for trending + MR evidence
        else:
            # Tests uninformative → classify by trend strength alone
            if ts > ts_high:
                adj_score = 0.15  # trending
            elif ts < ts_low:
                adj_score = 0.50  # sideways candidate
            else:
                adj_score = 0.35  # ambiguous

        final_scores.iloc[i] = round(adj_score, 3)

        if no_evidence:
            labels.iloc[i] = "NO_EVIDENCE"
        elif adj_score >= mr_high and tests_informative:
            # SIDEWAYS overrides MR when trend is very low
            if ts <= ts_low:
                labels.iloc[i] = Regime.SIDEWAYS.value
            else:
                labels.iloc[i] = Regime.MEAN_REVERTING.value
        elif ts <= ts_low and adj_score >= mr_low:
            labels.iloc[i] = Regime.SIDEWAYS.value
        elif adj_score < mr_low or ts > ts_high:
            labels.iloc[i] = Regime.TRENDING.value
        else:
            labels.iloc[i] = Regime.AMBIGUOUS.value

    return labels, final_scores


def regime_allows_action(
    action: Action, regime: str, cfg: MeanReversionConfig,
) -> bool:
    """Check if the regime filter permits the given action."""
    if not cfg.regime_filter_enabled:
        return True
    if action in (Action.HOLD, Action.BLOCKED, Action.REDUCE, Action.SELL):
        return True
    # BUY / ADD gated
    if regime in cfg.allowed_regimes:
        return True
    if regime == Regime.AMBIGUOUS.value:
        if cfg.ambiguous_policy == "allow_small":
            return True
        if cfg.ambiguous_policy == "tighten":
            return True
        return False  # "block"
    return False


# =============================================================================
# REVERSAL CONFIRMATION
# =============================================================================

def check_reversal_confirmation(
    df: pd.DataFrame, idx: int, cfg: MeanReversionConfig,
) -> bool:
    if not cfg.require_reversal_confirmation:
        return True
    if idx < 2:
        return False
    confirmed = False
    close = df["close"]
    for method in cfg.confirmation_methods:
        if method == "rsi_turning":
            rsi = calculate_rsi(close, 14)
            if idx < len(rsi) and idx - 1 >= 0:
                if not pd.isna(rsi.iloc[idx]) and not pd.isna(rsi.iloc[idx - 1]):
                    if rsi.iloc[idx] > rsi.iloc[idx - 1]:
                        confirmed = True
        elif method == "close_above_prior":
            if close.iloc[idx] > close.iloc[idx - 1]:
                confirmed = True
    return confirmed


# =============================================================================
# QUALITY FILTER
# =============================================================================

def passes_quality_filter(df: pd.DataFrame, idx: int, cfg: MeanReversionConfig) -> bool:
    close = df["close"]
    if cfg.require_above_sma200:
        sma200 = close.rolling(200).mean()
        if idx < len(sma200) and not pd.isna(sma200.iloc[idx]):
            if close.iloc[idx] < sma200.iloc[idx]:
                return False
    if cfg.require_positive_12m_momentum:
        if idx >= 252:
            if close.iloc[idx] / close.iloc[idx - 252] - 1 < 0:
                return False
    return True


# =============================================================================
# ACTION ENGINE
# =============================================================================

def _get_trim_action(ratio_z: float, cfg: MeanReversionConfig) -> Optional[Tuple[Action, float]]:
    """
    P2.2: Staged exits — check trim_levels from highest to lowest z.
    Returns (action, fraction_to_sell) or None.
    """
    levels = sorted(cfg.trim_levels, key=lambda t: t[0], reverse=True)
    for z_thr, frac in levels:
        if ratio_z >= z_thr:
            action = Action.SELL if frac >= 1.0 else Action.REDUCE
            return action, frac
    return None


def determine_action(
    ratio_z: float,
    regime: str,
    has_position: bool,
    position_pct: float,
    cfg: MeanReversionConfig,
    reversal_ok: bool,
    quality_ok: bool,
) -> Action:
    """Map current state to BUY / ADD / HOLD / REDUCE / SELL."""
    # Overvalued exits first (always allowed)
    if has_position:
        trim = _get_trim_action(ratio_z, cfg)
        if trim is not None:
            return trim[0]  # SELL or REDUCE

    # Entry / add logic
    if ratio_z <= cfg.entry_z and not has_position:
        if not reversal_ok or not quality_ok:
            return Action.HOLD
        action = Action.BUY
        if not regime_allows_action(action, regime, cfg):
            return Action.BLOCKED
        return action

    if ratio_z <= cfg.add_z and has_position:
        if position_pct >= cfg.max_position_pct:
            return Action.HOLD
        if not reversal_ok:
            return Action.HOLD
        action = Action.ADD
        if not regime_allows_action(action, regime, cfg):
            return Action.BLOCKED
        return action

    return Action.HOLD


# =============================================================================
# POSITION STATE
# =============================================================================

@dataclass
class _PositionState:
    """Tracks a single-name long position with cost-basis accounting."""
    shares: int = 0
    cost_basis: float = 0.0
    entry_bar: int = -1
    last_add_bar: int = -1
    stop_price: float = 0.0
    neg_slope_count: int = 0   # P0.2: consecutive negative SMA slope bars

    @property
    def avg_cost(self) -> float:
        return self.cost_basis / self.shares if self.shares > 0 else 0.0

    @property
    def is_open(self) -> bool:
        return self.shares > 0


def _summarize_confidence_buckets(
    trade_records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Aggregate per-trade confidence diagnostics for reporting/serialization."""
    bucket_order = ["<0.60", "0.60-0.70", "0.70-0.80", "0.80-0.90", "0.90-1.00", "NA"]
    bucket_stats = {
        b: {"trades": 0, "wins": 0, "pnl_sum": 0.0, "hold_sum": 0.0}
        for b in bucket_order
    }

    for rec in trade_records:
        b = rec.get("bucket", "NA")
        if b not in bucket_stats:
            bucket_stats[b] = {"trades": 0, "wins": 0, "pnl_sum": 0.0, "hold_sum": 0.0}
        bucket_stats[b]["trades"] += 1
        pnl = float(rec.get("pnl", 0.0))
        hold = float(rec.get("hold_days", 0.0))
        if pnl > 0:
            bucket_stats[b]["wins"] += 1
        bucket_stats[b]["pnl_sum"] += pnl
        bucket_stats[b]["hold_sum"] += hold

    rows: List[Dict[str, Any]] = []
    for b in bucket_order:
        st = bucket_stats[b]
        trades = int(st["trades"])
        win_rate = (st["wins"] / trades * 100.0) if trades > 0 else 0.0
        avg_pnl = (st["pnl_sum"] / trades) if trades > 0 else 0.0
        avg_hold = (st["hold_sum"] / trades) if trades > 0 else 0.0
        rows.append({
            "bucket": b,
            "trades": trades,
            "win_rate_pct": round(float(win_rate), 1),
            "avg_pnl": round(float(avg_pnl), 2),
            "avg_hold_days": round(float(avg_hold), 1),
            "low_sample": trades < 5,
        })

    return {
        "total_trades": len(trade_records),
        "rows": rows,
    }


# =============================================================================
# BACKTEST ENGINE v3.1
# =============================================================================

class BacktestEngine:
    """
    Bar-by-bar long-only simulator with partial fills.
    v3.1 fixes: thesis-break consecutive, dust filter, staged trims, vol sizing.
    """

    def __init__(self, config: Optional[MeanReversionConfig] = None):
        self.cfg = config or MeanReversionConfig()

    def run(
        self,
        df: pd.DataFrame,
        ratio_z_series: pd.Series,
        regime_labels: pd.Series,
        regime_scores: pd.Series,
    ) -> Dict:
        cfg = self.cfg
        cash = cfg.starting_capital
        pos = _PositionState()
        fills: List[Dict] = []
        actions_log: List[Dict] = []
        trade_records: List[Dict[str, Any]] = []
        open_trade_conf: Optional[float] = None
        open_trade_entry_idx: Optional[int] = None
        open_trade_entry_dt: Optional[str] = None
        open_trade_entry_price: Optional[float] = None
        open_trade_total_entry_shares: int = 0
        open_trade_regime_label: Optional[str] = None
        open_trade_ratio_z: Optional[float] = None
        trade_realized_pnl = 0.0
        equity_curve: List[float] = []
        exposure_pct_curve: List[float] = []   # tactical exposure each bar
        blocked_count = 0
        blocked_by_regime = 0
        blocked_by_confidence = 0
        blocked_by_cost = 0
        regime_no_evidence_bars = 0
        n = len(df)
        accrued_cash_yield = 0.0

        def _entry_confidence(
            i: int, labels: pd.Series, N: int = 60,
        ) -> Optional[float]:
            return _entry_confidence_from_labels(i, labels, N=N)

        def _bucket(conf: Optional[float]) -> str:
            if conf is None or pd.isna(conf):
                return "NA"
            if conf < 0.60:
                return "<0.60"
            if conf < 0.70:
                return "0.60-0.70"
            if conf < 0.80:
                return "0.70-0.80"
            if conf < 0.90:
                return "0.80-0.90"
            return "0.90-1.00"

        def _record_trade_close(
            close_bar_idx: int,
            final_fill_realized_pnl: float,
            close_bar_date: Optional[str] = None,
            exit_price: Optional[float] = None,
            exit_reason: str = "EXIT",
            exit_ratio_z: Optional[float] = None,
        ) -> None:
            nonlocal open_trade_conf, open_trade_entry_idx
            nonlocal open_trade_entry_dt, open_trade_entry_price, open_trade_total_entry_shares
            nonlocal open_trade_regime_label, open_trade_ratio_z
            nonlocal trade_realized_pnl
            hold_days = (
                close_bar_idx - open_trade_entry_idx
                if open_trade_entry_idx is not None else 0
            )
            total_trade_pnl = trade_realized_pnl + float(final_fill_realized_pnl)
            total_trade_pnl_rounded = float(round(total_trade_pnl, 2))
            conf_bucket = _bucket(open_trade_conf)
            trade_records.append({
                "conf": open_trade_conf,
                "entry_confidence": open_trade_conf,
                "entry_conf_bucket": conf_bucket,
                "bucket": conf_bucket,
                "pnl": total_trade_pnl_rounded,
                "realized_pnl": total_trade_pnl_rounded,
                "hold_days": int(hold_days),
                "entry_regime": open_trade_regime_label,
                "entry_regime_label": open_trade_regime_label,
                "entry_ratio_z": open_trade_ratio_z,
                "exit_ratio_z": exit_ratio_z,
                "entry_bar_index": open_trade_entry_idx,
                "exit_bar_index": int(close_bar_idx),
                "entry_dt": open_trade_entry_dt,
                "exit_dt": close_bar_date,
                "entry_price": open_trade_entry_price,
                "exit_price": exit_price,
                "shares": int(open_trade_total_entry_shares),
                "exit_reason": exit_reason or "EXIT",
            })
            trade_realized_pnl = 0.0
            open_trade_conf = None
            open_trade_entry_idx = None
            open_trade_entry_dt = None
            open_trade_entry_price = None
            open_trade_total_entry_shares = 0
            open_trade_regime_label = None
            open_trade_ratio_z = None

        # Cash yield on uninvested capital (daily compounding, credited end-of-bar)
        daily_cash_rate = _daily_cash_yield_rate(cfg.cash_yield_annual_pct)

        atr_series = pd.Series(calculate_atr(df, 14), index=df.index, dtype=float)
        close_series = pd.Series(df["close"], index=df.index, dtype=float)
        sma200 = pd.Series(close_series.rolling(200).mean(), index=df.index, dtype=float)
        sma200_slope = pd.Series(sma200.diff(1), index=df.index, dtype=float)  # P0.2: single-bar slope

        # P0.2: auto-calibrate min slope threshold
        min_slope_thr = cfg.thesis_break_min_slope
        if min_slope_thr == 0.0 and len(sma200_slope.dropna()) > 50:
            abs_slopes = sma200_slope.dropna().abs()
            min_slope_thr = float(abs_slopes.quantile(0.25))

        # P2.1: precompute median ATR% for vol-adjusted sizing
        atr_pct_series = pd.Series((atr_series / close_series) * 100.0, index=df.index, dtype=float)
        atr_pct_median_60 = atr_pct_series.rolling(60).median()
        ratio_series = pd.Series(compute_ratio_series(df, cfg), index=df.index, dtype=float)
        ratio_mu_series = ratio_series.rolling(int(cfg.ratio_lookback)).mean()
        # Feature A: realized volatility targeting (optional, disabled by default)
        log_ret = pd.Series(
            np.log(close_series / close_series.shift(1)),
            index=df.index,
            dtype=float,
        )
        sigma_daily = log_ret.rolling(int(cfg.tactical_vol_window)).std()
        sigma_annual = np.sqrt(252.0) * sigma_daily

        # Spec v1: realized volatility using 252-bar window (annual lookback)
        sigma_annual_252: pd.Series = pd.Series(np.nan, index=df.index, dtype=float)
        if cfg.enable_spec_v1_upgrades:
            # sigma_annual = np.log(df['close']).diff().rolling(window=252).std() * np.sqrt(252)
            log_close = pd.Series(np.log(close_series.to_numpy(dtype=float)), index=df.index, dtype=float)
            sigma_annual_252 = log_close.diff().rolling(window=252).std() * np.sqrt(252)

        # Spec v1: pre-compute half_life for ERet calculation
        half_life_val: Optional[float] = None
        if cfg.enable_spec_v1_upgrades:
            half_life_val = calculate_half_life(df["close"])

        def _apply_cash_yield_eod() -> None:
            nonlocal cash, accrued_cash_yield
            if daily_cash_rate > 0 and cash > 0:
                interest = cash * daily_cash_rate
                cash += interest
                accrued_cash_yield += interest

        def _sell_next_open_if_available(
            i: int, reason: str, action_reason: str, regime: str, rz: float,
        ) -> bool:
            nonlocal cash
            if i + 1 >= n or not pos.is_open:
                return False
            exit_date = df.index[i + 1].strftime("%Y-%m-%d")
            exit_px = self._fill_price(float(df.iloc[i + 1]["open"]), sell=True)
            cash, fill = self._sell_all(pos, i + 1, exit_date, exit_px, reason, cash)
            fills.append(fill)
            actions_log.append(
                {
                    "bar": i,
                    "date": bar_date,
                    "action": "SELL",
                    "reason": action_reason,
                    "regime": regime,
                }
            )
            _record_trade_close(
                i + 1, float(fill.get("realized_pnl", 0.0)),
                close_bar_date=str(fill.get("date", exit_date)),
                exit_price=safe_number(fill.get("price"), None),
                exit_reason=str(fill.get("reason", reason)),
                exit_ratio_z=(None if pd.isna(rz) else float(rz)),
            )
            _apply_cash_yield_eod()
            equity_curve.append(round(cash, 2))
            return True

        for i in range(n):
            bar = df.iloc[i]
            bar_date = df.index[i].strftime("%Y-%m-%d")
            price = float(bar["close"])
            signal_i = i
            if cfg.entry_at == "same_close" and cfg.lag_signals_for_same_close:
                signal_i = i - 1
            rz = (
                ratio_z_series.iloc[signal_i]
                if signal_i >= 0 and signal_i < len(ratio_z_series)
                else np.nan
            )
            regime = (
                regime_labels.iloc[signal_i]
                if signal_i >= 0 and signal_i < len(regime_labels)
                else Regime.AMBIGUOUS.value
            )
            if str(regime) == "NO_EVIDENCE":
                regime_no_evidence_bars += 1

            # --- Exposure tracking ---
            mark = cash + pos.shares * price
            exp = (pos.shares * price / mark * 100) if mark > 0 and pos.is_open else 0.0
            exposure_pct_curve.append(round(exp, 2))

            # --- Risk exits ---
            if pos.is_open:
                bars_held = i - pos.entry_bar

                # Spec v1: Mean-Reversion Exit (downside worsening trigger)
                if cfg.enable_spec_v1_upgrades and not pd.isna(rz):
                    should_exit_mr = (
                        float(rz) <= float(cfg.tactical_exit_z)
                        and bars_held >= int(cfg.tactical_min_hold_days)
                    )
                    if should_exit_mr:
                        exit_px = self._fill_price(price, sell=True)
                        cash, fill = self._sell_all(
                            pos, i, bar_date, exit_px, "REVERTED_TO_MEAN", cash,
                        )
                        fills.append(fill)
                        actions_log.append({
                            "bar": i, "date": bar_date, "action": "SELL",
                            "reason": "REVERTED_TO_MEAN", "regime": regime,
                        })
                        _record_trade_close(
                            i, float(fill.get("realized_pnl", 0.0)),
                            close_bar_date=str(fill.get("date", bar_date)),
                            exit_price=safe_number(fill.get("price"), None),
                            exit_reason="REVERTED_TO_MEAN",
                            exit_ratio_z=float(rz),
                        )
                        _apply_cash_yield_eod()
                        equity_curve.append(round(cash, 2))
                        continue

                # Stop-loss (intra-bar)
                if bar["low"] <= pos.stop_price:
                    is_gap_stop = float(bar["open"]) < float(pos.stop_price)
                    exec_base = _stop_execution_base(
                        open_px=float(bar["open"]),
                        high_px=float(bar["high"]),
                        low_px=float(bar["low"]),
                        stop_px=float(pos.stop_price),
                    )
                    exit_px = self._fill_price(exec_base, sell=True)
                    sell_reason = "SELL_GAP_STOP_OPEN" if is_gap_stop else "SELL_STOP"
                    action_reason = "GAP_STOP_OPEN" if is_gap_stop else "STOP_LOSS"
                    cash, fill = self._sell_all(pos, i, bar_date, exit_px, sell_reason, cash)
                    fills.append(fill)
                    actions_log.append({"bar": i, "date": bar_date, "action": "SELL",
                                        "reason": action_reason, "regime": regime})
                    _record_trade_close(
                        i, float(fill.get("realized_pnl", 0.0)),
                        close_bar_date=str(fill.get("date", bar_date)),
                        exit_price=safe_number(fill.get("price"), None),
                        exit_reason=str(fill.get("reason", sell_reason)),
                        exit_ratio_z=(None if pd.isna(rz) else float(rz)),
                    )
                    _apply_cash_yield_eod()
                    equity_curve.append(round(cash, 2))
                    continue

                # P0.2: Thesis break — consecutive negative slope + magnitude + guard
                if i < len(sma200_slope) and not pd.isna(sma200_slope.iloc[i]):
                    slope_val = float(sma200_slope.iloc[i])
                    if slope_val < -min_slope_thr:
                        pos.neg_slope_count += 1
                    else:
                        pos.neg_slope_count = 0

                    if pos.neg_slope_count >= cfg.thesis_break_sma_bars:
                        # Guard: only fire if price < SMA200 OR regime is trending
                        sma_val = sma200.iloc[i] if i < len(sma200) else np.nan
                        price_below_sma = (not pd.isna(sma_val) and price < sma_val)
                        regime_trending = (regime == Regime.TRENDING.value)

                        if not cfg.thesis_break_require_below_sma or price_below_sma or regime_trending:
                            exit_px = self._fill_price(price, sell=True)
                            cash, fill = self._sell_all(pos, i, bar_date, exit_px,
                                                        "SELL_THESIS_BREAK", cash)
                            fills.append(fill)
                            actions_log.append({"bar": i, "date": bar_date, "action": "SELL",
                                                "reason": "THESIS_BREAK", "regime": regime})
                            _record_trade_close(
                                i, float(fill.get("realized_pnl", 0.0)),
                                close_bar_date=str(fill.get("date", bar_date)),
                                exit_price=safe_number(fill.get("price"), None),
                                exit_reason=str(fill.get("reason", "SELL_THESIS_BREAK")),
                                exit_ratio_z=(None if pd.isna(rz) else float(rz)),
                            )
                            _apply_cash_yield_eod()
                            equity_curve.append(round(cash, 2))
                            continue

                if cfg.better_exits_enabled:
                    # Feature C: return-to-mean exit, gated by minimum hold; executes next open.
                    if (
                        bars_held >= int(cfg.tactical_min_hold_days)
                        and not pd.isna(rz)
                        and float(rz) >= float(cfg.tactical_exit_z)
                    ):
                        if _sell_next_open_if_available(
                            i,
                            reason="SELL_RETURN_TO_MEAN",
                            action_reason="RETURN_TO_MEAN_NEXT_OPEN",
                            regime=regime,
                            rz=float(rz),
                        ):
                            continue
                    # Feature C: hard time stop; executes next open.
                    if bars_held >= int(cfg.tactical_max_hold_days):
                        if _sell_next_open_if_available(
                            i,
                            reason="SELL_TIME_LIMIT",
                            action_reason="TIME_LIMIT_NEXT_OPEN",
                            regime=regime,
                            rz=float(rz) if not pd.isna(rz) else np.nan,
                        ):
                            continue
                else:
                    # Legacy time stop behavior (unchanged)
                    max_hold = cfg.max_holding_days
                    if regime == Regime.AMBIGUOUS.value:
                        max_hold = int(max_hold * 0.6)
                    if bars_held >= max_hold:
                        exit_px = self._fill_price(price, sell=True)
                        cash, fill = self._sell_all(pos, i, bar_date, exit_px,
                                                    "SELL_TIME_LIMIT", cash)
                        fills.append(fill)
                        actions_log.append({"bar": i, "date": bar_date, "action": "SELL",
                                            "reason": "TIME_LIMIT", "regime": regime})
                        _record_trade_close(
                            i, float(fill.get("realized_pnl", 0.0)),
                            close_bar_date=str(fill.get("date", bar_date)),
                            exit_price=safe_number(fill.get("price"), None),
                            exit_reason=str(fill.get("reason", "SELL_TIME_LIMIT")),
                            exit_ratio_z=(None if pd.isna(rz) else float(rz)),
                        )
                        _apply_cash_yield_eod()
                        equity_curve.append(round(cash, 2))
                        continue

            # --- Action engine ---
            if pd.isna(rz):
                actions_log.append({"bar": i, "date": bar_date, "action": "HOLD",
                                    "reason": "NO_DATA", "regime": regime})
                _apply_cash_yield_eod()
                equity_curve.append(round(cash + pos.shares * price, 2))
                continue

            equity = cash + pos.shares * price
            position_pct = (pos.shares * price / equity) if equity > 0 else 0.0

            reversal_ok = check_reversal_confirmation(df, i, cfg)
            quality_ok = passes_quality_filter(df, i, cfg)

            action = determine_action(
                rz, regime, pos.is_open, position_pct, cfg, reversal_ok, quality_ok,
            )

            # P2.1: vol-adjusted sizing multiplier
            atr_pct_now = (float(atr_pct_series.iloc[i])
                           if i < len(atr_pct_series) and not pd.isna(atr_pct_series.iloc[i])
                           else 2.0)
            atr_pct_med = (float(atr_pct_median_60.iloc[i])
                           if i < len(atr_pct_median_60) and not pd.isna(atr_pct_median_60.iloc[i])
                           else atr_pct_now)
            if cfg.enable_spec_v1_upgrades:
                # Deprecated: Spec v1 uses log-ret vol (sigma_annual_252)
                vol_mult = 1.0
            elif cfg.vol_adjust_sizing and atr_pct_now > 0:
                vol_mult = atr_pct_med / atr_pct_now
                vol_mult = max(cfg.vol_sizing_floor, min(cfg.vol_sizing_cap, vol_mult))
            else:
                vol_mult = 1.0

            mvol = 1.0
            if cfg.enable_spec_v1_upgrades:
                # Spec v1: m_vol = np.clip(target_vol / np.maximum(sigma_annual, vol_floor), 0, vol_cap)
                sig_a_252 = (
                    float(sigma_annual_252.iloc[i])
                    if i < len(sigma_annual_252) and not pd.isna(sigma_annual_252.iloc[i])
                    else None
                )
                mvol = _vol_target_multiplier(
                    sig_a_252,
                    sigma_target=float(cfg.tactical_vol_target),
                    sigma_floor=float(cfg.tactical_vol_floor),
                    cap=float(cfg.tactical_vol_cap),
                )
            elif cfg.tactical_vol_targeting_enabled:
                sig_a_now = (
                    float(sigma_annual.iloc[i])
                    if i < len(sigma_annual) and not pd.isna(sigma_annual.iloc[i])
                    else None
                )
                mvol = _vol_target_multiplier(
                    sig_a_now,
                    sigma_target=float(cfg.tactical_vol_target),
                    sigma_floor=float(cfg.tactical_vol_floor),
                    cap=float(cfg.tactical_vol_cap),
                )
            vol_mult *= mvol

            vol_hot = atr_pct_now > atr_pct_med * 1.5

            # --- Execute action ---
            if action == Action.BUY:
                if _should_block_next_open_entry(i, n, cfg.entry_at):
                    actions_log.append({"bar": i, "date": bar_date, "action": "HOLD",
                                        "reason": "LAST_BAR_NEXT_OPEN_GUARD", "regime": regime})
                    _apply_cash_yield_eod()
                    equity_curve.append(round(cash + pos.shares * price, 2))
                    continue
                was_closed = not pos.is_open
                entry_conf = _entry_confidence(signal_i, regime_labels, N=60)
                if was_closed and cfg.min_confidence > 0.0:
                    if entry_conf is not None and entry_conf < cfg.min_confidence:
                        blocked_by_confidence += 1
                        blocked_count += 1
                        actions_log.append({
                            "bar": i, "date": bar_date, "action": "BLOCKED",
                            "reason": (
                                f"confidence={entry_conf:.2f}<min={cfg.min_confidence:.2f}"
                            ),
                            "regime": regime,
                        })
                        _apply_cash_yield_eod()
                        equity_curve.append(round(cash + pos.shares * price, 2))
                        continue
                # --- Cost-Aware Entry Gate ---
                if was_closed and cfg.enable_spec_v1_upgrades:
                    # Spec v1: ERet = dR * (ratio_mu - ratio_now) / ratio_now
                    r_now = (
                        ratio_series.iloc[signal_i]
                        if signal_i >= 0 and signal_i < len(ratio_series)
                        else np.nan
                    )
                    r_mu = (
                        ratio_mu_series.iloc[signal_i]
                        if signal_i >= 0 and signal_i < len(ratio_mu_series)
                        else np.nan
                    )
                    eret = _expected_return_spec_v1(
                        safe_number(r_now, None),
                        safe_number(r_mu, None),
                        half_life_val,
                    )
                    # Spec v1: is_cost_effective = e_ret >= cost_k * (slippage_bps/10000 + commission/notional)
                    est_notional = max(float(equity * cfg.max_position_pct * 0.6), 1.0)
                    is_cost_effective = _is_cost_effective_spec_v1(
                        eret,
                        cost_k=float(cfg.cost_k),
                        slippage_bps=float(cfg.cost_bps_est),
                        commission=float(cfg.commission_per_trade),
                        notional=est_notional,
                    )
                    if not is_cost_effective:
                        blocked_count += 1
                        blocked_by_cost += 1
                        actions_log.append({
                            "bar": i, "date": bar_date, "action": "BLOCKED",
                            "reason": (
                                f"COST_AWARE_FILTER eret={eret:.4f}"
                            ),
                            "regime": regime,
                        })
                        _apply_cash_yield_eod()
                        equity_curve.append(round(cash + pos.shares * price, 2))
                        continue
                elif was_closed and cfg.cost_aware_entry_enabled:
                    r_now = (
                        ratio_series.iloc[signal_i]
                        if signal_i >= 0 and signal_i < len(ratio_series)
                        else np.nan
                    )
                    r_mu = (
                        ratio_mu_series.iloc[signal_i]
                        if signal_i >= 0 and signal_i < len(ratio_mu_series)
                        else np.nan
                    )
                    eret = _expected_return_proxy_from_ratio(
                        safe_number(r_now, None),
                        safe_number(r_mu, None),
                    )
                    cost_frac = float(cfg.cost_bps_est) / 10000.0
                    min_req = float(cfg.cost_k) * cost_frac
                    if eret < min_req:
                        blocked_count += 1
                        blocked_by_cost += 1
                        actions_log.append({
                            "bar": i, "date": bar_date, "action": "BLOCKED",
                            "reason": (
                                f"cost_aware eret={eret:.4f}<min={min_req:.4f}"
                            ),
                            "regime": regime,
                        })
                        _apply_cash_yield_eod()
                        equity_curve.append(round(cash + pos.shares * price, 2))
                        continue
                fill_px = self._get_entry_price(df, i, cfg)
                if fill_px is None:
                    actions_log.append({"bar": i, "date": bar_date, "action": "HOLD",
                                        "reason": "NO_FILL_BAR", "regime": regime})
                    _apply_cash_yield_eod()
                    equity_curve.append(round(cash + pos.shares * price, 2))
                    continue

                if cfg.enable_spec_v1_upgrades:
                    # Spec v1: final_size = base_size * m_vol * m_conf (linear ramp, no gamma)
                    base_size = equity * cfg.max_position_pct * 0.6
                    m_conf_v1 = _confidence_sizing_multiplier(
                        entry_conf,
                        c0=float(cfg.confidence_c0),
                        gamma=1.0,  # Spec v1: no power/gamma
                    )
                    max_val = base_size * mvol * m_conf_v1
                else:
                    # Dynamic sizing: 60% of max for initial BUY, regime + vol adjusted
                    size_pct = cfg.max_position_pct * 0.6
                    if regime == Regime.AMBIGUOUS.value:
                        size_pct *= 0.5
                    if vol_hot:
                        size_pct *= 0.7
                    max_val = equity * size_pct * vol_mult
                    if cfg.confidence_sizing_enabled:
                        m_conf = _confidence_sizing_multiplier(
                            entry_conf,
                            c0=float(cfg.confidence_c0),
                            gamma=float(cfg.confidence_gamma),
                        )
                        max_val *= m_conf
                shares = int(max_val // fill_px)
                min_cash = equity * cfg.min_cash_pct
                if shares * fill_px + cfg.commission_per_trade > cash - min_cash:
                    shares = int((cash - min_cash - cfg.commission_per_trade) // fill_px)

                # P0.4: dust filter
                if shares < cfg.min_shares or shares * fill_px < cfg.min_trade_notional:
                    actions_log.append({"bar": i, "date": bar_date, "action": "HOLD",
                                        "reason": "DUST_FILTER", "regime": regime})
                    _apply_cash_yield_eod()
                    equity_curve.append(round(cash + pos.shares * price, 2))
                    continue

                cost = shares * fill_px + cfg.commission_per_trade
                cash -= cost
                pos.shares += shares
                pos.cost_basis += shares * fill_px
                pos.entry_bar = i
                if was_closed:
                    entry_fill_date = (
                        df.index[i + 1].strftime("%Y-%m-%d")
                        if cfg.entry_at == "next_open" and i + 1 < n
                        else bar_date
                    )
                    open_trade_conf = entry_conf
                    open_trade_entry_idx = i
                    open_trade_entry_dt = entry_fill_date
                    open_trade_entry_price = float(fill_px)
                    open_trade_total_entry_shares = int(shares)
                    open_trade_regime_label = str(regime)
                    open_trade_ratio_z = float(rz) if not pd.isna(rz) else None
                    trade_realized_pnl = 0.0
                pos.last_add_bar = i
                pos.neg_slope_count = 0

                atr_val = (float(atr_series.iloc[i])
                           if i < len(atr_series) and not pd.isna(atr_series.iloc[i])
                           else fill_px * 0.02)
                pos.stop_price = round(fill_px - atr_val * cfg.stop_atr_multiple, 2)

                buy_fill: Dict[str, Any] = {
                    "bar": i, "date": bar_date, "action": "BUY",
                    "price": fill_px, "shares": shares,
                    "cost": round(cost, 2), "regime": regime,
                }
                if cfg.enable_spec_v1_upgrades:
                    # Spec v1 diagnostics: m_vol/m_conf in fill record
                    buy_fill["m_vol"] = round(float(mvol), 4)
                    buy_fill["m_conf"] = round(float(m_conf_v1), 4)
                fills.append(buy_fill)
                actions_log.append({"bar": i, "date": bar_date, "action": "BUY",
                                    "reason": f"ratio_z={rz:.2f}", "regime": regime})

            elif action == Action.ADD:
                if _should_block_next_open_entry(i, n, cfg.entry_at):
                    actions_log.append({"bar": i, "date": bar_date, "action": "HOLD",
                                        "reason": "LAST_BAR_NEXT_OPEN_GUARD", "regime": regime})
                    _apply_cash_yield_eod()
                    equity_curve.append(round(cash + pos.shares * price, 2))
                    continue
                fill_px = self._get_entry_price(df, i, cfg)
                if fill_px is None:
                    _apply_cash_yield_eod()
                    equity_curve.append(round(cash + pos.shares * price, 2))
                    continue

                # P0.4: target notional = equity * add_step_pct
                add_pct = cfg.add_step_pct
                if regime == Regime.AMBIGUOUS.value:
                    add_pct *= 0.5
                if vol_hot:
                    add_pct *= 0.7
                add_pct *= vol_mult

                current_pos_val = pos.shares * fill_px
                max_add_val = equity * cfg.max_position_pct - current_pos_val
                target_val = min(equity * add_pct, max_add_val)
                shares = int(target_val // fill_px)
                min_cash = equity * cfg.min_cash_pct
                if shares * fill_px + cfg.commission_per_trade > cash - min_cash:
                    shares = int((cash - min_cash - cfg.commission_per_trade) // fill_px)

                # P0.4: dust filter
                if shares < cfg.min_shares or shares * fill_px < cfg.min_trade_notional:
                    _apply_cash_yield_eod()
                    equity_curve.append(round(cash + pos.shares * price, 2))
                    actions_log.append({"bar": i, "date": bar_date, "action": "HOLD",
                                        "reason": "ADD_DUST", "regime": regime})
                    continue

                cost = shares * fill_px + cfg.commission_per_trade
                cash -= cost
                pos.shares += shares
                pos.cost_basis += shares * fill_px
                pos.last_add_bar = i
                if open_trade_entry_idx is not None:
                    open_trade_total_entry_shares += int(shares)

                atr_val = (float(atr_series.iloc[i])
                           if i < len(atr_series) and not pd.isna(atr_series.iloc[i])
                           else fill_px * 0.02)
                new_stop = round(fill_px - atr_val * cfg.stop_atr_multiple, 2)
                pos.stop_price = min(pos.stop_price, new_stop)

                fills.append({
                    "bar": i, "date": bar_date, "action": "ADD",
                    "price": fill_px, "shares": shares,
                    "cost": round(cost, 2), "regime": regime,
                })
                actions_log.append({"bar": i, "date": bar_date, "action": "ADD",
                                    "reason": f"ratio_z={rz:.2f}", "regime": regime})

            elif action == Action.REDUCE:
                if pos.is_open:
                    # P2.2: staged trim fraction
                    trim_result = _get_trim_action(rz, cfg)
                    frac = trim_result[1] if trim_result else cfg.trim_step_pct
                    if frac >= 1.0:
                        # Full sell handled below
                        pass
                    trim_shares = max(1, int(pos.shares * frac))
                    trim_shares = min(trim_shares, pos.shares)

                    # P0.4: dust filter for trims
                    if trim_shares < cfg.min_shares:
                        if trim_shares < pos.shares:
                            # Skip dust trim, let it accumulate
                            _apply_cash_yield_eod()
                            equity_curve.append(round(cash + pos.shares * price, 2))
                            actions_log.append({"bar": i, "date": bar_date, "action": "HOLD",
                                                "reason": "TRIM_DUST", "regime": regime})
                            continue
                        # If trim_shares == pos.shares (selling everything), allow it
                    sell_px = self._fill_price(price, sell=True)
                    proceeds = trim_shares * sell_px - cfg.commission_per_trade
                    cash += proceeds

                    realized = trim_shares * (sell_px - pos.avg_cost)
                    pos.cost_basis -= trim_shares * pos.avg_cost
                    pos.shares -= trim_shares

                    fills.append({
                        "bar": i, "date": bar_date, "action": "REDUCE",
                        "price": sell_px, "shares": trim_shares,
                        "proceeds": round(proceeds, 2),
                        "realized_pnl": round(realized, 2), "regime": regime,
                    })
                    actions_log.append({"bar": i, "date": bar_date, "action": "REDUCE",
                                        "reason": f"ratio_z={rz:.2f}", "regime": regime})
                    reduce_realized_pnl = float(fills[-1].get("realized_pnl", 0.0))
                    if pos.shares == 0:
                        _record_trade_close(
                            i, reduce_realized_pnl,
                            close_bar_date=bar_date,
                            exit_price=float(sell_px),
                            exit_reason="REDUCE",
                            exit_ratio_z=(None if pd.isna(rz) else float(rz)),
                        )
                    else:
                        trade_realized_pnl += reduce_realized_pnl

            elif action == Action.SELL:
                if pos.is_open:
                    sell_px = self._fill_price(price, sell=True)
                    cash, fill = self._sell_all(pos, i, bar_date, sell_px,
                                                "SELL_OVERVALUED", cash)
                    fills.append(fill)
                    actions_log.append({"bar": i, "date": bar_date, "action": "SELL",
                                        "reason": f"ratio_z={rz:.2f}", "regime": regime})
                    _record_trade_close(
                        i, float(fill.get("realized_pnl", 0.0)),
                        close_bar_date=str(fill.get("date", bar_date)),
                        exit_price=safe_number(fill.get("price"), None),
                        exit_reason=str(fill.get("reason", "SELL_OVERVALUED")),
                        exit_ratio_z=(None if pd.isna(rz) else float(rz)),
                    )

            elif action == Action.BLOCKED:
                blocked_count += 1
                blocked_by_regime += 1
                actions_log.append({"bar": i, "date": bar_date, "action": "BLOCKED",
                                    "reason": f"regime={regime}", "regime": regime})

            else:  # HOLD
                actions_log.append({"bar": i, "date": bar_date, "action": "HOLD",
                                    "reason": "", "regime": regime})

            _apply_cash_yield_eod()
            mark = cash + pos.shares * price
            equity_curve.append(round(mark, 2))

        # Force close at end
        if pos.is_open:
            last_px = self._fill_price(float(df["close"].iloc[-1]), sell=True)
            cash, fill = self._sell_all(pos, n - 1,
                                        df.index[-1].strftime("%Y-%m-%d"),
                                        last_px, "SELL_END_OF_DATA", cash)
            fills.append(fill)
            final_rz = (
                ratio_z_series.iloc[n - 1]
                if n - 1 < len(ratio_z_series) else np.nan
            )
            _record_trade_close(
                n - 1, float(fill.get("realized_pnl", 0.0)),
                close_bar_date=str(fill.get("date", df.index[-1].strftime("%Y-%m-%d"))),
                exit_price=safe_number(fill.get("price"), None),
                exit_reason=str(fill.get("reason", "SELL_END_OF_DATA")),
                exit_ratio_z=(None if pd.isna(final_rz) else float(final_rz)),
            )
            equity_curve[-1] = round(cash, 2)

        bucket_summary = _summarize_confidence_buckets(trade_records)
        if bucket_summary["total_trades"] > 0:
            print("CONFIDENCE BUCKETS (REGIME PROXY - MR/SW SHARE AT ENTRY)")
            print("Bucket Trades WinRate AvgPnL AvgHoldDays")
            for row in bucket_summary.get("rows", []):
                sample_note = " (LOW SAMPLE)" if row.get("low_sample") else ""
                print(
                    f"{row['bucket']:<10} {row['trades']:>6} {row['win_rate_pct']:>7.1f}% "
                    f"{row['avg_pnl']:>8.2f} {row['avg_hold_days']:>11.1f}{sample_note}"
                )

        return {
            "equity_curve": equity_curve,
            "fills": fills,
            "actions_log": actions_log,
            "blocked_count": blocked_count,
            "blocked_by_confidence": blocked_by_confidence,
            "blocked_by_regime": blocked_by_regime,
            "blocked_by_cost": blocked_by_cost,
            "regime_no_evidence_bars": regime_no_evidence_bars,
            "exposure_pct_curve": exposure_pct_curve,
            "accrued_cash_yield": round(accrued_cash_yield, 2),
            "confidence_bucket_summary": bucket_summary,
            "trade_records": trade_records,
        }

    # --- helpers ---

    def _fill_price(self, base: float, sell: bool = False) -> float:
        slip = self.cfg.slippage_pct
        return round(base * (1 - slip) if sell else base * (1 + slip), 4)

    def _get_entry_price(
        self, df: pd.DataFrame, i: int, cfg: MeanReversionConfig,
    ) -> Optional[float]:
        if cfg.entry_at == "next_open" and i + 1 < len(df):
            return self._fill_price(float(df.iloc[i + 1]["open"]), sell=False)
        return self._fill_price(float(df.iloc[i]["close"]), sell=False)

    def _sell_all(
        self, pos: _PositionState, bar_idx: int, bar_date: str,
        exit_px: float, reason: str, cash: float,
    ) -> Tuple[float, Dict]:
        proceeds = pos.shares * exit_px - self.cfg.commission_per_trade
        realized = pos.shares * (exit_px - pos.avg_cost)
        cash += proceeds
        fill = {
            "bar": bar_idx, "date": bar_date, "action": "SELL",
            "price": exit_px, "shares": pos.shares,
            "proceeds": round(proceeds, 2),
            "realized_pnl": round(realized, 2),
            "reason": reason, "days_held": bar_idx - pos.entry_bar,
        }
        pos.shares = 0
        pos.cost_basis = 0.0
        pos.entry_bar = -1
        pos.stop_price = 0.0
        pos.neg_slope_count = 0
        return cash, fill


# =============================================================================
# PERFORMANCE METRICS
# =============================================================================

SHARPE_STD_EPS = 1e-10


def _daily_cash_proxy_return(cash_yield_annual_pct: float) -> float:
    """Convert annualized cash yield (%) to daily compounded cash return."""
    return _daily_cash_yield_rate(cash_yield_annual_pct)


def _cash_proxy_compounded_return_pct(
    cash_yield_annual_pct: float,
    n_bars: int,
) -> float:
    """Compounded cash-benchmark return (%) over n_bars daily periods."""
    bars = max(int(n_bars), 0)
    if bars == 0:
        return 0.0
    daily = _daily_cash_proxy_return(cash_yield_annual_pct)
    return ((1 + daily) ** bars - 1) * 100


def _excess_return_sharpe(
    equity_curve: List[float],
    cash_yield_annual_pct: float = 0.0,
    std_epsilon: float = SHARPE_STD_EPS,
) -> Optional[float]:
    """Annualized Sharpe on daily excess returns; None if degenerate."""
    eq = pd.Series(equity_curve, dtype=float)
    daily_ret = eq.pct_change().dropna()
    if len(daily_ret) < 2:
        return None

    rf_daily = _daily_cash_proxy_return(cash_yield_annual_pct)
    excess_ret = daily_ret - rf_daily
    ex_std = float(excess_ret.std())
    if not np.isfinite(ex_std) or ex_std < std_epsilon:
        return None

    sharpe = float((excess_ret.mean() / ex_std) * np.sqrt(252))
    if not np.isfinite(sharpe):
        return None
    return sharpe


def _format_sharpe(value: Any, width: Optional[int] = None) -> str:
    """Format sharpe-like metric for reports while allowing 'N/A' strings."""
    if isinstance(value, (int, float, np.floating)):
        text = f"{float(value):.3f}"
    else:
        text = str(value)
    return f"{text:>{width}}" if width is not None else text


def calculate_performance_metrics(
    equity_curve: List[float], fills: List[Dict],
    cash_yield_annual_pct: float = 0.0,
) -> Dict:
    eq = pd.Series(equity_curve, dtype=float)
    daily_ret = eq.pct_change().dropna()
    sharpe_raw = _excess_return_sharpe(
        equity_curve, cash_yield_annual_pct=cash_yield_annual_pct,
    )
    sharpe = round(sharpe_raw, 3) if sharpe_raw is not None else "N/A"
    cum = (1 + daily_ret).cumprod()
    running_max = cum.expanding().max()
    dd = (cum - running_max) / running_max
    max_dd = float(dd.min()) if len(dd) > 0 else 0.0
    max_dd_dur = 0
    if (dd < 0).any():
        groups = (~(dd < 0)).cumsum()
        dd_lens = (dd < 0).groupby(groups).sum()
        max_dd_dur = int(dd_lens.max()) if len(dd_lens) > 0 else 0

    sells = [f for f in fills if f.get("action") in ("SELL", "REDUCE")]
    winners = [f for f in sells if f.get("realized_pnl", 0) > 0]
    losers = [f for f in sells if f.get("realized_pnl", 0) <= 0]
    gross_profit = sum(f["realized_pnl"] for f in winners)
    gross_loss = abs(sum(f["realized_pnl"] for f in losers))
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    start_eq = equity_curve[0] if equity_curve else 1
    end_eq = equity_curve[-1] if equity_curve else 1
    total_ret = (end_eq / start_eq - 1) * 100
    n_bars = max(len(equity_curve), 1)
    ann_ret = ((end_eq / start_eq) ** (252 / n_bars) - 1) * 100 if n_bars > 0 else 0

    buys = len([f for f in fills if f["action"] == "BUY"])
    adds = len([f for f in fills if f["action"] == "ADD"])
    reduces = len([f for f in fills if f["action"] == "REDUCE"])
    sell_count = len([f for f in fills if f["action"] == "SELL"])

    return {
        "starting_capital": round(start_eq, 2),
        "ending_capital": round(end_eq, 2),
        "total_return_pct": round(total_ret, 2),
        "annualized_return_pct": round(ann_ret, 2),
        "sharpe_ratio": sharpe,
        "max_drawdown_pct": round(max_dd * 100, 2),
        "max_drawdown_duration_bars": max_dd_dur,
        "total_realized_pnl": round(gross_profit - gross_loss, 2),
        "profit_factor": round(pf, 2) if pf != float("inf") else "inf",
        "gross_profit": round(gross_profit, 2),
        "gross_loss": round(gross_loss, 2),
        "fills_buy": buys, "fills_add": adds,
        "fills_reduce": reduces, "fills_sell": sell_count,
        "total_fills": buys + adds + reduces + sell_count,
        "buy_and_hold_return_pct": None,
    }


# =============================================================================
# DIAGNOSTICS
# =============================================================================

def _metrics_by_regime(actions_log: List[Dict], fills: List[Dict]) -> Dict:
    """Return {regime: {fills, realized_pnl, wins, losses}}."""
    # Map fill bars to regime
    fill_regimes = {}
    for f in fills:
        r = f.get("regime", "UNKNOWN")
        fill_regimes.setdefault(r, []).append(f)

    result = {}
    for r, flist in fill_regimes.items():
        exit_fills = [f for f in flist if f["action"] in ("SELL", "REDUCE")]
        pnl = sum(f.get("realized_pnl", 0) for f in exit_fills)
        wins = sum(1 for f in exit_fills if f.get("realized_pnl", 0) > 0)
        losses = sum(1 for f in exit_fills if f.get("realized_pnl", 0) <= 0)
        result[r] = {"fills": len(flist), "realized_pnl": round(pnl, 2),
                      "wins": wins, "losses": losses}
    return result


def _exit_reason_breakdown(fills: List[Dict]) -> Dict:
    """Return {reason: {count, avg_days, avg_pnl, wins, losses}}."""
    exits = [f for f in fills if f["action"] in ("SELL", "REDUCE")]
    reasons: Dict[str, List[Dict]] = {}
    for f in exits:
        r = f.get("reason", "UNKNOWN")
        reasons.setdefault(r, []).append(f)
    result = {}
    for r, flist in reasons.items():
        pnls = [f.get("realized_pnl", 0) for f in flist]
        days = [f.get("days_held", 0) for f in flist]
        result[r] = {
            "count": len(flist),
            "avg_pnl": round(np.mean(pnls), 2) if pnls else 0,
            "avg_days": round(np.mean(days), 1) if days else 0,
            "wins": sum(1 for p in pnls if p > 0),
            "losses": sum(1 for p in pnls if p <= 0),
        }
    return result


def _regime_transition_matrix(actions_log: List[Dict]) -> Dict:
    """Count transitions between regimes."""
    trans: Dict[str, Dict[str, int]] = {}
    prev = None
    for a in actions_log:
        cur = a.get("regime", "UNKNOWN")
        if prev is not None and prev != cur:
            trans.setdefault(prev, {})
            trans[prev][cur] = trans[prev].get(cur, 0) + 1
        prev = cur
    return trans


def _tactical_diagnostics(bt_results: Dict) -> Dict:
    """Compute tactical-layer specific diagnostics."""
    actions = bt_results.get("actions_log", [])
    fills = bt_results.get("fills", [])
    exposure = bt_results.get("exposure_pct_curve", [])
    total_bars = max(len(actions), 1)

    # Time in market (% of bars with open position)
    invested_bars = 0
    in_pos = False
    for a in actions:
        if a["action"] == "BUY":
            in_pos = True
        elif a["action"] == "SELL":
            in_pos = False
        if in_pos:
            invested_bars += 1
    time_in_market = round(invested_bars / total_bars * 100, 1)

    # Average exposure
    avg_exposure = round(float(np.mean(exposure)), 2) if exposure else 0.0

    # Blocked signal rate
    buy_signals = sum(1 for a in actions if a["action"] in ("BUY", "BLOCKED"))
    blocked = bt_results.get("blocked_count", 0)
    blocked_rate = round(blocked / max(buy_signals, 1) * 100, 1)
    blocked_by_confidence = int(bt_results.get("blocked_by_confidence", 0))
    blocked_by_regime = int(bt_results.get("blocked_by_regime", 0))
    blocked_by_cost = int(bt_results.get("blocked_by_cost", 0))
    regime_no_evidence_bars = int(bt_results.get("regime_no_evidence_bars", 0))

    return {
        "time_in_market_pct": time_in_market,
        "avg_exposure_pct": avg_exposure,
        "blocked_signal_rate_pct": blocked_rate,
        "blocked_by_confidence": blocked_by_confidence,
        "blocked_by_regime": blocked_by_regime,
        "blocked_by_cost": blocked_by_cost,
        "regime_no_evidence_bars": regime_no_evidence_bars,
        "total_bars": total_bars,
        "invested_bars": invested_bars,
        "accrued_cash_yield": bt_results.get("accrued_cash_yield", 0.0),
    }


def _capture_ratios(
    portfolio_curve: List[float],
    benchmark_curve: List[float],
) -> Dict[str, Optional[float]]:
    """Upside and downside capture ratios vs a benchmark.

    Capture = (mean portfolio return on benchmark-up/down days)
            / (mean benchmark return on those days) * 100
    """
    p = pd.Series(portfolio_curve, dtype=float).pct_change().dropna()
    b = pd.Series(benchmark_curve, dtype=float).pct_change().dropna()
    # Align lengths
    min_len = min(len(p), len(b))
    p, b = p.iloc[:min_len], b.iloc[:min_len]

    up_days = b > 0
    dn_days = b < 0

    up_cap = None
    if up_days.sum() > 0 and b[up_days].mean() != 0:
        up_cap = round(float(p[up_days].mean() / b[up_days].mean()) * 100, 1)

    dn_cap = None
    if dn_days.sum() > 0 and b[dn_days].mean() != 0:
        dn_cap = round(float(p[dn_days].mean() / b[dn_days].mean()) * 100, 1)

    cap_ratio = None
    if up_cap is not None and dn_cap is not None and abs(dn_cap) > SHARPE_STD_EPS:
        cap_ratio = round(float(up_cap / dn_cap), 2)

    return {
        "upside_capture": up_cap,
        "downside_capture": dn_cap,
        "capture_ratio": cap_ratio,
    }


def _calmar_ratio(equity_curve: List[float]) -> Optional[float]:
    """Annualized return / abs(max drawdown). None if no drawdown."""
    eq = pd.Series(equity_curve, dtype=float)
    n_bars = len(eq)
    if n_bars < 2:
        return None
    total_ret = eq.iloc[-1] / eq.iloc[0] - 1
    ann_ret = (1 + total_ret) ** (252 / n_bars) - 1
    cum = eq / eq.iloc[0]
    rm = cum.expanding().max()
    dd = ((cum - rm) / rm).min()
    if dd >= 0:
        return None
    return round(float(ann_ret / abs(dd)), 3)


def _return_attribution(
    total_curve: List[float],
    core_curve: List[float],
    tactical_curve: List[float],
    starting_capital: float,
) -> Dict:
    """Attribute % of total return to core (beta) vs tactical (alpha)."""
    total_ret = total_curve[-1] - total_curve[0]
    core_ret = core_curve[-1] - core_curve[0]
    tactical_ret = tactical_curve[-1] - tactical_curve[0]

    if abs(total_ret) < 1e-6:
        return {"core_pct": 0.0, "tactical_pct": 0.0,
                "core_dollars": 0.0, "tactical_dollars": 0.0}

    return {
        "core_pct": round(core_ret / total_ret * 100, 1) if total_ret != 0 else 0.0,
        "tactical_pct": round(tactical_ret / total_ret * 100, 1) if total_ret != 0 else 0.0,
        "core_dollars": round(core_ret, 2),
        "tactical_dollars": round(tactical_ret, 2),
    }


# =============================================================================
# STRATEGY FACADE
# =============================================================================

class MeanReversionStrategy:
    def __init__(self, config: Optional[MeanReversionConfig] = None):
        self.config = config or MeanReversionConfig()

    def get_metrics(self, df: pd.DataFrame) -> Dict:
        return calculate_mean_reversion_metrics(
            df, self.config.lookback_window, debug=self.config.debug)

    def compute_ratio(self, df: pd.DataFrame) -> pd.Series:
        return compute_ratio_series(df, self.config)

    def compute_ratio_z(self, ratio: pd.Series) -> pd.Series:
        return compute_ratio_z(ratio, self.config.ratio_lookback)

    def classify_regime(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        return classify_regime(df, self.config)

    def on_bar(
        self,
        history: pd.DataFrame,
        current_bar: pd.Series,
        *,
        has_position: bool,
        position_pct: float,
        reversal_ok: bool,
        quality_ok: bool,
    ) -> Action:
        """Single-bar strategy interface (signal generation decoupled from execution)."""
        if history is None or len(history) == 0:
            return Action.HOLD
        ratio = self.compute_ratio(history)
        rz_series = self.compute_ratio_z(ratio)
        z = rz_series.iloc[-1] if len(rz_series) > 0 else np.nan
        if pd.isna(z):
            return Action.HOLD
        labels, _ = self.classify_regime(history)
        regime = labels.iloc[-1] if len(labels) > 0 else Regime.AMBIGUOUS.value
        return determine_action(
            float(z), str(regime), bool(has_position), float(position_pct),
            self.config, bool(reversal_ok), bool(quality_ok),
        )

    def detect_mean_reversion_signals(self, df: pd.DataFrame, technicals: Dict) -> List[Dict]:
        ratio = self.compute_ratio(df)
        rz = self.compute_ratio_z(ratio)
        cfg = self.config
        signals = []
        n = len(df)
        scan_start = max(cfg.ratio_anchor_window + cfg.ratio_lookback, n - cfg.recent_signal_days)
        for i in range(scan_start, n):
            z = rz.iloc[i]
            if pd.isna(z) or z > cfg.entry_z:
                continue
            price = float(df["close"].iloc[i])
            atr_s = calculate_atr(df, 14)
            atr_val = float(atr_s.iloc[i]) if i < len(atr_s) and not pd.isna(atr_s.iloc[i]) else price * 0.02
            ts = df.index[i]
            signals.append({
                "bar_index": i, "timestamp": ts,
                "date_str": ts.strftime("%Y-%m-%d"),
                "signal_type": "RATIO_OVERSOLD",
                "z_score": round(float(z), 2),
                "entry_price": round(price, 2),
                "stop_loss": round(price - atr_val * cfg.stop_atr_multiple, 2),
                "take_profit": round(price + atr_val * cfg.target_atr_multiple, 2),
                "atr": round(atr_val, 2),
                "confidence_score": min(1.0, abs(z) / 3.0),
            })
        signals.sort(key=lambda s: s["confidence_score"], reverse=True)
        return signals


# =============================================================================
# REPORT GENERATION
# =============================================================================

def _action_summary(actions_log: List[Dict]) -> Dict:
    counts = {}
    for a in actions_log:
        counts[a["action"]] = counts.get(a["action"], 0) + 1
    return counts


def _regime_timeline(actions_log: List[Dict]) -> Dict:
    counts = {}
    for a in actions_log:
        r = a.get("regime", "UNKNOWN")
        counts[r] = counts.get(r, 0) + 1
    return counts


def _pct_time_invested(actions_log: List[Dict], fills: List[Dict]) -> float:
    invested_bars = 0
    in_position = False
    for a in actions_log:
        if a["action"] == "BUY":
            in_position = True
        elif a["action"] == "SELL":
            in_position = False
        if in_position:
            invested_bars += 1
    return round(invested_bars / max(len(actions_log), 1) * 100, 1)


def _append_confidence_bucket_table(lines: List[str], bt_results: Dict) -> None:
    """Append confidence bucket diagnostics when available."""
    summary = bt_results.get("confidence_bucket_summary", {})
    rows = summary.get("rows", [])
    if not rows or int(summary.get("total_trades", 0)) <= 0:
        return

    lines.append("-" * 74)
    lines.append("CONFIDENCE BUCKETS (REGIME PROXY - MR/SW SHARE AT ENTRY)")
    lines.append("-" * 74)
    lines.append("  Bucket Trades WinRate AvgPnL AvgHoldDays")
    for row in rows:
        sample_note = " (LOW SAMPLE)" if row.get("low_sample") else ""
        lines.append(
            f"  {row['bucket']:<10} {row['trades']:>6} {row['win_rate_pct']:>7.1f}% "
            f"{row['avg_pnl']:>8.2f} {row['avg_hold_days']:>11.1f}{sample_note}"
        )
    lines.append("")


def generate_report(
    ticker: str, df: pd.DataFrame, technicals: Dict,
    stat_metrics: Dict, bt_results: Dict, perf: Dict,
    bt_baseline: Optional[Dict], perf_baseline: Optional[Dict],
    info: Optional[Dict[str, Any]] = None, cfg: Optional[MeanReversionConfig] = None,
    data_quality: Optional[Dict[str, Any]] = None,
    bias_audit: Optional[Dict[str, Any]] = None,
    survivorship_sensitivity: Optional[Dict[str, Any]] = None,
) -> str:
    cfg = cfg or MeanReversionConfig()
    if not isinstance(bias_audit, dict):
        bias_audit = _build_bias_audit_payload(cfg, data_quality)
    if not isinstance(survivorship_sensitivity, dict):
        survivorship_sensitivity = _build_survivorship_sensitivity_payload(df, cfg, None, 0.0)
    L: List[str] = []
    name = (info or {}).get("name", ticker)

    L.append("=" * 74)
    L.append(f"{'MEAN REVERSION BACKTESTER v3.1 — LONG-ONLY ACTION ENGINE':^74}")
    L.append(f"{ticker} -- {name}")
    L.append("=" * 74)
    L.append(f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    L.append(f"Bars      : {len(df)} | Ratio: {cfg.ratio_mode} "
             f"(anchor={cfg.ratio_anchor_window}, lookback={cfg.ratio_lookback})")
    L.append(f"Capital   : ${cfg.starting_capital:,.0f} | "
             f"Max pos: {cfg.max_position_pct*100:.0f}% | "
             f"Add step: {cfg.add_step_pct*100:.0f}% | "
             f"Trim step: {cfg.trim_step_pct*100:.0f}%")
    L.append(f"Entry     : {cfg.entry_at} | Slippage: {cfg.slippage_pct*10000:.1f}bps | "
             f"Comm: ${cfg.commission_per_trade:.2f}")
    L.append(f"Regime    : filter={'ON' if cfg.regime_filter_enabled else 'OFF'} | "
             f"allowed={cfg.allowed_regimes} | adaptive={cfg.regime_adaptive_thresholds}")
    trim_desc = ", ".join(f"z>={z:.1f}:{int(f*100)}%" for z, f in cfg.trim_levels)
    L.append(f"Trims     : [{trim_desc}] | Min notional: ${cfg.min_trade_notional:.0f} "
             f"| Min shares: {cfg.min_shares}")
    L.append("")
    _append_data_quality_section(L, data_quality)

    # --- Statistical tests ---
    L.append("-" * 74)
    L.append("STATISTICAL TESTS")
    L.append("-" * 74)
    h = stat_metrics.get("hurst_exponent")
    h_r2 = stat_metrics.get("hurst_r_squared")
    if h is not None:
        L.append(f"  Hurst Exponent : {h:.3f} ({stat_metrics.get('hurst_interpretation')})  R2={h_r2:.3f}")
    else:
        L.append("  Hurst Exponent : N/A")
    # P0.1: always show ADF with note
    p = stat_metrics.get("adf_p_value")
    adf_note = stat_metrics.get("adf_note", "")
    if p is not None:
        sl = "STATIONARY" if stat_metrics.get("is_stationary") else "NON-STATIONARY"
        L.append(f"  ADF p-value    : {p:.4f} ({sl})")
    else:
        reason = f" [{adf_note}]" if adf_note else ""
        L.append(f"  ADF p-value    : N/A{reason}")
        if adf_note:
            L.append(f"    Note: ADF excluded from regime scoring (weight=0)")
    hl = stat_metrics.get("half_life_days")
    L.append(f"  Half-life      : {hl:.1f} days" if hl else "  Half-life      : N/A")
    vr = stat_metrics.get("variance_ratio")
    if vr:
        L.append(f"  Variance Ratio : {vr:.4f} ({stat_metrics.get('variance_ratio_interpretation')})")
    else:
        L.append("  Variance Ratio : N/A")
    L.append(f"  Stat Verdict   : {stat_metrics.get('stat_verdict', stat_metrics.get('verdict', 'N/A'))}")

    # P1.2: regime-derived verdict
    rt = _regime_timeline(bt_results["actions_log"])
    total_bars = max(sum(rt.values()), 1)
    dominant = max(rt.items(), key=lambda kv: kv[1])[0] if rt else "UNKNOWN"
    dominant_pct = rt.get(dominant, 0) / total_bars * 100
    L.append(f"  Regime Verdict : {dominant} ({dominant_pct:.0f}% of bars)")
    L.append("")

    # --- A/B Comparison ---
    if perf_baseline is not None:
        L.append("=" * 74)
        L.append("A/B COMPARISON: Regime-Filtered vs Baseline (no filter)")
        L.append("=" * 74)

        # P0.3: sanity warning
        if (bt_results["blocked_count"] == 0 and bt_baseline is not None
                and bt_baseline["blocked_count"] == 0
                and abs(perf["total_return_pct"] - perf_baseline["total_return_pct"]) < 0.01):
            L.append("  ** NOTE: Regime filter had no effect in this run.")
            L.append("     (No signals occurred during TRENDING regime bars.)")
            L.append("")

        L.append(f"  {'Metric':<28} {'Filtered':>14} {'Baseline':>14}")
        L.append("  " + "-" * 58)

        def _row(label, key, fmt=".2f"):
            v1 = perf.get(key, 0)
            v2 = perf_baseline.get(key, 0)
            s1 = f"{v1:{fmt}}" if isinstance(v1, (int, float)) else str(v1)
            s2 = f"{v2:{fmt}}" if isinstance(v2, (int, float)) else str(v2)
            L.append(f"  {label:<28} {s1:>14} {s2:>14}")

        _row("Total Return %", "total_return_pct", "+.2f")
        _row("Annualized Return %", "annualized_return_pct", "+.2f")
        _row("Sharpe Ratio", "sharpe_ratio", ".3f")
        _row("Max Drawdown %", "max_drawdown_pct", ".2f")
        _row("Profit Factor", "profit_factor")
        _row("Total Fills", "total_fills", "d")
        _row("BUY fills", "fills_buy", "d")
        _row("ADD fills", "fills_add", "d")
        _row("REDUCE fills", "fills_reduce", "d")
        _row("SELL fills", "fills_sell", "d")

        pct_inv = _pct_time_invested(bt_results["actions_log"], bt_results["fills"])
        pct_inv_b = _pct_time_invested(bt_baseline["actions_log"], bt_baseline["fills"]) if bt_baseline else 0
        L.append(f"  {'% Time Invested':<28} {pct_inv:>13.1f}% {pct_inv_b:>13.1f}%")
        L.append(f"  {'Signals Blocked':<28} {bt_results['blocked_count']:>14d} "
                 f"{bt_baseline['blocked_count'] if bt_baseline else 0:>14d}")
        bh = perf.get("buy_and_hold_return_pct")
        if bh is not None:
            L.append(f"  {'Buy & Hold Return %':<28} {bh:>+13.2f}%")
        L.append("")

    # --- Backtest performance ---
    L.append("=" * 74)
    L.append("BACKTEST PERFORMANCE (Regime-Filtered)")
    L.append("=" * 74)
    L.append(f"  Starting Capital     : ${perf['starting_capital']:>12,.2f}")
    L.append(f"  Ending Capital       : ${perf['ending_capital']:>12,.2f}")
    L.append(f"  Total Return         : {perf['total_return_pct']:>+10.2f}%")
    L.append(f"  Annualized Return    : {perf['annualized_return_pct']:>+10.2f}%")
    L.append(f"  Sharpe Ratio         : {_format_sharpe(perf['sharpe_ratio'], width=10)}")
    L.append(f"  Max Drawdown         : {perf['max_drawdown_pct']:>10.2f}%")
    L.append(f"  Profit Factor        : {str(perf['profit_factor']):>10}")
    bh = perf.get("buy_and_hold_return_pct")
    if bh is not None:
        L.append(f"  Buy & Hold Return    : {bh:>+10.2f}%")
    L.append("")

    # --- Action summary ---
    ac = _action_summary(bt_results["actions_log"])
    L.append("-" * 74)
    L.append("LONG-ONLY ACTIONS SUMMARY")
    L.append("-" * 74)
    for act_name in ["BUY", "ADD", "HOLD", "REDUCE", "SELL", "BLOCKED"]:
        L.append(f"  {act_name:<12}: {ac.get(act_name, 0):>6}")
    L.append("")

    # --- Regime timeline ---
    L.append("-" * 74)
    L.append("REGIME TIMELINE")
    L.append("-" * 74)
    for r_name in [Regime.MEAN_REVERTING.value, Regime.SIDEWAYS.value,
                   Regime.AMBIGUOUS.value, Regime.TRENDING.value]:
        cnt = rt.get(r_name, 0)
        L.append(f"  {r_name:<20}: {cnt:>5} bars ({cnt/total_bars*100:.1f}%)")
    no_evidence_bars = sum(
        1 for a in bt_results.get("actions_log", [])
        if str(a.get("regime", "")) == "NO_EVIDENCE"
    )
    if no_evidence_bars > 0:
        L.append(f"  {'NO_EVIDENCE':<20}: {no_evidence_bars:>5} bars")
    L.append("")

    # --- Diagnostics: exit reason breakdown ---
    exit_bd = _exit_reason_breakdown(bt_results["fills"])
    if exit_bd:
        L.append("-" * 74)
        L.append("EXIT REASON BREAKDOWN")
        L.append("-" * 74)
        L.append(f"  {'Reason':<22} {'Count':>5} {'AvgPnL':>10} {'AvgDays':>8} {'Wins':>5} {'Loss':>5}")
        for reason, d in sorted(exit_bd.items()):
            L.append(f"  {reason:<22} {d['count']:>5} "
                     f"${d['avg_pnl']:>+8.2f} {d['avg_days']:>7.1f} "
                     f"{d['wins']:>5} {d['losses']:>5}")
        L.append("")

    # --- Diagnostics: metrics by regime ---
    mbr = _metrics_by_regime(bt_results["actions_log"], bt_results["fills"])
    if mbr:
        L.append("-" * 74)
        L.append("METRICS BY REGIME")
        L.append("-" * 74)
        for r_name, d in sorted(mbr.items()):
            w = d["wins"] + d["losses"]
            wr = f"{d['wins']/w*100:.0f}%" if w > 0 else "N/A"
            L.append(f"  {r_name:<20}: {d['fills']:>3} fills  P&L=${d['realized_pnl']:>+9.2f}  WR={wr}")
        L.append("")

    # --- Confidence buckets ---
    _append_confidence_bucket_table(L, bt_results)

    # --- Regime transitions ---
    trans = _regime_transition_matrix(bt_results["actions_log"])
    if trans:
        L.append("-" * 74)
        L.append("REGIME TRANSITIONS")
        L.append("-" * 74)
        for from_r, tos in sorted(trans.items()):
            for to_r, cnt in sorted(tos.items()):
                L.append(f"  {from_r:<20} -> {to_r:<20}: {cnt:>4}")
        L.append("")

    # --- Fill log ---
    fills_list = bt_results.get("fills", [])
    if fills_list:
        L.append("-" * 74)
        L.append("FILL LOG")
        L.append("-" * 74)
        L.append(f"  {'#':>3}  {'Date':>10}  {'Action':<8}  {'Price':>8}  "
                 f"{'Shares':>6}  {'P&L':>10}  {'Reason'}")
        L.append("  " + "-" * 68)
        for idx, f in enumerate(fills_list[:30], 1):
            pnl_str = f"${f.get('realized_pnl', 0):>+9.2f}" if "realized_pnl" in f else "         "
            reason = f.get("reason", f.get("regime", ""))
            L.append(
                f"  {idx:>3}  {f['date']:>10}  {f['action']:<8}  "
                f"${f['price']:>7.2f}  {f['shares']:>6}  {pnl_str}  {reason}"
            )
        if len(fills_list) > 30:
            L.append(f"  ... and {len(fills_list) - 30} more fills")
    L.append("")

    _append_bias_audit_section(L, bias_audit, survivorship_sensitivity)
    L.append("=" * 74)
    L.append("DISCLAIMER: Educational only. NOT financial advice.")
    L.append("Ratio framework is a configurable placeholder; it does NOT replicate")
    L.append("any proprietary valuation system. Past performance != future results.")
    L.append("=" * 74)
    return "\n".join(L)


# =============================================================================
# CORE POSITION (v3.2 — Two-Layer)
# =============================================================================

@dataclass
class CorePosition:
    """Buy-and-hold core sleeve.  Never sells once established.

    Instant mode: buys at first bar's close.
    DCA mode (v3.4): decision at bar i schedules buy at open of bar i+1
    (next_open execution). No buy is scheduled on the last bar.
    The core is a passive benchmark
    sleeve — not a tactical trade — so the one-tick difference is immaterial.
    """
    shares: float = 0.0       # fractional OK; it's an accounting sleeve
    entry_price: float = 0.0
    cost_basis: float = 0.0
    cash: float = 0.0         # uninvested core cash (used in DCA mode)

    @property
    def is_established(self) -> bool:
        return self.shares > 0

    @property
    def avg_entry_price(self) -> float:
        return self.cost_basis / self.shares if self.shares > 0 else 0.0


# =============================================================================
# TWO-LAYER PORTFOLIO ENGINE (v3.2)
# =============================================================================

class TwoLayerPortfolioEngine:
    """Core (buy-and-hold) + Tactical (existing MR engine) wrapper.

    The core sleeve buys at bar-0 close and holds forever.
    The tactical sleeve receives ``tactical_allocation_pct * starting_capital``
    and runs the existing :class:`BacktestEngine` with its own ``starting_capital``.

    The engine produces three equity curves aligned 1:1 with ``df.index``:

    * **core_equity_curve** — core_shares * close[i]
    * **tactical_equity_curve** — from :class:`BacktestEngine`
    * **total_equity_curve** — sum of the above

    It also computes a **static 80/20 baseline** (core_pct% buy-hold + remaining
    cash, no rebalancing) for benchmark comparison.
    """

    def __init__(self, cfg: MeanReversionConfig):
        self.cfg = cfg

    def run(
        self,
        df: pd.DataFrame,
        ratio_z_series: pd.Series,
        regime_labels: pd.Series,
        regime_scores: pd.Series,
    ) -> Dict:
        cfg = self.cfg
        total_capital = cfg.starting_capital
        core_capital = total_capital * cfg.core_allocation_pct
        tactical_capital = total_capital * cfg.tactical_allocation_pct

        n = len(df)
        close = df["close"]
        daily_cash_rate = _daily_cash_yield_rate(cfg.cash_yield_annual_pct)

        # --- Core sleeve ---
        core = CorePosition()
        if cfg.core_entry_mode == "dca" and core_capital > 0:
            # DCA: deploy core_capital linearly over core_dca_days bars.
            # Decision at bar i schedules buy at open of bar i+1 (next_open).
            dca_start = min(cfg.core_dca_start, n - 1)
            dca_end = min(dca_start + cfg.core_dca_days, n)
            dca_n = max(dca_end - dca_start, 1)
            invest_per_day = core_capital / dca_n
            core.cash = core_capital
            _pending_dca_spend = 0.0  # scheduled spend awaiting next open

            core_eq: List[float] = []
            core_cash_curve: List[float] = []
            core_shares_curve: List[float] = []
            for i in range(n):
                # Earn yield on uninvested core cash
                if daily_cash_rate > 0 and core.cash > 0:
                    core.cash *= (1 + daily_cash_rate)

                # Execute pending DCA buy at this bar's open
                if _pending_dca_spend > 0 and core.cash > 0:
                    px = float(df["open"].iloc[i])
                    spend = min(_pending_dca_spend, core.cash)
                    cost_after_slip = px * (1 + cfg.core_dca_slippage_pct)
                    new_shares = spend / cost_after_slip
                    actual_cost = new_shares * cost_after_slip + cfg.core_dca_commission
                    if actual_cost > core.cash:
                        actual_cost = core.cash
                        new_shares = (actual_cost - cfg.core_dca_commission) / cost_after_slip
                        new_shares = max(0.0, new_shares)
                    core.shares += new_shares
                    core.cost_basis += new_shares * cost_after_slip
                    core.cash -= actual_cost
                    core.cash = max(0.0, core.cash)
                    _pending_dca_spend = 0.0

                # Schedule next DCA buy (guard: don't schedule on last bar)
                if dca_start <= i < dca_end and i + 1 < n and core.cash > 0:
                    _pending_dca_spend = invest_per_day

                bar_val = core.shares * float(close.iloc[i]) + core.cash
                core_eq.append(round(bar_val, 2))
                core_cash_curve.append(round(core.cash, 2))
                core_shares_curve.append(float(core.shares))

            core.entry_price = core.avg_entry_price
        elif cfg.core_entry_mode == "adaptive" and core_capital > 0:
            # Adaptive: deploy at variable rate based on trend/dd/vol.
            # Buys at same-close (no future price usage).
            core.cash = core_capital
            ada_start = min(int(cfg.core_adaptive_start), n - 1)

            # Pre-compute indicators (vectorised, no lookahead)
            close_arr = close.to_numpy(dtype=float)
            sma200 = pd.Series(close_arr, index=df.index).rolling(200, min_periods=200).mean().to_numpy()
            log_ret = np.empty(n, dtype=float)
            log_ret[0] = np.nan
            log_ret[1:] = np.log(close_arr[1:] / close_arr[:-1])
            vol_win = int(cfg.core_adaptive_vol_window)
            sigma_ann = pd.Series(log_ret).rolling(vol_win, min_periods=vol_win).std().to_numpy() * np.sqrt(252)
            dd_win = int(cfg.core_adaptive_dd_window)
            rolling_peak = pd.Series(close_arr, index=df.index).rolling(dd_win, min_periods=1).max().to_numpy()
            dd_arr = 1.0 - close_arr / np.maximum(rolling_peak, 1e-12)

            core_eq: List[float] = []
            core_cash_curve: List[float] = []
            core_shares_curve: List[float] = []
            core_adaptive_states: List[str] = []

            for i in range(n):
                # Earn yield on uninvested core cash
                if daily_cash_rate > 0 and core.cash > 0:
                    core.cash *= (1 + daily_cash_rate)

                # Deploy at same_close
                if i >= ada_start and core.cash > 0.01:
                    spend, state = _adaptive_core_deploy_amount(
                        close_i=close_arr[i],
                        sma200_i=sma200[i],
                        dd_i=dd_arr[i],
                        sigma_i=sigma_ann[i],
                        remaining_cash=core.cash,
                        base_days=int(cfg.core_adaptive_base_days),
                        slow_days=int(cfg.core_adaptive_slow_days),
                        fast_days=int(cfg.core_adaptive_fast_days),
                        vol_target=float(cfg.core_adaptive_vol_target),
                        vol_floor=float(cfg.core_adaptive_vol_floor),
                        max_deploy_pct=float(cfg.core_adaptive_max_deploy_pct),
                        min_deploy_pct=float(cfg.core_adaptive_min_deploy_pct),
                    )
                    core_adaptive_states.append(state)
                    if spend > 0 and close_arr[i] > 0:
                        spend = min(spend, core.cash)
                        new_shares = spend / close_arr[i]
                        core.shares += new_shares
                        core.cost_basis += spend
                        core.cash -= spend
                        core.cash = max(0.0, core.cash)
                else:
                    core_adaptive_states.append("WAITING" if i < ada_start else "DEPLETED")

                bar_val = core.shares * close_arr[i] + core.cash
                core_eq.append(round(bar_val, 2))
                core_cash_curve.append(round(core.cash, 2))
                core_shares_curve.append(float(core.shares))

            core.entry_price = core.avg_entry_price
        else:
            # Instant: buy at first close (original behavior)
            first_close = float(close.iloc[0])
            if core_capital > 0 and first_close > 0:
                core.shares = core_capital / first_close
                core.entry_price = first_close
                core.cost_basis = core_capital
            core_eq = [round(core.shares * float(close.iloc[i]), 2) for i in range(n)]
            core_cash_curve = [0.0 for _ in range(n)]
            core_shares_curve = [float(core.shares) for _ in range(n)]

        # --- Tactical sleeve: run existing BacktestEngine ---
        # CRITICAL: create a copy of cfg with overridden starting_capital.
        # Never mutate the caller's cfg.
        tactical_cfg = cfg.copy_with(
            starting_capital=tactical_capital,
            two_layer_mode=False,   # prevent infinite recursion
        )
        tactical_engine = BacktestEngine(tactical_cfg)
        tactical_bt = tactical_engine.run(df, ratio_z_series, regime_labels, regime_scores)
        tactical_eq = tactical_bt["equity_curve"]

        # --- Optional sleeve rebalancing (drift guard) ---
        rebalancing_events = 0
        if cfg.rebalance_freq in ("M", "Q", "A") and n > 1:
            try:
                idx = pd.DatetimeIndex(df.index)
                idx_naive = idx.tz_localize(None) if idx.tz is not None else idx
                periods = idx_naive.to_period(cfg.rebalance_freq)
                rebalance_flags = [False] * n
                for i in range(1, n):
                    rebalance_flags[i] = periods[i] != periods[i - 1]

                core_raw = np.asarray(core_eq, dtype=float)
                tact_raw = np.asarray(tactical_eq, dtype=float)
                core_adj = np.zeros(n, dtype=float)
                tact_adj = np.zeros(n, dtype=float)
                core_adj[0] = core_raw[0]
                tact_adj[0] = tact_raw[0]

                for i in range(1, n):
                    core_prev = core_raw[i - 1]
                    tact_prev = tact_raw[i - 1]
                    core_ret = (core_raw[i] / core_prev) if core_prev > 0 else 1.0
                    tact_ret = (tact_raw[i] / tact_prev) if tact_prev > 0 else 1.0
                    if not np.isfinite(core_ret):
                        core_ret = 1.0
                    if not np.isfinite(tact_ret):
                        tact_ret = 1.0
                    core_adj[i] = core_adj[i - 1] * core_ret
                    tact_adj[i] = tact_adj[i - 1] * tact_ret

                    if rebalance_flags[i]:
                        total_now = core_adj[i] + tact_adj[i]
                        if total_now > 0:
                            core_w = core_adj[i] / total_now
                            drift = abs(core_w - float(cfg.core_allocation_pct))
                            if drift > float(cfg.rebalance_drift_threshold):
                                target_core = total_now * float(cfg.core_allocation_pct)
                                target_tact = total_now * float(cfg.tactical_allocation_pct)
                                turnover = abs(core_adj[i] - target_core)
                                tc = turnover * float(cfg.slippage_pct)
                                if turnover > 0:
                                    tc += 2.0 * float(cfg.commission_per_trade)
                                total_after_cost = max(total_now - tc, 0.0)
                                core_adj[i] = total_after_cost * float(cfg.core_allocation_pct)
                                tact_adj[i] = total_after_cost * float(cfg.tactical_allocation_pct)
                                rebalancing_events += 1

                core_eq = [round(float(v), 2) for v in core_adj]
                tactical_eq = [round(float(v), 2) for v in tact_adj]
            except Exception as exc:
                if cfg.debug:
                    logger.warning("Rebalancing step failed; continuing without rebalance: %s", exc)

        # --- Total ---
        total_eq = [round(c + t, 2) for c, t in zip(core_eq, tactical_eq)]

        # --- Static baseline: core_pct% buy-hold (instant) + rest in cash ---
        # No rebalancing; constant initial dollar split.
        first_close_base = float(close.iloc[0])
        cash_sleeve = total_capital * cfg.tactical_allocation_pct
        baseline_eq: List[float] = []
        cash_bal = cash_sleeve
        for i in range(n):
            stock_val = (core_capital * float(close.iloc[i]) / first_close_base
                         if first_close_base > 0 else 0.0)
            cash_bal *= (1 + daily_cash_rate)
            baseline_eq.append(round(stock_val + cash_bal, 2))

        # --- Matched-core-entry baseline: same core entry mode, no tactical ---
        # Tactical portion stays as cash earning yield.
        baseline_matched_eq: List[float] = []
        if cfg.core_entry_mode == "dca" and core_capital > 0:
            # Replay DCA for baseline (identical core entry, tactical = cash)
            m_shares = 0.0
            m_cash_core = core_capital
            m_cash_tactical = tactical_capital
            dca_start_m = min(cfg.core_dca_start, n - 1)
            dca_end_m = min(dca_start_m + cfg.core_dca_days, n)
            dca_n_m = max(dca_end_m - dca_start_m, 1)
            invest_per_day_m = core_capital / dca_n_m
            _m_pending = 0.0
            for i in range(n):
                if daily_cash_rate > 0:
                    if m_cash_core > 0:
                        m_cash_core *= (1 + daily_cash_rate)
                    m_cash_tactical *= (1 + daily_cash_rate)
                # Execute pending buy at this bar's open
                if _m_pending > 0 and m_cash_core > 0:
                    px = float(df["open"].iloc[i])
                    spend = min(_m_pending, m_cash_core)
                    cost_after_slip = px * (1 + cfg.core_dca_slippage_pct)
                    new_shares = spend / cost_after_slip
                    actual_cost = new_shares * cost_after_slip + cfg.core_dca_commission
                    if actual_cost > m_cash_core:
                        actual_cost = m_cash_core
                        new_shares = (actual_cost - cfg.core_dca_commission) / cost_after_slip
                        new_shares = max(0.0, new_shares)
                    m_shares += new_shares
                    m_cash_core -= actual_cost
                    m_cash_core = max(0.0, m_cash_core)
                    _m_pending = 0.0
                # Schedule next buy (guard: not on last bar)
                if dca_start_m <= i < dca_end_m and i + 1 < n and m_cash_core > 0:
                    _m_pending = invest_per_day_m
                val = m_shares * float(close.iloc[i]) + m_cash_core + m_cash_tactical
                baseline_matched_eq.append(round(val, 2))
        elif cfg.core_entry_mode == "adaptive" and core_capital > 0:
            # Replay adaptive core entry for matched baseline (tactical = cash)
            m_shares = 0.0
            m_cash_core = core_capital
            m_cash_tactical = tactical_capital
            m_ada_start = min(int(cfg.core_adaptive_start), n - 1)
            # Re-use same pre-computed indicators (close_arr, sma200, etc.)
            # They were computed above in the adaptive core block.
            for i in range(n):
                if daily_cash_rate > 0:
                    if m_cash_core > 0:
                        m_cash_core *= (1 + daily_cash_rate)
                    m_cash_tactical *= (1 + daily_cash_rate)
                if i >= m_ada_start and m_cash_core > 0.01:
                    spend_m, _ = _adaptive_core_deploy_amount(
                        close_i=close_arr[i],
                        sma200_i=sma200[i],
                        dd_i=dd_arr[i],
                        sigma_i=sigma_ann[i],
                        remaining_cash=m_cash_core,
                        base_days=int(cfg.core_adaptive_base_days),
                        slow_days=int(cfg.core_adaptive_slow_days),
                        fast_days=int(cfg.core_adaptive_fast_days),
                        vol_target=float(cfg.core_adaptive_vol_target),
                        vol_floor=float(cfg.core_adaptive_vol_floor),
                        max_deploy_pct=float(cfg.core_adaptive_max_deploy_pct),
                        min_deploy_pct=float(cfg.core_adaptive_min_deploy_pct),
                    )
                    if spend_m > 0 and close_arr[i] > 0:
                        spend_m = min(spend_m, m_cash_core)
                        m_shares += spend_m / close_arr[i]
                        m_cash_core -= spend_m
                        m_cash_core = max(0.0, m_cash_core)
                val = m_shares * close_arr[i] + m_cash_core + m_cash_tactical
                baseline_matched_eq.append(round(val, 2))
        else:
            # Instant mode: matched baseline == original baseline
            baseline_matched_eq = list(baseline_eq)

        return {
            # Curves (all length == n)
            "core_equity_curve": core_eq,
            "tactical_equity_curve": tactical_eq,
            "total_equity_curve": total_eq,
            "baseline_equity_curve": baseline_eq,
            "baseline_matched_core_entry": baseline_matched_eq,
            "core_cash_curve": core_cash_curve,
            "core_shares_curve": core_shares_curve,
            # Core accounting
            "core_shares": core.shares,
            "core_entry_price": core.entry_price,
            "core_capital": core_capital,
            "core_entry_mode": cfg.core_entry_mode,
            "core_dca_days": cfg.core_dca_days if cfg.core_entry_mode == "dca" else 0,
            "core_adaptive_states": (
                core_adaptive_states
                if cfg.core_entry_mode == "adaptive" else []
            ),
            # Tactical passthrough
            "tactical_capital": tactical_capital,
            "tactical_bt": tactical_bt,
            "rebalance_freq": cfg.rebalance_freq,
            "rebalance_drift_threshold": float(cfg.rebalance_drift_threshold),
            "rebalancing_events": int(rebalancing_events),
        }


# =============================================================================
# VISUALIZATION (v3.2)
# =============================================================================

def compute_buy_hold_curve(
    df: pd.DataFrame, starting_capital: float,
) -> pd.Series:
    """Full buy-and-hold equity curve (no slippage/commissions)."""
    close = df["close"]
    first_close = float(close.iloc[0])
    shares = starting_capital / first_close
    return close * shares


def generate_benchmark_outputs(
    equity_curve: List[float],
    df: pd.DataFrame,
    starting_capital: float,
    ticker: str,
    output_dir: str = "reports",
    *,
    two_layer_result: Optional[Dict] = None,
    bt_results: Optional[Dict] = None,
) -> Dict[str, str]:
    """Produce benchmark PNG, drawdown PNG, curves CSV, and enhanced plots.

    New in v3.3:
    - Exposure vs Price scatter (requires *bt_results*)
    - Return Contribution stacked area (requires *two_layer_result*)
    - Enhanced Underwater drawdown (total + baseline)

    When *two_layer_result* is provided the charts include core / tactical /
    total / static-baseline curves instead of a single strategy line.

    Returns ``{kind: filepath}`` for every file written.
    Requires matplotlib; returns empty dict (with warning) if unavailable.
    """
    if not MPL_OK:
        logger.warning("matplotlib not installed — skipping plot outputs. "
                       "Install with: pip install matplotlib")
        return {}

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, str] = {}

    # ----- Build series -----
    eq_series = pd.Series(equity_curve, index=df.index, dtype=float)
    bh_series = compute_buy_hold_curve(df, starting_capital)

    if two_layer_result is not None:
        core_s = pd.Series(two_layer_result["core_equity_curve"], index=df.index, dtype=float)
        tact_s = pd.Series(two_layer_result["tactical_equity_curve"], index=df.index, dtype=float)
        total_s = pd.Series(two_layer_result["total_equity_curve"], index=df.index, dtype=float)
        base_s = pd.Series(two_layer_result["baseline_equity_curve"], index=df.index, dtype=float)

    # ----- 1. Benchmark chart -----
    fig, ax = plt.subplots(figsize=(10, 5))

    if two_layer_result is not None:
        # Normalize to growth-of-$1
        t0 = total_s.iloc[0]
        ax.plot(total_s.index.to_numpy(), np.asarray((total_s / t0).to_numpy(dtype=float), dtype=float),
                label="Total (Core+Tactical)", linewidth=1.6, color="#2563eb")
        ax.plot(core_s.index.to_numpy(), np.asarray((core_s / core_s.iloc[0]).to_numpy(dtype=float), dtype=float),
                label="Core (buy & hold)", linewidth=1.0, color="#16a34a", alpha=0.7)
        ax.plot(tact_s.index.to_numpy(), np.asarray((tact_s / tact_s.iloc[0]).to_numpy(dtype=float), dtype=float),
                label="Tactical (MR sleeve)", linewidth=1.0, color="#9333ea", alpha=0.7)
        ax.plot(base_s.index.to_numpy(), np.asarray((base_s / base_s.iloc[0]).to_numpy(dtype=float), dtype=float),
                label="Static 80/20 baseline", linewidth=1.2, color="#6b7280", linestyle="--")
    else:
        eq_norm = eq_series / eq_series.iloc[0]
        bh_norm = bh_series / bh_series.iloc[0]
        ax.plot(eq_norm.index.to_numpy(), np.asarray(eq_norm.to_numpy(dtype=float), dtype=float),
                label="Strategy", linewidth=1.4, color="#2563eb")
        ax.plot(bh_norm.index.to_numpy(), np.asarray(bh_norm.to_numpy(dtype=float), dtype=float),
                label="Buy & Hold", linewidth=1.2, color="#6b7280", linestyle="--")

    ax.set_title(f"{ticker} — Equity Curves (normalized)", fontsize=12)
    ax.set_ylabel("Growth of $1")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate()
    fig.tight_layout()
    p = out / f"{ticker}_benchmark.png"
    try:
        fig.savefig(str(p), dpi=150)
        paths["benchmark_png"] = str(p)
    finally:
        plt.close(fig)

    # ----- 2. Drawdown chart -----
    primary = pd.Series(
        two_layer_result["total_equity_curve"] if two_layer_result else equity_curve,
        index=df.index, dtype=float,
    )
    cum = primary / primary.iloc[0]
    running_max = cum.expanding().max()
    drawdown = (cum - running_max) / running_max

    fig2, ax2 = plt.subplots(figsize=(10, 3.5))
    ax2.fill_between(
        drawdown.index.to_numpy(),
        np.asarray(drawdown.to_numpy(dtype=float), dtype=float),
        0.0,
                     color="#dc2626", alpha=0.35, label="Strategy Drawdown")
    ax2.plot(
        drawdown.index.to_numpy(),
        np.asarray(drawdown.to_numpy(dtype=float), dtype=float),
        color="#dc2626",
        linewidth=0.8,
    )

    if two_layer_result is not None:
        base_cum = base_s / base_s.iloc[0]
        base_rm = base_cum.expanding().max()
        base_dd = (base_cum - base_rm) / base_rm
        ax2.plot(base_dd.index.to_numpy(), np.asarray(base_dd.to_numpy(dtype=float), dtype=float), color="#6b7280", linewidth=0.8,
                 linestyle="--", label="Baseline DD")

    ax2.set_title(f"{ticker} — Drawdown", fontsize=12)
    ax2.set_ylabel("Drawdown")
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax2.legend(loc="lower left", framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig2.autofmt_xdate()
    fig2.tight_layout()
    p2 = out / f"{ticker}_drawdown.png"
    try:
        fig2.savefig(str(p2), dpi=150)
        paths["drawdown_png"] = str(p2)
    finally:
        plt.close(fig2)

    # ----- 3. CSV -----
    dt_index = pd.DatetimeIndex(df.index)
    csv_data: Dict[str, Any] = {
        "date": dt_index.strftime("%Y-%m-%d"),
        "equity_strategy": eq_series.values,
        "equity_buyhold": bh_series.values,
    }
    if two_layer_result is not None:
        csv_data["equity_core"] = core_s.values
        csv_data["equity_tactical"] = tact_s.values
        csv_data["equity_total"] = total_s.values
        csv_data["equity_baseline_80_20"] = base_s.values
        matched_eq = two_layer_result.get("baseline_matched_core_entry", [])
        if matched_eq and len(matched_eq) == len(df):
            csv_data["equity_baseline_matched"] = matched_eq

    csv_df = pd.DataFrame(csv_data)
    p3 = out / f"{ticker}_curves.csv"
    csv_df.to_csv(str(p3), index=False)
    paths["curves_csv"] = str(p3)

    # ----- 4. Exposure vs Price scatter -----
    exposure = (bt_results or {}).get("exposure_pct_curve", [])
    if exposure and len(exposure) == len(df):
        fig4, ax4 = plt.subplots(figsize=(8, 5))
        prices = np.asarray(df["close"].to_numpy(dtype=float), dtype=float)
        exposure_arr = np.asarray(exposure, dtype=float)
        ax4.scatter(prices, exposure_arr, s=3, alpha=0.35, color="#9333ea")
        ax4.set_xlabel("Asset Price ($)")
        ax4.set_ylabel("Tactical Exposure (%)")
        ax4.set_title(f"{ticker} — Exposure vs Price", fontsize=12)
        ax4.grid(True, alpha=0.3)
        fig4.tight_layout()
        p4 = out / f"{ticker}_exposure.png"
        try:
            fig4.savefig(str(p4), dpi=150)
            paths["exposure_png"] = str(p4)
        finally:
            plt.close(fig4)

    # ----- 5. Return Contribution (two-layer only) -----
    if two_layer_result is not None:
        fig5, ax5 = plt.subplots(figsize=(10, 5))
        # Express as % of starting capital
        sc = starting_capital
        core_contrib = (core_s - core_s.iloc[0]) / sc * 100
        tact_contrib = (tact_s - tact_s.iloc[0]) / sc * 100

        ax5.fill_between(core_contrib.index.to_numpy(), 0.0, np.asarray(core_contrib.to_numpy(dtype=float), dtype=float),
                         alpha=0.4, color="#16a34a", label="Core (beta)")
        ax5.fill_between(tact_contrib.index.to_numpy(), 0.0, np.asarray(tact_contrib.to_numpy(dtype=float), dtype=float),
                         alpha=0.4, color="#9333ea", label="Tactical (alpha)")
        total_contrib = core_contrib + tact_contrib
        ax5.plot(total_contrib.index.to_numpy(), np.asarray(total_contrib.to_numpy(dtype=float), dtype=float),
                 linewidth=1.4, color="#2563eb", label="Total")
        ax5.axhline(0, color="black", linewidth=0.5, linestyle="-")
        ax5.set_title(f"{ticker} — Return Contribution (% of starting capital)", fontsize=12)
        ax5.set_ylabel("Cumulative Return Contribution (%)")
        ax5.legend(loc="upper left", framealpha=0.9)
        ax5.grid(True, alpha=0.3)
        ax5.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax5.xaxis.set_major_locator(mdates.AutoDateLocator())
        fig5.autofmt_xdate()
        fig5.tight_layout()
        p5 = out / f"{ticker}_contribution.png"
        try:
            fig5.savefig(str(p5), dpi=150)
            paths["contribution_png"] = str(p5)
        finally:
            plt.close(fig5)

    # ----- 6. Enhanced Underwater (total + baseline) -----
    if two_layer_result is not None:
        fig6, ax6 = plt.subplots(figsize=(10, 3.5))
        # Total portfolio drawdown
        t_cum = total_s / total_s.iloc[0]
        t_rm = t_cum.expanding().max()
        t_dd = (t_cum - t_rm) / t_rm
        ax6.fill_between(t_dd.index.to_numpy(), np.asarray(t_dd.to_numpy(dtype=float), dtype=float), 0.0,
                         color="#2563eb", alpha=0.3, label="Total Portfolio")
        ax6.plot(t_dd.index.to_numpy(), np.asarray(t_dd.to_numpy(dtype=float), dtype=float), color="#2563eb", linewidth=0.8)
        # Baseline drawdown
        b_cum = base_s / base_s.iloc[0]
        b_rm = b_cum.expanding().max()
        b_dd = (b_cum - b_rm) / b_rm
        ax6.plot(b_dd.index.to_numpy(), np.asarray(b_dd.to_numpy(dtype=float), dtype=float), color="#6b7280", linewidth=0.8,
                 linestyle="--", label="Static Baseline")
        ax6.set_title(f"{ticker} — Underwater Chart", fontsize=12)
        ax6.set_ylabel("Drawdown")
        ax6.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0%}"))
        ax6.legend(loc="lower left", framealpha=0.9)
        ax6.grid(True, alpha=0.3)
        ax6.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax6.xaxis.set_major_locator(mdates.AutoDateLocator())
        fig6.autofmt_xdate()
        fig6.tight_layout()
        p6 = out / f"{ticker}_underwater.png"
        try:
            fig6.savefig(str(p6), dpi=150)
            paths["underwater_png"] = str(p6)
        finally:
            plt.close(fig6)

    return paths


# =============================================================================
# TWO-LAYER REPORT
# =============================================================================

def generate_two_layer_report(
    ticker: str,
    df: pd.DataFrame,
    tl_result: Dict,
    tactical_perf: Dict,
    stat_metrics: Dict,
    bt_results: Dict,
    info: Dict,
    cfg: MeanReversionConfig,
    data_quality: Optional[Dict[str, Any]] = None,
    bias_audit: Optional[Dict[str, Any]] = None,
    survivorship_sensitivity: Optional[Dict[str, Any]] = None,
) -> str:
    """Generate a professional Two-Layer report separating Core / Tactical / Total."""
    if not isinstance(bias_audit, dict):
        bias_audit = _build_bias_audit_payload(cfg, data_quality)
    if not isinstance(survivorship_sensitivity, dict):
        survivorship_sensitivity = _build_survivorship_sensitivity_payload(df, cfg, tl_result, 0.0)
    L: List[str] = []
    name = (info or {}).get("name", ticker)
    n = len(df)
    is_tactical_only = cfg.two_layer_mode and cfg.core_allocation_pct <= 0.0

    # ---- Header ----
    L.append("=" * 74)
    title = (
        "MEAN REVERSION BACKTESTER v3.3 - TACTICAL-ONLY MODE (Core disabled)"
        if is_tactical_only
        else "MEAN REVERSION BACKTESTER v3.3 - TWO-LAYER PORTFOLIO"
    )
    L.append(f"{title:^74}")
    L.append(f"{ticker} -- {name}")
    L.append("=" * 74)
    L.append(f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    L.append(f"Bars      : {n}")
    if is_tactical_only:
        L.append(f"Capital   : ${cfg.starting_capital:,.0f}  "
                 f"(Tactical {cfg.tactical_allocation_pct*100:.0f}% = "
                 f"${tl_result['tactical_capital']:,.0f})")
    else:
        L.append(f"Capital   : ${cfg.starting_capital:,.0f}  "
                 f"(Core {cfg.core_allocation_pct*100:.0f}% = "
                 f"${tl_result['core_capital']:,.0f}  |  "
                 f"Tactical {cfg.tactical_allocation_pct*100:.0f}% = "
                 f"${tl_result['tactical_capital']:,.0f})")
    if cfg.cash_yield_annual_pct > 0:
        L.append(f"Cash Yield: {cfg.cash_yield_annual_pct:.2f}% annualized (T-bill proxy)")
    core_entry_mode = tl_result.get("core_entry_mode", cfg.core_entry_mode)
    if core_entry_mode == "dca":
        dca_days = tl_result.get("core_dca_days", cfg.core_dca_days)
        L.append(f"Core Entry: DCA over {dca_days} bars "
                 f"(start={cfg.core_dca_start}, slip={cfg.core_dca_slippage_pct*10000:.1f}bps, "
                 f"comm=${cfg.core_dca_commission:.2f})")
    elif core_entry_mode == "adaptive":
        ada_states = tl_result.get("core_adaptive_states", [])
        from collections import Counter as _Counter
        state_counts = _Counter(ada_states)
        L.append(f"Core Entry: Adaptive (base={cfg.core_adaptive_base_days}, "
                 f"slow={cfg.core_adaptive_slow_days}, fast={cfg.core_adaptive_fast_days}, "
                 f"vol_target={cfg.core_adaptive_vol_target:.0%}, "
                 f"dd_window={cfg.core_adaptive_dd_window})")
        if state_counts:
            parts = [f"{k}={v}" for k, v in sorted(state_counts.items())]
            L.append(f"  States  : {', '.join(parts)}")
    L.append("")
    _append_data_quality_section(L, data_quality)

    # Helper
    def _pct(start, end):
        return (end / start - 1) * 100 if start > 0 else 0.0

    def _sharpe(curve):
        sharpe = _excess_return_sharpe(
            curve, cash_yield_annual_pct=cfg.cash_yield_annual_pct,
        )
        return round(sharpe, 3) if sharpe is not None else "N/A"

    def _maxdd(curve):
        s = pd.Series(curve, dtype=float)
        if len(s) < 2 or s.iloc[0] <= 0:
            return 0.0
        cum = s / s.iloc[0]
        rm = cum.expanding().max()
        dd = (cum - rm) / rm
        return float(dd.min()) * 100

    total_eq = tl_result["total_equity_curve"]
    core_eq = tl_result["core_equity_curve"]
    tact_eq = tl_result["tactical_equity_curve"]
    base_eq = tl_result["baseline_equity_curve"]
    tact_diag = _tactical_diagnostics(bt_results)
    cash_proxy_bars = max(len(tact_eq) - 1, 0)
    cash_bench_ret = _cash_proxy_compounded_return_pct(
        cfg.cash_yield_annual_pct, cash_proxy_bars,
    )
    # In tactical-only mode baseline is the cash proxy; keep this numerically aligned.
    if is_tactical_only and len(base_eq) > 1:
        cash_bench_ret = _pct(base_eq[0], base_eq[-1])

    # ---- TOTAL PORTFOLIO ----
    L.append("=" * 74)
    L.append("TOTAL PORTFOLIO PERFORMANCE (Tactical Only)" if is_tactical_only
             else "TOTAL PORTFOLIO PERFORMANCE (Core + Tactical)")
    L.append("=" * 74)
    t_ret = _pct(total_eq[0], total_eq[-1])
    t_sharpe = _sharpe(total_eq)
    t_dd = _maxdd(total_eq)
    L.append(f"  Starting Capital : ${total_eq[0]:>12,.2f}")
    L.append(f"  Ending Capital   : ${total_eq[-1]:>12,.2f}")
    L.append(f"  Total Return     : {t_ret:>+10.2f}%")
    L.append(f"  Sharpe Ratio     : {_format_sharpe(t_sharpe, width=10)}")
    L.append(f"  Max Drawdown     : {t_dd:>10.2f}%")
    L.append("")

    c_ret = 0.0
    c_sharpe: Any = "N/A"
    c_dd = 0.0
    if not is_tactical_only:
        # ---- CORE LAYER ----
        L.append("=" * 74)
        L.append("CORE LAYER (Buy & Hold)")
        L.append("=" * 74)
        c_ret = _pct(core_eq[0], core_eq[-1])
        c_sharpe = _sharpe(core_eq)
        c_dd = _maxdd(core_eq)
        L.append(f"  Entry Price      : ${tl_result['core_entry_price']:>12.2f}")
        L.append(f"  Shares           : {tl_result['core_shares']:>12.2f}")
        L.append(f"  Starting Value   : ${core_eq[0]:>12,.2f}")
        L.append(f"  Ending Value     : ${core_eq[-1]:>12,.2f}")
        L.append(f"  Total Return     : {c_ret:>+10.2f}%")
        L.append(f"  Sharpe Ratio     : {_format_sharpe(c_sharpe, width=10)}")
        L.append(f"  Max Drawdown     : {c_dd:>10.2f}%")
        L.append("")

    # ---- TACTICAL LAYER ----
    L.append("=" * 74)
    L.append("TACTICAL LAYER (Mean-Reversion Sleeve)")
    L.append("=" * 74)
    tr_ret = _pct(tact_eq[0], tact_eq[-1])
    tr_sharpe = _sharpe(tact_eq)
    tr_dd = _maxdd(tact_eq)
    L.append(f"  Starting Capital : ${tact_eq[0]:>12,.2f}")
    L.append(f"  Ending Capital   : ${tact_eq[-1]:>12,.2f}")
    L.append(f"  Total Return     : {tr_ret:>+10.2f}%")
    L.append(f"  Sharpe Ratio     : {_format_sharpe(tr_sharpe, width=10)}")
    L.append(f"  Max Drawdown     : {tr_dd:>10.2f}%")
    L.append(f"  Fills            : {tactical_perf.get('total_fills', 0)}")
    L.append(f"  Profit Factor    : {tactical_perf.get('profit_factor', 'N/A')}")
    L.append(f"  vs Cash Proxy    : {tr_ret - cash_bench_ret:>+10.2f}% "
             f"(cash={cash_bench_ret:+.2f}%)")
    L.append("")

    # ---- STATIC 80/20 BASELINE ----
    L.append("=" * 74)
    if is_tactical_only:
        L.append("STATIC BASELINE (Cash Proxy)")
    else:
        L.append(f"STATIC BASELINE ({cfg.core_allocation_pct*100:.0f}/{cfg.tactical_allocation_pct*100:.0f} "
                 f"Buy-Hold / Cash)")
    L.append("=" * 74)
    b_ret = _pct(base_eq[0], base_eq[-1])
    b_sharpe = _sharpe(base_eq)
    b_dd = _maxdd(base_eq)
    L.append(f"  Starting Capital : ${base_eq[0]:>12,.2f}")
    L.append(f"  Ending Capital   : ${base_eq[-1]:>12,.2f}")
    L.append(f"  Total Return     : {b_ret:>+10.2f}%")
    L.append(f"  Sharpe Ratio     : {_format_sharpe(b_sharpe, width=10)}")
    L.append(f"  Max Drawdown     : {b_dd:>10.2f}%")
    L.append(f"  Note: No rebalancing; constant initial dollar split.")
    L.append("")

    # ---- MATCHED-CORE-ENTRY BASELINE (v3.4) ----
    matched_eq = tl_result.get("baseline_matched_core_entry", [])
    if matched_eq and len(matched_eq) == n:
        m_ret = _pct(matched_eq[0], matched_eq[-1])
        m_sharpe = _sharpe(matched_eq)
        m_dd = _maxdd(matched_eq)
        L.append("=" * 74)
        entry_tag = tl_result.get("core_entry_mode", cfg.core_entry_mode)
        L.append(f"MATCHED-ENTRY BASELINE (core={entry_tag}, tactical=cash)")
        L.append("=" * 74)
        L.append(f"  Starting Capital : ${matched_eq[0]:>12,.2f}")
        L.append(f"  Ending Capital   : ${matched_eq[-1]:>12,.2f}")
        L.append(f"  Total Return     : {m_ret:>+10.2f}%")
        L.append(f"  Sharpe Ratio     : {_format_sharpe(m_sharpe, width=10)}")
        L.append(f"  Max Drawdown     : {m_dd:>10.2f}%")
        L.append(f"  Note: Same core entry mode as strategy; tactical = cash.")
        L.append("")

    # ---- SIDE-BY-SIDE ----
    L.append("=" * 74)
    L.append("SIDE-BY-SIDE COMPARISON")
    L.append("=" * 74)
    if is_tactical_only:
        L.append(f"  {'Metric':<22} {'Total':>12} {'Baseline':>12} {'Tactical':>12}")
        L.append("  " + "-" * 48)
        L.append(f"  {'Return %':<22} {t_ret:>+11.2f}% {b_ret:>+11.2f}% {tr_ret:>+11.2f}%")
        L.append(f"  {'Sharpe':<22} {_format_sharpe(t_sharpe, width=12)} "
                 f"{_format_sharpe(b_sharpe, width=12)} {_format_sharpe(tr_sharpe, width=12)}")
        L.append(f"  {'Max DD %':<22} {t_dd:>12.2f} {b_dd:>12.2f} {tr_dd:>12.2f}")
    else:
        L.append(f"  {'Metric':<22} {'Total':>12} {'Baseline':>12} {'Tactical':>12} {'Core':>12}")
        L.append("  " + "-" * 60)
        L.append(f"  {'Return %':<22} {t_ret:>+11.2f}% {b_ret:>+11.2f}% {tr_ret:>+11.2f}% {c_ret:>+11.2f}%")
        L.append(f"  {'Sharpe':<22} {_format_sharpe(t_sharpe, width=12)} {_format_sharpe(b_sharpe, width=12)} "
                 f"{_format_sharpe(tr_sharpe, width=12)} {_format_sharpe(c_sharpe, width=12)}")
        L.append(f"  {'Max DD %':<22} {t_dd:>12.2f} {b_dd:>12.2f} {tr_dd:>12.2f} {c_dd:>12.2f}")
    L.append("")

    # ---- PORTFOLIO DIAGNOSTICS (new in v3.3) ----
    L.append("=" * 74)
    L.append("PORTFOLIO DIAGNOSTICS")
    L.append("=" * 74)

    # Buy-and-hold benchmark curve for capture ratios
    bh_curve = compute_buy_hold_curve(df, cfg.starting_capital).tolist()
    bh_ret = pd.Series(bh_curve, dtype=float).pct_change().dropna()
    benchmark_is_cash_proxy = len(bh_ret) > 1 and float(bh_ret.std()) < SHARPE_STD_EPS
    avg_exposure_frac = float(tact_diag.get("avg_exposure_pct", 0.0)) / 100.0
    capture_na_for_tactical_or_low_exposure = is_tactical_only or avg_exposure_frac < 0.02
    capture_na = capture_na_for_tactical_or_low_exposure or benchmark_is_cash_proxy
    cap = {"upside_capture": None, "downside_capture": None, "capture_ratio": None}
    if not capture_na:
        cap = _capture_ratios(total_eq, bh_curve)

    calmar = _calmar_ratio(total_eq)

    L.append(f"  Upside Capture   : {cap['upside_capture']:.1f}%"
             if cap["upside_capture"] is not None else
             "  Upside Capture   : N/A")
    L.append(f"  Downside Capture : {cap['downside_capture']:.1f}%"
             if cap["downside_capture"] is not None else
             "  Downside Capture : N/A")
    L.append(f"  Capture Ratio    : {cap['capture_ratio']:.2f}"
             if cap["capture_ratio"] is not None else
             "  Capture Ratio    : N/A")
    if capture_na_for_tactical_or_low_exposure:
        L.append("  Capture ratios N/A for tactical-only or low-exposure runs.")
    elif benchmark_is_cash_proxy:
        L.append("  Capture ratios N/A when benchmark is a cash proxy.")
    L.append(f"  Calmar Ratio     : {calmar:.3f}" if calmar else "  Calmar Ratio     : N/A")
    L.append("")

    # ---- RETURN ATTRIBUTION ----
    L.append("-" * 74)
    L.append("RETURN ATTRIBUTION")
    L.append("-" * 74)
    attr = _return_attribution(total_eq, core_eq, tact_eq, cfg.starting_capital)
    if is_tactical_only:
        L.append("  Core layer       : Disabled (0% allocation)")
        L.append(f"  Tactical sleeve  : {attr['tactical_pct']:>+7.1f}%  "
                 f"(${attr['tactical_dollars']:>+12,.2f})")
        L.append("  Note: Tactical-only run; no passive core beta sleeve.")
    else:
        L.append(f"  Core (beta)      : {attr['core_pct']:>+7.1f}%  (${attr['core_dollars']:>+12,.2f})")
        L.append(f"  Tactical (alpha) : {attr['tactical_pct']:>+7.1f}%  (${attr['tactical_dollars']:>+12,.2f})")
        L.append(f"  Note: Tactical layer is a defensive overlay — designed to")
        L.append(f"        outperform cash, not replace equity beta.")
    L.append("")

    # ---- TACTICAL DIAGNOSTICS ----
    L.append("-" * 74)
    L.append("TACTICAL LAYER DIAGNOSTICS")
    L.append("-" * 74)
    L.append(f"  Time in Market   : {tact_diag['time_in_market_pct']:.1f}%")
    L.append(f"  Avg Exposure     : {tact_diag['avg_exposure_pct']:.1f}%")
    L.append(f"  Blocked Rate     : {tact_diag['blocked_signal_rate_pct']:.1f}%")
    L.append(f"  blocked_by_confidence : {int(tact_diag.get('blocked_by_confidence', 0))}")
    L.append(f"  blocked_by_regime : {int(tact_diag.get('blocked_by_regime', 0))}")
    L.append(f"  blocked_by_cost : {int(tact_diag.get('blocked_by_cost', 0))}")
    L.append(f"  regime_no_evidence_bars : {int(tact_diag.get('regime_no_evidence_bars', 0))}")
    L.append(f"  Return vs Cash   : {tr_ret - cash_bench_ret:>+.2f}% "
             f"(cash benchmark = {cash_bench_ret:+.2f}%)")
    if tact_diag["accrued_cash_yield"] > 0:
        L.append(f"  Accrued Yield    : ${tact_diag['accrued_cash_yield']:>12,.2f}")
    L.append("")

    # ---- TACTICAL DETAILS (regime, fills, exits — reuse existing diagnostics) ----
    L.append("=" * 74)
    L.append("TACTICAL LAYER DETAILS")
    L.append("=" * 74)

    # Stat tests
    L.append("-" * 74)
    L.append("STATISTICAL TESTS")
    L.append("-" * 74)
    h = stat_metrics.get("hurst_exponent")
    if h is not None:
        L.append(f"  Hurst : {h:.3f}  R2={stat_metrics.get('hurst_r_squared', 'N/A')}")
    else:
        L.append("  Hurst : N/A")
    p = stat_metrics.get("adf_p_value")
    if p is not None:
        sl = "STATIONARY" if stat_metrics.get("is_stationary") else "NON-STATIONARY"
        L.append(f"  ADF   : p={p:.4f} ({sl})")
    else:
        L.append(f"  ADF   : N/A [{stat_metrics.get('adf_note', '')}]")
    L.append(f"  Verdict: {stat_metrics.get('verdict', 'N/A')}")
    L.append("")

    # Action summary
    ac = {}
    for a in bt_results.get("actions_log", []):
        ac[a["action"]] = ac.get(a["action"], 0) + 1
    L.append("-" * 74)
    L.append("LONG-ONLY ACTIONS SUMMARY (Tactical)")
    L.append("-" * 74)
    for act_name in ["BUY", "ADD", "HOLD", "REDUCE", "SELL", "BLOCKED"]:
        L.append(f"  {act_name:<12}: {ac.get(act_name, 0):>6}")
    L.append("")

    # Confidence buckets
    _append_confidence_bucket_table(L, bt_results)

    # Regime timeline
    rt = _regime_timeline(bt_results.get("actions_log", []))
    total_bars = max(sum(rt.values()), 1)
    L.append("-" * 74)
    L.append("REGIME TIMELINE")
    L.append("-" * 74)
    for r_name in [Regime.MEAN_REVERTING.value, Regime.SIDEWAYS.value,
                   Regime.AMBIGUOUS.value, Regime.TRENDING.value]:
        cnt = rt.get(r_name, 0)
        L.append(f"  {r_name:<20}: {cnt:>5} bars ({cnt/total_bars*100:.1f}%)")
    no_evidence_bars = sum(
        1 for a in bt_results.get("actions_log", [])
        if str(a.get("regime", "")) == "NO_EVIDENCE"
    )
    if no_evidence_bars > 0:
        L.append(f"  {'NO_EVIDENCE':<20}: {no_evidence_bars:>5} bars")
    L.append("")

    # Exit reason breakdown
    exit_bd = _exit_reason_breakdown(bt_results.get("fills", []))
    if exit_bd:
        L.append("-" * 74)
        L.append("EXIT REASON BREAKDOWN")
        L.append("-" * 74)
        L.append(f"  {'Reason':<22} {'Count':>5} {'AvgPnL':>10} {'AvgDays':>8} {'Wins':>5} {'Loss':>5}")
        for reason, d in sorted(exit_bd.items()):
            L.append(f"  {reason:<22} {d['count']:>5} "
                     f"${d['avg_pnl']:>+8.2f} {d['avg_days']:>7.1f} "
                     f"{d['wins']:>5} {d['losses']:>5}")
        L.append("")

    # Fill log
    fills_list = bt_results.get("fills", [])
    if fills_list:
        L.append("-" * 74)
        L.append("FILL LOG (Tactical)")
        L.append("-" * 74)
        L.append(f"  {'#':>3}  {'Date':>10}  {'Action':<8}  {'Price':>8}  "
                 f"{'Shares':>6}  {'P&L':>10}  {'Reason'}")
        L.append("  " + "-" * 68)
        for idx, f in enumerate(fills_list[:30], 1):
            pnl_str = f"${f.get('realized_pnl', 0):>+9.2f}" if "realized_pnl" in f else "         "
            reason = f.get("reason", f.get("regime", ""))
            L.append(
                f"  {idx:>3}  {f['date']:>10}  {f['action']:<8}  "
                f"${f['price']:>7.2f}  {f['shares']:>6}  {pnl_str}  {reason}"
            )
        if len(fills_list) > 30:
            L.append(f"  ... and {len(fills_list) - 30} more fills")
    L.append("")

    _append_bias_audit_section(L, bias_audit, survivorship_sensitivity)
    L.append("=" * 74)
    L.append("DISCLAIMER: Educational only. NOT financial advice.")
    L.append("Ratio framework is a configurable placeholder; it does NOT replicate")
    L.append("any proprietary valuation system. Past performance != future results.")
    L.append("=" * 74)
    return "\n".join(L)


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def _run_backtest(
    df: pd.DataFrame, ratio_z: pd.Series,
    regime_labels: pd.Series, regime_scores: pd.Series,
    cfg: MeanReversionConfig,
) -> Tuple[Dict, Dict]:
    engine = BacktestEngine(cfg)
    bt = engine.run(df, ratio_z, regime_labels, regime_scores)
    perf = calculate_performance_metrics(
        bt["equity_curve"], bt["fills"],
        cash_yield_annual_pct=cfg.cash_yield_annual_pct,
    )
    if len(df) >= 2:
        bh = (float(df["close"].iloc[-1]) / float(df["close"].iloc[0]) - 1) * 100
        perf["buy_and_hold_return_pct"] = round(bh, 2)
    return bt, perf


def _run_portfolio_tactical_backtest(
    data_by_ticker: Dict[str, pd.DataFrame],
    cfg: MeanReversionConfig,
) -> Tuple[Dict[str, Any], Dict[str, Any], pd.DatetimeIndex]:
    """Feature E: shared-cash tactical portfolio over a ticker universe."""
    tickers = sorted([str(t).upper() for t in data_by_ticker.keys()])
    if not tickers:
        bt_empty = {
            "equity_curve": [],
            "fills": [],
            "actions_log": [],
            "blocked_count": 0,
            "blocked_by_confidence": 0,
            "blocked_by_regime": 0,
            "blocked_by_cost": 0,
            "regime_no_evidence_bars": 0,
            "exposure_pct_curve": [],
            "accrued_cash_yield": 0.0,
            "confidence_bucket_summary": {"total_trades": 0, "rows": []},
            "trade_records": [],
            "portfolio_mode": True,
            "portfolio_tickers": [],
        }
        perf_empty = calculate_performance_metrics([], [], cash_yield_annual_pct=cfg.cash_yield_annual_pct)
        return bt_empty, perf_empty, pd.DatetimeIndex([])

    common_index = pd.DatetimeIndex(data_by_ticker[tickers[0]].index)
    for t in tickers[1:]:
        common_index = common_index.intersection(pd.DatetimeIndex(data_by_ticker[t].index))
    common_index = common_index.sort_values()

    aligned: Dict[str, pd.DataFrame] = {
        t: data_by_ticker[t].loc[common_index].copy() for t in tickers
    }
    n = len(common_index)
    tactical_entry_cfg = cfg.copy_with(entry_z=float(cfg.tactical_entry_z))
    engine = BacktestEngine(cfg)

    ratio_z_by_ticker: Dict[str, pd.Series] = {}
    regime_by_ticker: Dict[str, pd.Series] = {}
    ratio_by_ticker: Dict[str, pd.Series] = {}
    ratio_mu_by_ticker: Dict[str, pd.Series] = {}
    atr_by_ticker: Dict[str, pd.Series] = {}
    sma200_by_ticker: Dict[str, pd.Series] = {}
    sma200_slope_by_ticker: Dict[str, pd.Series] = {}
    sigma_annual_by_ticker: Dict[str, pd.Series] = {}
    atr_pct_by_ticker: Dict[str, pd.Series] = {}
    atr_pct_med_by_ticker: Dict[str, pd.Series] = {}

    for t in tickers:
        df_t = aligned[t]
        strategy_t = MeanReversionStrategy(cfg)
        ratio_t = pd.Series(strategy_t.compute_ratio(df_t), index=df_t.index, dtype=float)
        ratio_z_t = strategy_t.compute_ratio_z(ratio_t)
        regime_t, _ = strategy_t.classify_regime(df_t)
        ratio_z_by_ticker[t] = ratio_z_t
        regime_by_ticker[t] = regime_t
        ratio_by_ticker[t] = ratio_t
        ratio_mu_by_ticker[t] = ratio_t.rolling(int(cfg.ratio_lookback)).mean()
        atr_t = pd.Series(calculate_atr(df_t, 14), index=df_t.index, dtype=float)
        atr_by_ticker[t] = atr_t
        close_t = pd.Series(df_t["close"], index=df_t.index, dtype=float)
        sma200_t = pd.Series(close_t.rolling(200).mean(), index=df_t.index, dtype=float)
        sma200_by_ticker[t] = sma200_t
        sma200_slope_by_ticker[t] = pd.Series(sma200_t.diff(1), index=df_t.index, dtype=float)
        atr_pct_t = pd.Series((atr_t / close_t) * 100.0, index=df_t.index, dtype=float)
        atr_pct_by_ticker[t] = atr_pct_t
        atr_pct_med_by_ticker[t] = pd.Series(
            atr_pct_t.rolling(60).median(), index=df_t.index, dtype=float,
        )
        log_ret_t = pd.Series(
            np.log(close_t / close_t.shift(1)),
            index=df_t.index,
            dtype=float,
        )
        sigma_daily_t = log_ret_t.rolling(int(cfg.tactical_vol_window)).std()
        sigma_annual_by_ticker[t] = np.sqrt(252.0) * sigma_daily_t

    # Spec v1: pre-compute 252-bar sigma_annual and half_life per ticker
    sigma_annual_252_by_ticker: Dict[str, pd.Series] = {}
    half_life_by_ticker: Dict[str, Optional[float]] = {}
    if cfg.enable_spec_v1_upgrades:
        for t in tickers:
            df_t = aligned[t]
            # Spec v1: sigma_annual = np.log(df['close']).diff().rolling(252).std() * sqrt(252)
            log_close_t = pd.Series(np.log(df_t["close"].values), index=df_t.index)
            sigma_annual_252_by_ticker[t] = (
                log_close_t.diff().rolling(window=252).std() * np.sqrt(252)
            )
            half_life_by_ticker[t] = calculate_half_life(df_t["close"])

    positions: Dict[str, _PositionState] = {t: _PositionState() for t in tickers}
    open_meta: Dict[str, Dict[str, Any]] = {}

    cash = float(cfg.starting_capital)
    daily_cash_rate = _daily_cash_yield_rate(cfg.cash_yield_annual_pct)
    accrued_cash_yield = 0.0
    blocked_count = 0
    blocked_by_regime = 0
    blocked_by_confidence = 0
    blocked_by_cost = 0
    regime_no_evidence_bars = 0
    fills: List[Dict[str, Any]] = []
    actions_log: List[Dict[str, Any]] = []
    trade_records: List[Dict[str, Any]] = []
    equity_curve: List[float] = []
    exposure_pct_curve: List[float] = []
    cash_curve: List[float] = []
    invested_value_curve: List[float] = []

    def _mark_equity(i: int) -> float:
        mv = 0.0
        for tk, pos in positions.items():
            if pos.is_open:
                px = float(aligned[tk].iloc[i]["close"])
                mv += pos.shares * px
        return float(cash + mv)

    def _apply_cash_yield_eod() -> None:
        nonlocal cash, accrued_cash_yield
        if daily_cash_rate > 0 and cash > 0:
            interest = cash * daily_cash_rate
            cash += interest
            accrued_cash_yield += interest

    def _record_trade_close(
        ticker: str,
        close_bar_idx: int,
        final_fill_realized_pnl: float,
        close_bar_date: str,
        exit_price: Optional[float],
        exit_reason: str,
        exit_ratio_z: Optional[float],
    ) -> None:
        meta = open_meta.get(ticker, {})
        entry_idx = safe_number(meta.get("entry_bar_index"), 0) or 0
        hold_days = int(max(int(close_bar_idx) - int(entry_idx), 0))
        conf = safe_number(meta.get("entry_confidence"), None)
        conf_bucket = "NA"
        if conf is not None:
            if conf < 0.60:
                conf_bucket = "<0.60"
            elif conf < 0.70:
                conf_bucket = "0.60-0.70"
            elif conf < 0.80:
                conf_bucket = "0.70-0.80"
            elif conf < 0.90:
                conf_bucket = "0.80-0.90"
            else:
                conf_bucket = "0.90-1.00"
        trade_records.append({
            "ticker": ticker,
            "conf": conf,
            "entry_confidence": conf,
            "entry_conf_bucket": conf_bucket,
            "bucket": conf_bucket,
            "pnl": round(float(final_fill_realized_pnl), 2),
            "realized_pnl": round(float(final_fill_realized_pnl), 2),
            "hold_days": hold_days,
            "entry_regime": meta.get("entry_regime"),
            "entry_regime_label": meta.get("entry_regime"),
            "entry_ratio_z": safe_number(meta.get("entry_ratio_z"), None),
            "exit_ratio_z": safe_number(exit_ratio_z, None),
            "entry_bar_index": int(entry_idx),
            "exit_bar_index": int(close_bar_idx),
            "entry_dt": meta.get("entry_dt"),
            "exit_dt": close_bar_date,
            "entry_price": safe_number(meta.get("entry_price"), None),
            "exit_price": safe_number(exit_price, None),
            "shares": int(safe_number(meta.get("shares"), 0) or 0),
            "exit_reason": exit_reason or "EXIT",
        })
        if ticker in open_meta:
            del open_meta[ticker]

    if cfg.thesis_break_min_slope == 0.0:
        min_slope_thr_by_ticker: Dict[str, float] = {}
        for t in tickers:
            slopes = sma200_slope_by_ticker[t].dropna()
            if len(slopes) > 50:
                min_slope_thr_by_ticker[t] = float(slopes.abs().quantile(0.25))
            else:
                min_slope_thr_by_ticker[t] = 0.0
    else:
        min_slope_thr_by_ticker = {t: float(cfg.thesis_break_min_slope) for t in tickers}

    for i in range(n):
        bar_dt = common_index[i].strftime("%Y-%m-%d")
        # --- Exits first ---
        for t in tickers:
            pos = positions[t]
            if not pos.is_open:
                continue
            df_t = aligned[t]
            bar = df_t.iloc[i]
            rz = ratio_z_by_ticker[t].iloc[i] if i < len(ratio_z_by_ticker[t]) else np.nan
            regime = regime_by_ticker[t].iloc[i] if i < len(regime_by_ticker[t]) else Regime.AMBIGUOUS.value
            if str(regime) == "NO_EVIDENCE":
                regime_no_evidence_bars += 1
            bars_held = i - pos.entry_bar

            # Spec v1: Mean-Reversion Exit (downside worsening trigger)
            if cfg.enable_spec_v1_upgrades and not pd.isna(rz):
                should_exit_mr = (
                    float(rz) <= float(cfg.tactical_exit_z)
                    and bars_held >= int(cfg.tactical_min_hold_days)
                )
                if should_exit_mr:
                    exit_px = engine._fill_price(float(bar["close"]), sell=True)
                    cash, fill = engine._sell_all(
                        pos, i, bar_dt, exit_px, "REVERTED_TO_MEAN", cash,
                    )
                    fill["ticker"] = t
                    fills.append(fill)
                    actions_log.append({
                        "ticker": t, "bar": i, "date": bar_dt, "action": "SELL",
                        "reason": "REVERTED_TO_MEAN", "regime": regime,
                    })
                    _record_trade_close(
                        t, i, float(fill.get("realized_pnl", 0.0)),
                        close_bar_date=str(fill.get("date", bar_dt)),
                        exit_price=safe_number(fill.get("price"), None),
                        exit_reason="REVERTED_TO_MEAN",
                        exit_ratio_z=float(rz),
                    )
                    continue

            if float(bar["low"]) <= float(pos.stop_price):
                is_gap_stop = float(bar["open"]) < float(pos.stop_price)
                exec_base = _stop_execution_base(
                    open_px=float(bar["open"]),
                    high_px=float(bar["high"]),
                    low_px=float(bar["low"]),
                    stop_px=float(pos.stop_price),
                )
                exit_px = engine._fill_price(exec_base, sell=True)
                sell_reason = "SELL_GAP_STOP_OPEN" if is_gap_stop else "SELL_STOP"
                action_reason = "GAP_STOP_OPEN" if is_gap_stop else "STOP_LOSS"
                cash, fill = engine._sell_all(pos, i, bar_dt, exit_px, sell_reason, cash)
                fill["ticker"] = t
                fills.append(fill)
                actions_log.append({
                    "ticker": t, "bar": i, "date": bar_dt, "action": "SELL",
                    "reason": action_reason, "regime": regime,
                })
                _record_trade_close(
                    t, i, float(fill.get("realized_pnl", 0.0)),
                    close_bar_date=str(fill.get("date", bar_dt)),
                    exit_price=safe_number(fill.get("price"), None),
                    exit_reason=str(fill.get("reason", sell_reason)),
                    exit_ratio_z=(None if pd.isna(rz) else float(rz)),
                )
                continue

            slope_series = sma200_slope_by_ticker[t]
            if i < len(slope_series) and not pd.isna(slope_series.iloc[i]):
                slope_val = float(slope_series.iloc[i])
                if slope_val < -float(min_slope_thr_by_ticker.get(t, 0.0)):
                    pos.neg_slope_count += 1
                else:
                    pos.neg_slope_count = 0
                if pos.neg_slope_count >= int(cfg.thesis_break_sma_bars):
                    sma_val = sma200_by_ticker[t].iloc[i] if i < len(sma200_by_ticker[t]) else np.nan
                    price_now = float(bar["close"])
                    price_below_sma = (not pd.isna(sma_val) and price_now < float(sma_val))
                    regime_trending = (regime == Regime.TRENDING.value)
                    if (not cfg.thesis_break_require_below_sma) or price_below_sma or regime_trending:
                        exit_px = engine._fill_price(price_now, sell=True)
                        cash, fill = engine._sell_all(pos, i, bar_dt, exit_px, "SELL_THESIS_BREAK", cash)
                        fill["ticker"] = t
                        fills.append(fill)
                        actions_log.append({
                            "ticker": t, "bar": i, "date": bar_dt, "action": "SELL",
                            "reason": "THESIS_BREAK", "regime": regime,
                        })
                        _record_trade_close(
                            t, i, float(fill.get("realized_pnl", 0.0)),
                            close_bar_date=str(fill.get("date", bar_dt)),
                            exit_price=safe_number(fill.get("price"), None),
                            exit_reason=str(fill.get("reason", "SELL_THESIS_BREAK")),
                            exit_ratio_z=(None if pd.isna(rz) else float(rz)),
                        )
                        continue

            if cfg.better_exits_enabled:
                if (
                    bars_held >= int(cfg.tactical_min_hold_days)
                    and not pd.isna(rz)
                    and float(rz) >= float(cfg.tactical_exit_z)
                    and i + 1 < n
                ):
                    exit_dt = common_index[i + 1].strftime("%Y-%m-%d")
                    exit_px = engine._fill_price(float(df_t.iloc[i + 1]["open"]), sell=True)
                    cash, fill = engine._sell_all(pos, i + 1, exit_dt, exit_px, "SELL_RETURN_TO_MEAN", cash)
                    fill["ticker"] = t
                    fills.append(fill)
                    actions_log.append({
                        "ticker": t, "bar": i, "date": bar_dt, "action": "SELL",
                        "reason": "RETURN_TO_MEAN_NEXT_OPEN", "regime": regime,
                    })
                    _record_trade_close(
                        t, i + 1, float(fill.get("realized_pnl", 0.0)),
                        close_bar_date=str(fill.get("date", exit_dt)),
                        exit_price=safe_number(fill.get("price"), None),
                        exit_reason=str(fill.get("reason", "SELL_RETURN_TO_MEAN")),
                        exit_ratio_z=(None if pd.isna(rz) else float(rz)),
                    )
                    continue
                if bars_held >= int(cfg.tactical_max_hold_days) and i + 1 < n:
                    exit_dt = common_index[i + 1].strftime("%Y-%m-%d")
                    exit_px = engine._fill_price(float(df_t.iloc[i + 1]["open"]), sell=True)
                    cash, fill = engine._sell_all(pos, i + 1, exit_dt, exit_px, "SELL_TIME_LIMIT", cash)
                    fill["ticker"] = t
                    fills.append(fill)
                    actions_log.append({
                        "ticker": t, "bar": i, "date": bar_dt, "action": "SELL",
                        "reason": "TIME_LIMIT_NEXT_OPEN", "regime": regime,
                    })
                    _record_trade_close(
                        t, i + 1, float(fill.get("realized_pnl", 0.0)),
                        close_bar_date=str(fill.get("date", exit_dt)),
                        exit_price=safe_number(fill.get("price"), None),
                        exit_reason=str(fill.get("reason", "SELL_TIME_LIMIT")),
                        exit_ratio_z=(None if pd.isna(rz) else float(rz)),
                    )
                    continue
            else:
                max_hold = int(cfg.max_holding_days)
                if regime == Regime.AMBIGUOUS.value:
                    max_hold = int(max_hold * 0.6)
                if bars_held >= max_hold:
                    exit_px = engine._fill_price(float(bar["close"]), sell=True)
                    cash, fill = engine._sell_all(pos, i, bar_dt, exit_px, "SELL_TIME_LIMIT", cash)
                    fill["ticker"] = t
                    fills.append(fill)
                    actions_log.append({
                        "ticker": t, "bar": i, "date": bar_dt, "action": "SELL",
                        "reason": "TIME_LIMIT", "regime": regime,
                    })
                    _record_trade_close(
                        t, i, float(fill.get("realized_pnl", 0.0)),
                        close_bar_date=str(fill.get("date", bar_dt)),
                        exit_price=safe_number(fill.get("price"), None),
                        exit_reason=str(fill.get("reason", "SELL_TIME_LIMIT")),
                        exit_ratio_z=(None if pd.isna(rz) else float(rz)),
                    )
                    continue

        # --- Entries (cross-sectional ranking) ---
        if not _should_block_next_open_entry(i, n, cfg.entry_at):
            open_count = sum(1 for pos in positions.values() if pos.is_open)
            slots = max(int(cfg.tactical_max_positions) - open_count, 0)
            candidates: List[Tuple[float, str, float, str, Optional[float]]] = []
            for t in tickers:
                pos = positions[t]
                if pos.is_open:
                    continue
                df_t = aligned[t]
                rz = ratio_z_by_ticker[t].iloc[i] if i < len(ratio_z_by_ticker[t]) else np.nan
                if pd.isna(rz):
                    continue
                regime = regime_by_ticker[t].iloc[i] if i < len(regime_by_ticker[t]) else Regime.AMBIGUOUS.value
                if str(regime) == "NO_EVIDENCE":
                    regime_no_evidence_bars += 1
                reversal_ok = check_reversal_confirmation(df_t, i, tactical_entry_cfg)
                quality_ok = passes_quality_filter(df_t, i, tactical_entry_cfg)
                action = determine_action(
                    float(rz), str(regime), False, 0.0, tactical_entry_cfg, reversal_ok, quality_ok,
                )
                if action == Action.BLOCKED:
                    blocked_count += 1
                    blocked_by_regime += 1
                    actions_log.append({
                        "ticker": t, "bar": i, "date": bar_dt, "action": "BLOCKED",
                        "reason": f"regime={regime}", "regime": regime,
                    })
                    continue
                if action != Action.BUY:
                    continue
                entry_conf = _entry_confidence_from_labels(i, regime_by_ticker[t], N=60)
                if cfg.min_confidence > 0.0 and entry_conf is not None and entry_conf < cfg.min_confidence:
                    blocked_count += 1
                    blocked_by_confidence += 1
                    actions_log.append({
                        "ticker": t, "bar": i, "date": bar_dt, "action": "BLOCKED",
                        "reason": f"confidence={entry_conf:.2f}<min={cfg.min_confidence:.2f}",
                        "regime": regime,
                    })
                    continue
                if cfg.enable_spec_v1_upgrades:
                    # Spec v1: cost gate using half-life ERet
                    r_now = ratio_by_ticker[t].iloc[i] if i < len(ratio_by_ticker[t]) else np.nan
                    r_mu = ratio_mu_by_ticker[t].iloc[i] if i < len(ratio_mu_by_ticker[t]) else np.nan
                    hl_t = half_life_by_ticker.get(t)
                    eret = _expected_return_spec_v1(
                        safe_number(r_now, None), safe_number(r_mu, None), hl_t,
                    )
                    est_notional = max(float(equity_now) / max(int(cfg.tactical_max_positions), 1), 1.0) if 'equity_now' in dir() else max(float(cash), 1.0)
                    is_cost_effective = _is_cost_effective_spec_v1(
                        eret,
                        cost_k=float(cfg.cost_k),
                        slippage_bps=float(cfg.cost_bps_est),
                        commission=float(cfg.commission_per_trade),
                        notional=est_notional,
                    )
                    if not is_cost_effective:
                        blocked_count += 1
                        blocked_by_cost += 1
                        actions_log.append({
                            "ticker": t, "bar": i, "date": bar_dt, "action": "BLOCKED",
                            "reason": f"COST_AWARE_FILTER eret={eret:.4f}",
                            "regime": regime,
                        })
                        continue
                elif cfg.cost_aware_entry_enabled:
                    r_now = ratio_by_ticker[t].iloc[i] if i < len(ratio_by_ticker[t]) else np.nan
                    r_mu = ratio_mu_by_ticker[t].iloc[i] if i < len(ratio_mu_by_ticker[t]) else np.nan
                    eret = _expected_return_proxy_from_ratio(
                        safe_number(r_now, None), safe_number(r_mu, None),
                    )
                    cost_frac = float(cfg.cost_bps_est) / 10000.0
                    min_req = float(cfg.cost_k) * cost_frac
                    if eret < min_req:
                        blocked_count += 1
                        blocked_by_cost += 1
                        actions_log.append({
                            "ticker": t, "bar": i, "date": bar_dt, "action": "BLOCKED",
                            "reason": f"cost_aware eret={eret:.4f}<min={min_req:.4f}",
                            "regime": regime,
                        })
                        continue
                candidates.append((float(rz), t, float(entry_conf) if entry_conf is not None else np.nan, str(regime), entry_conf))

            if slots > 0 and candidates:
                candidates.sort(key=lambda x: x[0])  # most oversold first
                selected = candidates[:slots]
                equity_now = _mark_equity(i)
                d_base = float(equity_now) / max(int(cfg.tactical_max_positions), 1)
                for rz, t, _, regime, entry_conf in selected:
                    df_t = aligned[t]
                    fill_px = engine._get_entry_price(df_t, i, cfg)
                    if fill_px is None:
                        actions_log.append({
                            "ticker": t, "bar": i, "date": bar_dt, "action": "HOLD",
                            "reason": "NO_FILL_BAR", "regime": regime,
                        })
                        continue
                    mvol = 1.0
                    mconf = 1.0
                    if cfg.enable_spec_v1_upgrades:
                        # Spec v1: m_vol from 252-bar log-ret vol
                        sig_252 = (
                            sigma_annual_252_by_ticker[t].iloc[i]
                            if t in sigma_annual_252_by_ticker and i < len(sigma_annual_252_by_ticker[t])
                            else np.nan
                        )
                        mvol = _vol_target_multiplier(
                            safe_number(sig_252, None),
                            sigma_target=float(cfg.tactical_vol_target),
                            sigma_floor=float(cfg.tactical_vol_floor),
                            cap=float(cfg.tactical_vol_cap),
                        )
                        # Spec v1: m_conf linear ramp (no gamma)
                        mconf = _confidence_sizing_multiplier(
                            entry_conf,
                            c0=float(cfg.confidence_c0),
                            gamma=1.0,  # Spec v1: no power/gamma
                        )
                    else:
                        if cfg.tactical_vol_targeting_enabled:
                            sig = sigma_annual_by_ticker[t].iloc[i] if i < len(sigma_annual_by_ticker[t]) else np.nan
                            mvol = _vol_target_multiplier(
                                safe_number(sig, None),
                                sigma_target=float(cfg.tactical_vol_target),
                                sigma_floor=float(cfg.tactical_vol_floor),
                                cap=float(cfg.tactical_vol_cap),
                            )
                        if cfg.vol_adjust_sizing:
                            atr_now = atr_pct_by_ticker[t].iloc[i] if i < len(atr_pct_by_ticker[t]) else np.nan
                            atr_med = atr_pct_med_by_ticker[t].iloc[i] if i < len(atr_pct_med_by_ticker[t]) else np.nan
                            atr_now_v = safe_number(atr_now, None)
                            atr_med_v = safe_number(atr_med, atr_now_v if atr_now_v is not None else None)
                            if atr_now_v is not None and atr_now_v > 0 and atr_med_v is not None:
                                v_adj = float(atr_med_v) / float(atr_now_v)
                                mvol *= max(float(cfg.vol_sizing_floor), min(float(cfg.vol_sizing_cap), v_adj))
                        if cfg.confidence_sizing_enabled:
                            mconf = _confidence_sizing_multiplier(
                                entry_conf,
                                c0=float(cfg.confidence_c0),
                                gamma=float(cfg.confidence_gamma),
                            )
                    d_final = d_base * mvol * mconf
                    shares = int(d_final // float(fill_px))
                    min_cash = float(equity_now) * float(cfg.min_cash_pct)
                    if shares * float(fill_px) + float(cfg.commission_per_trade) > cash - min_cash:
                        shares = int((cash - min_cash - float(cfg.commission_per_trade)) // float(fill_px))
                    if shares < int(cfg.min_shares) or shares * float(fill_px) < float(cfg.min_trade_notional):
                        actions_log.append({
                            "ticker": t, "bar": i, "date": bar_dt, "action": "HOLD",
                            "reason": "DUST_FILTER", "regime": regime,
                        })
                        continue
                    cost = shares * float(fill_px) + float(cfg.commission_per_trade)
                    if cost > cash:
                        continue
                    cash -= cost
                    pos = positions[t]
                    pos.shares += shares
                    pos.cost_basis += shares * float(fill_px)
                    pos.entry_bar = i
                    pos.last_add_bar = i
                    pos.neg_slope_count = 0
                    atr_val = atr_by_ticker[t].iloc[i] if i < len(atr_by_ticker[t]) else np.nan
                    atr_num_opt = safe_number(atr_val, None)
                    atr_num = float(atr_num_opt) if atr_num_opt is not None else float(fill_px) * 0.02
                    pos.stop_price = round(float(fill_px) - atr_num * float(cfg.stop_atr_multiple), 2)
                    entry_fill_date = (
                        common_index[i + 1].strftime("%Y-%m-%d")
                        if cfg.entry_at == "next_open" and i + 1 < n
                        else bar_dt
                    )
                    open_meta[t] = {
                        "entry_bar_index": i,
                        "entry_dt": entry_fill_date,
                        "entry_price": float(fill_px),
                        "shares": int(shares),
                        "entry_confidence": entry_conf,
                        "entry_regime": regime,
                        "entry_ratio_z": float(rz),
                    }
                    fills.append({
                        "ticker": t, "bar": i, "date": bar_dt, "action": "BUY",
                        "price": float(fill_px), "shares": int(shares),
                        "cost": round(float(cost), 2), "regime": regime,
                    })
                    actions_log.append({
                        "ticker": t, "bar": i, "date": bar_dt, "action": "BUY",
                        "reason": f"ratio_z={rz:.2f}", "regime": regime,
                    })

        _apply_cash_yield_eod()
        eq = _mark_equity(i)
        equity_curve.append(round(eq, 2))
        invested = 0.0
        for t, pos in positions.items():
            if pos.is_open:
                invested += pos.shares * float(aligned[t].iloc[i]["close"])
        invested_value_curve.append(round(float(invested), 2))
        cash_curve.append(round(float(cash), 2))
        exposure_pct_curve.append(round((invested / eq * 100.0) if eq > 0 else 0.0, 2))

    # Force close all open positions at end-of-data.
    if n > 0:
        last_dt = common_index[-1].strftime("%Y-%m-%d")
        for t, pos in positions.items():
            if not pos.is_open:
                continue
            last_px = engine._fill_price(float(aligned[t]["close"].iloc[-1]), sell=True)
            cash, fill = engine._sell_all(pos, n - 1, last_dt, last_px, "SELL_END_OF_DATA", cash)
            fill["ticker"] = t
            fills.append(fill)
            final_rz = ratio_z_by_ticker[t].iloc[n - 1] if (n - 1) < len(ratio_z_by_ticker[t]) else np.nan
            _record_trade_close(
                t, n - 1, float(fill.get("realized_pnl", 0.0)),
                close_bar_date=str(fill.get("date", last_dt)),
                exit_price=safe_number(fill.get("price"), None),
                exit_reason=str(fill.get("reason", "SELL_END_OF_DATA")),
                exit_ratio_z=(None if pd.isna(final_rz) else float(final_rz)),
            )
        if equity_curve:
            equity_curve[-1] = round(float(cash), 2)

    bucket_summary = _summarize_confidence_buckets(trade_records)
    bt = {
        "equity_curve": equity_curve,
        "fills": fills,
        "actions_log": actions_log,
        "blocked_count": int(blocked_count),
        "blocked_by_confidence": int(blocked_by_confidence),
        "blocked_by_regime": int(blocked_by_regime),
        "blocked_by_cost": int(blocked_by_cost),
        "regime_no_evidence_bars": int(regime_no_evidence_bars),
        "exposure_pct_curve": exposure_pct_curve,
        "cash_curve": cash_curve,
        "invested_value_curve": invested_value_curve,
        "accrued_cash_yield": round(float(accrued_cash_yield), 2),
        "confidence_bucket_summary": bucket_summary,
        "trade_records": trade_records,
        "portfolio_mode": True,
        "portfolio_tickers": tickers,
    }
    perf = calculate_performance_metrics(
        bt["equity_curve"], bt["fills"], cash_yield_annual_pct=cfg.cash_yield_annual_pct,
    )
    return bt, perf, common_index


def _summary_confidence_buckets(bt_results: Dict) -> List[Dict[str, Any]]:
    summary = bt_results.get("confidence_bucket_summary", {})
    rows = summary.get("rows", []) if isinstance(summary, dict) else []
    table: List[Dict[str, Any]] = []
    for row in rows:
        table.append({
            "bucket": row.get("bucket", "NA"),
            "trades": int(row.get("trades", 0)),
            "win_rate": float(row.get("win_rate_pct", 0.0)),
            "avg_pnl": float(row.get("avg_pnl", 0.0)),
            "avg_hold_days": float(row.get("avg_hold_days", 0.0)),
        })
    return table


def _compute_confidence_bins_0p02(
    trade_records: List[Dict[str, Any]],
    start: float = 0.50,
    end: float = 1.00,
) -> List[Dict[str, Any]]:
    """Fine confidence bins over completed trades, using existing realized PnL accounting."""
    records: List[Tuple[float, float, float]] = []
    for rec in trade_records:
        conf = safe_number(rec.get("entry_confidence", rec.get("conf")), None)
        pnl = safe_number(rec.get("realized_pnl", rec.get("pnl")), None)
        hold = safe_number(rec.get("hold_days"), 0.0)
        if conf is None or pnl is None:
            continue
        conf_f = float(conf)
        if conf_f < start or conf_f > end:
            continue
        pnl_f = float(pnl)
        hold_f = float(hold) if hold is not None else 0.0
        records.append((conf_f, pnl_f, hold_f))

    rows: List[Dict[str, Any]] = []
    lo_i_start = int(round(start * 100))
    end_i = int(round(end * 100))
    step_i = 2  # 0.02 bins
    for lo_i in range(lo_i_start, end_i, step_i):
        hi_i = min(lo_i + step_i, end_i)
        lo = lo_i / 100.0
        hi = hi_i / 100.0
        is_last = hi_i >= end_i
        bucket = [
            (c, p, h) for (c, p, h) in records
            if ((lo <= c <= hi) if is_last else (lo <= c < hi))
        ]
        pnls = [p for _, p, _ in bucket]
        holds = [h for _, _, h in bucket]
        n_trades = len(bucket)
        win_rate = (sum(1 for p in pnls if p > 0) / n_trades * 100.0) if n_trades > 0 else 0.0
        avg_pnl = (sum(pnls) / n_trades) if n_trades > 0 else 0.0
        median_pnl = (float(np.median(pnls)) if n_trades > 0 else 0.0)
        avg_hold = (sum(holds) / n_trades) if n_trades > 0 else 0.0
        bin_label = f"[{lo:.2f},{hi:.2f}]" if is_last else f"[{lo:.2f},{hi:.2f})"
        rows.append({
            "bin": bin_label,
            "bin_lo": round(lo, 2),
            "bin_hi": round(hi, 2),
            "n_trades": n_trades,
            "win_rate": round(float(win_rate), 2),
            "avg_pnl": round(float(avg_pnl), 2),
            "median_pnl": round(float(median_pnl), 2),
            "avg_hold_days": round(float(avg_hold), 2),
        })
    return rows


def _write_trade_ledger_csv(
    out_path: Path,
    ticker: str,
    trade_records: List[Dict[str, Any]],
    min_confidence_used: float,
) -> int:
    """Write one row per completed trade; returns rows written."""
    fieldnames = [
        "ticker",
        "entry_dt",
        "exit_dt",
        "hold_days",
        "entry_confidence",
        "realized_pnl",
        "entry_regime",
        "entry_ratio_z",
        "entry_z",
        "exit_z",
        "mae_pct",
        "mfe_pct",
        "mae_abs",
        "mfe_abs",
        "exit_reason",
        "min_confidence_used",
    ]
    n_rows = 0
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for rec in trade_records:
            entry_conf = safe_number(rec.get("entry_confidence", rec.get("conf")), None)
            realized = safe_number(rec.get("realized_pnl", rec.get("pnl")), 0.0)
            realized_f = float(realized) if realized is not None else 0.0
            entry_ratio_z = safe_number(rec.get("entry_ratio_z"), None)
            entry_z = safe_number(rec.get("entry_z", entry_ratio_z), None)
            exit_z = safe_number(rec.get("exit_z", rec.get("exit_ratio_z")), None)
            mae_pct = safe_number(rec.get("mae_pct"), None)
            mfe_pct = safe_number(rec.get("mfe_pct"), None)
            mae_abs = safe_number(rec.get("mae_abs"), None)
            mfe_abs = safe_number(rec.get("mfe_abs"), None)
            writer.writerow({
                "ticker": str(rec.get("ticker", ticker)),
                "entry_dt": rec.get("entry_dt", ""),
                "exit_dt": rec.get("exit_dt", ""),
                "hold_days": int(safe_number(rec.get("hold_days"), 0) or 0),
                "entry_confidence": "" if entry_conf is None else round(float(entry_conf), 4),
                "realized_pnl": round(realized_f, 2),
                "entry_regime": rec.get("entry_regime", ""),
                "entry_ratio_z": (
                    ""
                    if entry_ratio_z is None
                    else round(float(entry_ratio_z), 4)
                ),
                "entry_z": "" if entry_z is None else round(float(entry_z), 4),
                "exit_z": "" if exit_z is None else round(float(exit_z), 4),
                "mae_pct": "" if mae_pct is None else round(float(mae_pct), 4),
                "mfe_pct": "" if mfe_pct is None else round(float(mfe_pct), 4),
                "mae_abs": "" if mae_abs is None else round(float(mae_abs), 2),
                "mfe_abs": "" if mfe_abs is None else round(float(mfe_abs), 2),
                "exit_reason": rec.get("exit_reason", "EXIT"),
                "min_confidence_used": round(float(min_confidence_used), 4),
            })
            n_rows += 1
    return n_rows


def _format_min_confidence_tag(min_confidence: float) -> str:
    """Compact confidence tag for batch artifact names (e.g., mc0, mc060)."""
    conf = safe_number(min_confidence, 0.0)
    if conf is None or conf <= 0.0:
        return "mc0"
    scaled = int(round(float(conf) * 100))
    return f"mc{scaled:03d}"


def _build_summary_payload(
    ticker: str,
    df: pd.DataFrame,
    cfg: MeanReversionConfig,
    perf: Dict,
    bt_results: Dict,
    tl_result: Optional[Dict] = None,
    period: str = "500d",
    interval: str = "1d",
    price_field: str = "close",
    require_min_bars: int = 400,
    data_quality: Optional[Dict[str, Any]] = None,
    config_hash: str = "",
    survivorship_drag_ann: float = 0.0,
    bias_audit: Optional[Dict[str, Any]] = None,
    survivorship_sensitivity: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    tact_diag = _tactical_diagnostics(bt_results)
    trade_records = bt_results.get("trade_records", [])
    if not isinstance(trade_records, list):
        trade_records = []

    def _curve_return_pct(curve: Any) -> float:
        if not isinstance(curve, list) or len(curve) < 2:
            return 0.0
        start = safe_number(curve[0], None)
        end = safe_number(curve[-1], None)
        if start is None or end is None or start <= 0:
            return 0.0
        return (float(end) / float(start) - 1.0) * 100.0

    def _curve_max_dd_pct(curve: Any) -> float:
        if not isinstance(curve, list) or len(curve) < 2:
            return 0.0
        s = pd.Series(curve, dtype=float)
        if len(s) < 2 or s.iloc[0] <= 0:
            return 0.0
        cum = s / s.iloc[0]
        rm = cum.expanding().max()
        dd = (cum - rm) / rm
        return float(dd.min()) * 100.0

    def _rounded_or_none(value: Any, digits: int = 3) -> Optional[float]:
        v = safe_number(value, None)
        return None if v is None else round(float(v), digits)

    bars = int(len(df))
    cash_proxy_bars = max(bars - 1, 0)
    cash_bench_ret = _cash_proxy_compounded_return_pct(
        cfg.cash_yield_annual_pct, cash_proxy_bars,
    )
    if (
        tl_result is not None and cfg.two_layer_mode
        and cfg.core_allocation_pct <= 0.0
    ):
        base_eq = tl_result.get("baseline_equity_curve", [])
        if len(base_eq) > 1 and base_eq[0] > 0:
            cash_bench_ret = (float(base_eq[-1]) / float(base_eq[0]) - 1) * 100

    tactical_return = float(perf.get("total_return_pct", 0.0))
    tactical_vs_cash = tactical_return - float(cash_bench_ret)
    tactical_max_dd = float(perf.get("max_drawdown_pct", 0.0))
    tactical_fills = int(perf.get("total_fills", len(bt_results.get("fills", []))))
    dq = data_quality if isinstance(data_quality, dict) else {}
    bias = (
        bias_audit if isinstance(bias_audit, dict)
        else _build_bias_audit_payload(cfg, dq)
    )
    surv = (
        survivorship_sensitivity if isinstance(survivorship_sensitivity, dict)
        else _build_survivorship_sensitivity_payload(df, cfg, tl_result, survivorship_drag_ann)
    )
    tail_diag = _tail_diagnostics(trade_records)
    payload: Dict[str, Any] = {
        "ticker": ticker,
        "bars": bars,
        "bars_downloaded": bars,
        "period": str(period),
        "interval": str(interval),
        "price_field": str(price_field),
        "price_field_used": str(dq.get("price_field_used", price_field)),
        "price_field_warning": str(dq.get("price_field_warning", "")),
        "ohlc_scaled_by_adjclose": bool(dq.get("ohlc_scaled_by_adjclose", False)),
        "volume_inverse_scaled": bool(dq.get("volume_inverse_scaled", False)),
        "require_min_bars": int(require_min_bars),
        "config_hash": str(config_hash),
        "cash_yield": float(cfg.cash_yield_annual_pct),
        "universe_name": str(bias.get("universe_name", "")),
        "universe_asof": str(bias.get("universe_asof", "")),
        "universe_source": str(bias.get("universe_source", "")),
        "survivorship_drag_ann": float(safe_number(surv.get("survivorship_drag_ann"), 0.0) or 0.0),
        "heuristic_sensitivity_only": bool(surv.get("heuristic_sensitivity_only", True)),
        "core_weight": float(safe_number(surv.get("core_weight"), 0.0) or 0.0),
        "core_cagr": safe_number(surv.get("core_cagr"), None),
        "static_baseline_cagr": safe_number(surv.get("static_baseline_cagr"), None),
        "core_cagr_adj": safe_number(surv.get("core_cagr_adj"), None),
        "static_baseline_cagr_adj": safe_number(surv.get("static_baseline_cagr_adj"), None),
        "bias_audit": bias,
        "survivorship_sensitivity": surv,
        "core_pct": float(cfg.core_allocation_pct),
        "tactical_pct": float(cfg.tactical_allocation_pct),
        "rebalance_freq": cfg.rebalance_freq,
        "rebalance_drift_threshold": float(cfg.rebalance_drift_threshold),
        "tactical_vol_targeting_enabled": bool(cfg.tactical_vol_targeting_enabled),
        "tactical_vol_target": float(cfg.tactical_vol_target),
        "tactical_vol_window": int(cfg.tactical_vol_window),
        "tactical_vol_floor": float(cfg.tactical_vol_floor),
        "tactical_vol_cap": float(cfg.tactical_vol_cap),
        "cost_aware_entry_enabled": bool(cfg.cost_aware_entry_enabled),
        "cost_bps_est": float(cfg.cost_bps_est),
        "cost_k": float(cfg.cost_k),
        "better_exits_enabled": bool(cfg.better_exits_enabled),
        "tactical_exit_z": float(cfg.tactical_exit_z),
        "tactical_min_hold_days": int(cfg.tactical_min_hold_days),
        "tactical_max_hold_days": int(cfg.tactical_max_hold_days),
        "confidence_sizing_enabled": bool(cfg.confidence_sizing_enabled),
        "confidence_c0": float(cfg.confidence_c0),
        "confidence_gamma": float(cfg.confidence_gamma),
        "tactical_mode": str(cfg.tactical_mode),
        "tactical_max_positions": int(cfg.tactical_max_positions),
        "tactical_entry_z": float(cfg.tactical_entry_z),
        "tactical_weighting": str(cfg.tactical_weighting),
        "total_return_pct": round(tactical_return, 2),
        "vs_cash_pct": round(tactical_vs_cash, 2),
        "max_dd_pct": tactical_max_dd,
        "fills": tactical_fills,
        "time_in_market_pct": float(tact_diag.get("time_in_market_pct", 0.0)),
        "avg_exposure_pct": float(tact_diag.get("avg_exposure_pct", 0.0)),
        "blocked_rate_pct": float(tact_diag.get("blocked_signal_rate_pct", 0.0)),
        "blocked_by_confidence": int(tact_diag.get("blocked_by_confidence", 0)),
        "blocked_by_regime": int(tact_diag.get("blocked_by_regime", 0)),
        "blocked_by_cost": int(tact_diag.get("blocked_by_cost", 0)),
        "regime_no_evidence_bars": int(tact_diag.get("regime_no_evidence_bars", 0)),
        "min_confidence_used": float(cfg.min_confidence),
        "data_quality": dq,
        "data_first_dt": dq.get("data_first_dt", ""),
        "data_last_dt": dq.get("data_last_dt", ""),
        "calendar_span_days": int(dq.get("calendar_span_days", 0) or 0),
        "duplicate_timestamps_count": int(dq.get("duplicate_timestamps_count", 0) or 0),
        "is_monotonic_index": bool(dq.get("is_monotonic_index", True)),
        "gap_stats": dq.get("gap_stats", {"max_gap_days": 0, "n_gaps_over_3d": 0}),
        "pct_missing_est": float(dq.get("pct_missing_est", 0.0) or 0.0),
        "has_splits": bool(dq.get("has_splits", False)),
        "has_dividends": bool(dq.get("has_dividends", False)),
        "has_split_events": bool(dq.get("has_split_events", False)),
        "has_dividend_events": bool(dq.get("has_dividend_events", False)),
        "any_nonpositive_prices": int(dq.get("any_nonpositive_prices", 0) or 0),
        "any_nan_prices": int(dq.get("any_nan_prices", 0) or 0),
        "tail_diagnostics": tail_diag,
        "tactical_sleeve_metrics": {
            "total_return_pct": round(tactical_return, 2),
            "vs_cash_pct": round(tactical_vs_cash, 2),
            "max_dd_pct": tactical_max_dd,
            "sharpe": _rounded_or_none(perf.get("sharpe_ratio", None), digits=3),
            "fills": tactical_fills,
            "time_in_market_pct": float(tact_diag.get("time_in_market_pct", 0.0)),
            "avg_exposure_pct": float(tact_diag.get("avg_exposure_pct", 0.0)),
            "blocked_rate_pct": float(tact_diag.get("blocked_signal_rate_pct", 0.0)),
            "blocked_by_confidence": int(tact_diag.get("blocked_by_confidence", 0)),
            "blocked_by_regime": int(tact_diag.get("blocked_by_regime", 0)),
            "blocked_by_cost": int(tact_diag.get("blocked_by_cost", 0)),
            "regime_no_evidence_bars": int(tact_diag.get("regime_no_evidence_bars", 0)),
        },
        "confidence_bucket_table": _summary_confidence_buckets(bt_results),
        "confidence_bins_0p02_0p50_to_1p00": _compute_confidence_bins_0p02(
            trade_records, start=0.50, end=1.00,
        ),
        "confidence_bins_0p02_0p60_to_1p00": _compute_confidence_bins_0p02(
            trade_records, start=0.60, end=1.00,
        ),
    }

    # In two-layer mode, expose total-portfolio metrics for batch aggregation.
    if tl_result is not None and cfg.two_layer_mode:
        total_eq = tl_result.get("total_equity_curve", [])
        base_eq = tl_result.get("baseline_equity_curve", [])
        total_ret = _curve_return_pct(total_eq)
        total_dd = _curve_max_dd_pct(total_eq)
        total_sharpe = _rounded_or_none(
            _excess_return_sharpe(
                total_eq, cash_yield_annual_pct=cfg.cash_yield_annual_pct,
            ),
            digits=3,
        )
        static_ret = _curve_return_pct(base_eq)

        # Keep tactical fields available while promoting top-level totals.
        payload["tactical_return_pct"] = round(tactical_return, 2)
        payload["tactical_max_dd_pct"] = tactical_max_dd
        payload["tactical_sharpe"] = _rounded_or_none(perf.get("sharpe_ratio", None), digits=3)
        payload["tactical_vs_cash_pct"] = round(tactical_vs_cash, 2)
        payload["tactical_fills"] = tactical_fills

        payload["total_return_pct"] = round(total_ret, 2)
        payload["max_dd_pct"] = round(total_dd, 2)
        payload["sharpe"] = total_sharpe
        payload["static_baseline_return_pct"] = round(static_ret, 2)
        payload["overlay_alpha_vs_static"] = round(total_ret - static_ret, 2)
        payload["total_portfolio_metrics"] = {
            "total_return_pct": round(total_ret, 2),
            "max_dd_pct": round(total_dd, 2),
            "sharpe": total_sharpe,
            "static_baseline_return_pct": round(static_ret, 2),
            "overlay_alpha_vs_static": round(total_ret - static_ret, 2),
        }
        # v3.4: matched-core-entry baseline
        matched_eq = tl_result.get("baseline_matched_core_entry", [])
        if matched_eq and len(matched_eq) >= 2:
            matched_ret = _curve_return_pct(matched_eq)
            payload["matched_baseline_return_pct"] = round(matched_ret, 2)
        payload["core_entry_mode"] = str(cfg.core_entry_mode)
        payload["rebalancing_events"] = int(tl_result.get("rebalancing_events", 0))
        if cfg.core_entry_mode == "dca":
            payload["core_dca_days"] = int(cfg.core_dca_days)
            payload["core_dca_start"] = int(cfg.core_dca_start)
        elif cfg.core_entry_mode == "adaptive":
            ada_states = tl_result.get("core_adaptive_states", [])
            from collections import Counter as _Counter
            payload["core_adaptive_params"] = {
                "base_days": int(cfg.core_adaptive_base_days),
                "slow_days": int(cfg.core_adaptive_slow_days),
                "fast_days": int(cfg.core_adaptive_fast_days),
                "vol_target": float(cfg.core_adaptive_vol_target),
                "vol_floor": float(cfg.core_adaptive_vol_floor),
                "dd_window": int(cfg.core_adaptive_dd_window),
                "vol_window": int(cfg.core_adaptive_vol_window),
            }
            payload["core_adaptive_state_counts"] = dict(_Counter(ada_states))
    return payload


def _generate_insufficient_data_report(
    ticker: str,
    cfg: MeanReversionConfig,
    bars: int,
    require_min_bars: int,
    data_quality: Optional[Dict[str, Any]] = None,
    bias_audit: Optional[Dict[str, Any]] = None,
    survivorship_sensitivity: Optional[Dict[str, Any]] = None,
) -> str:
    if not isinstance(bias_audit, dict):
        bias_audit = _build_bias_audit_payload(cfg, data_quality)
    lines: List[str] = []
    lines.append("=" * 74)
    lines.append(f"{'MEAN REVERSION BACKTESTER - INSUFFICIENT DATA':^74}")
    lines.append(f"{ticker} -- INSUFFICIENT DATA")
    lines.append("=" * 74)
    lines.append(f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Bars      : {bars}")
    lines.append(f"Requirement: require_min_bars={int(require_min_bars)}")
    lines.append("")
    lines.append("Status: INSUFFICIENT DATA")
    lines.append("No backtest executed for this run.")
    lines.append("")
    _append_data_quality_section(lines, data_quality)
    _append_bias_audit_section(lines, bias_audit, survivorship_sensitivity)
    lines.append("=" * 74)
    lines.append("DISCLAIMER: Educational only. NOT financial advice.")
    lines.append("=" * 74)
    return "\n".join(lines)


def _generate_portfolio_report(
    tickers: List[str],
    common_index: pd.DatetimeIndex,
    bt_results: Dict[str, Any],
    perf: Dict[str, Any],
    cfg: MeanReversionConfig,
    *,
    skipped: Optional[List[Dict[str, Any]]] = None,
) -> str:
    lines: List[str] = []
    lines.append("=" * 74)
    lines.append(f"{'MEAN REVERSION BACKTESTER - PORTFOLIO TACTICAL MODE':^74}")
    lines.append(f"UNIVERSE: {', '.join(tickers)}")
    lines.append("=" * 74)
    lines.append(f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Bars      : {len(common_index)}")
    lines.append(f"Mode      : portfolio | K={int(cfg.tactical_max_positions)} | weighting={cfg.tactical_weighting}")
    lines.append("")
    if skipped:
        lines.append("SKIPPED TICKERS")
        lines.append("-" * 74)
        for item in skipped:
            lines.append(f"  {item.get('ticker','')}: {item.get('reason','')}")
        lines.append("")

    lines.append("PERFORMANCE")
    lines.append("-" * 74)
    lines.append(f"  Starting Capital : ${perf.get('starting_capital', cfg.starting_capital):>12,.2f}")
    lines.append(f"  Ending Capital   : ${perf.get('ending_capital', cfg.starting_capital):>12,.2f}")
    lines.append(f"  Total Return     : {safe_number(perf.get('total_return_pct'), 0.0):>+10.2f}%")
    lines.append(f"  Sharpe Ratio     : {_format_sharpe(perf.get('sharpe_ratio', 'N/A'), width=10)}")
    lines.append(f"  Max Drawdown     : {safe_number(perf.get('max_drawdown_pct'), 0.0):>10.2f}%")
    lines.append(f"  Total Fills      : {int(perf.get('total_fills', len(bt_results.get('fills', [])))):>10d}")
    lines.append("")

    tact_diag = _tactical_diagnostics(bt_results)
    lines.append("TACTICAL LAYER DIAGNOSTICS")
    lines.append("-" * 74)
    lines.append(f"  Time in Market   : {tact_diag.get('time_in_market_pct', 0.0):.1f}%")
    lines.append(f"  Avg Exposure     : {tact_diag.get('avg_exposure_pct', 0.0):.1f}%")
    lines.append(f"  Blocked Rate     : {tact_diag.get('blocked_signal_rate_pct', 0.0):.1f}%")
    lines.append(f"  blocked_by_confidence : {int(tact_diag.get('blocked_by_confidence', 0))}")
    lines.append(f"  blocked_by_regime : {int(tact_diag.get('blocked_by_regime', 0))}")
    lines.append(f"  blocked_by_cost : {int(tact_diag.get('blocked_by_cost', 0))}")
    lines.append("")

    _append_confidence_bucket_table(lines, bt_results)
    lines.append("=" * 74)
    lines.append("DISCLAIMER: Educational only. NOT financial advice.")
    lines.append("=" * 74)
    return "\n".join(lines)


def analyze_portfolio(
    tickers: List[str],
    cfg: Optional[MeanReversionConfig] = None,
    quiet: bool = False,
    reports_dir: str = "reports",
    export_trades: bool = True,
    run_label: str = "",
    period: str = "500d",
    interval: str = "1d",
    price_field: str = "adjclose",
    require_min_bars: int = 400,
    use_adjclose_scaling: bool = True,
) -> Dict[str, Any]:
    cfg = cfg or MeanReversionConfig()
    uniq_tickers = [str(t).upper() for t in tickers if str(t).strip()]
    uniq_tickers = list(dict.fromkeys(uniq_tickers))
    out_dir = Path(reports_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_suffix = f"_{run_label}" if run_label else ""

    if not quiet:
        print(f"\n{'='*58}")
        print(f"Mean Reversion Backtester v3.3 -- PORTFOLIO ({len(uniq_tickers)} tickers)")
        print("=" * 58)

    data_by_ticker: Dict[str, pd.DataFrame] = {}
    skipped: List[Dict[str, Any]] = []
    for t in uniq_tickers:
        df, _ = fetch_price_data(
            t,
            period=period,
            interval=interval,
            price_field=price_field,
            use_adjclose_scaling=use_adjclose_scaling,
        )
        if df is None or len(df) == 0:
            skipped.append({"ticker": t, "reason": "NO DATA"})
            continue
        if cfg.universe_filter_enabled:
            passed, reasons, metrics = universe_filter(
                df, cfg, require_min_bars=int(require_min_bars), interval=interval,
            )
            if not passed:
                skipped.append({"ticker": t, "reason": "; ".join(reasons), "metrics": metrics})
                # Optional per-ticker skip artifact.
                skip_report = _generate_skipped_report(t, cfg, reasons, metrics)
                safe_t = _safe_ticker_for_filename(t)
                (out_dir / f"{safe_t}_SKIPPED_{ts}.txt").write_text(skip_report, encoding="utf-8")
                continue
        data_by_ticker[t] = df

    if not data_by_ticker:
        report = "PORTFOLIO MODE: no eligible tickers after filtering."
        report_path = out_dir / f"PORTFOLIO_TL{run_suffix}_{ts}.txt"
        report_path.write_text(report, encoding="utf-8")
        summary = {
            "ticker": "PORTFOLIO",
            "mode": "portfolio",
            "reason": "NO_ELIGIBLE_TICKERS",
            "input_tickers": uniq_tickers,
            "skipped": skipped,
            "period": period,
            "interval": interval,
        }
        summary_path = out_dir / f"PORTFOLIO_SUMMARY_{ts}{run_suffix}.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return {
            "ticker": "PORTFOLIO",
            "file": str(report_path),
            "summary_file": str(summary_path),
            "trades_file": None,
            "reason": "NO_ELIGIBLE_TICKERS",
        }

    bt_results, perf, common_index = _run_portfolio_tactical_backtest(data_by_ticker, cfg)
    if len(common_index) < int(require_min_bars):
        report = (
            "PORTFOLIO MODE: INSUFFICIENT INTERSECTION DATA\n"
            f"bars={len(common_index)}, require_min_bars={int(require_min_bars)}"
        )
        report_path = out_dir / f"PORTFOLIO_TL{run_suffix}_{ts}.txt"
        report_path.write_text(report, encoding="utf-8")
        summary = {
            "ticker": "PORTFOLIO",
            "mode": "portfolio",
            "reason": "INSUFFICIENT_INTERSECTION_DATA",
            "bars": int(len(common_index)),
            "require_min_bars": int(require_min_bars),
            "active_tickers": sorted(list(data_by_ticker.keys())),
            "skipped": skipped,
            "period": period,
            "interval": interval,
        }
        summary_path = out_dir / f"PORTFOLIO_SUMMARY_{ts}{run_suffix}.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return {
            "ticker": "PORTFOLIO",
            "file": str(report_path),
            "summary_file": str(summary_path),
            "trades_file": None,
            "reason": "INSUFFICIENT_INTERSECTION_DATA",
        }

    report = _generate_portfolio_report(
        sorted(list(data_by_ticker.keys())), common_index, bt_results, perf, cfg, skipped=skipped,
    )
    report_path = out_dir / f"PORTFOLIO_TL{run_suffix}_{ts}.txt"
    report_path.write_text(report, encoding="utf-8")

    sharpe_raw = safe_number(perf.get("sharpe_ratio", None), None)
    sharpe_val = round(float(sharpe_raw), 3) if sharpe_raw is not None else None

    summary = {
        "ticker": "PORTFOLIO",
        "mode": "portfolio",
        "period": str(period),
        "interval": str(interval),
        "bars": int(len(common_index)),
        "active_tickers": sorted(list(data_by_ticker.keys())),
        "skipped": skipped,
        "tactical_max_positions": int(cfg.tactical_max_positions),
        "tactical_weighting": str(cfg.tactical_weighting),
        "tactical_entry_z": float(cfg.tactical_entry_z),
        "tactical_exit_z": float(cfg.tactical_exit_z),
        "tactical_min_hold_days": int(cfg.tactical_min_hold_days),
        "tactical_max_hold_days": int(cfg.tactical_max_hold_days),
        "confidence_sizing_enabled": bool(cfg.confidence_sizing_enabled),
        "confidence_c0": float(cfg.confidence_c0),
        "confidence_gamma": float(cfg.confidence_gamma),
        "cost_aware_entry_enabled": bool(cfg.cost_aware_entry_enabled),
        "cost_bps_est": float(cfg.cost_bps_est),
        "cost_k": float(cfg.cost_k),
        "min_confidence_used": float(cfg.min_confidence),
        "cash_yield": float(cfg.cash_yield_annual_pct),
        "total_return_pct": round(float(perf.get("total_return_pct", 0.0)), 2),
        "max_dd_pct": round(float(perf.get("max_drawdown_pct", 0.0)), 2),
        "sharpe": sharpe_val,
        "fills": int(perf.get("total_fills", len(bt_results.get("fills", [])))),
        "blocked_count": int(bt_results.get("blocked_count", 0)),
        "blocked_by_confidence": int(bt_results.get("blocked_by_confidence", 0)),
        "blocked_by_regime": int(bt_results.get("blocked_by_regime", 0)),
        "blocked_by_cost": int(bt_results.get("blocked_by_cost", 0)),
        "time_in_market_pct": float(_tactical_diagnostics(bt_results).get("time_in_market_pct", 0.0)),
        "avg_exposure_pct": float(_tactical_diagnostics(bt_results).get("avg_exposure_pct", 0.0)),
        "confidence_bucket_table": _summary_confidence_buckets(bt_results),
        "confidence_bins_0p02_0p50_to_1p00": _compute_confidence_bins_0p02(
            bt_results.get("trade_records", []), start=0.50, end=1.00,
        ),
        "confidence_bins_0p02_0p60_to_1p00": _compute_confidence_bins_0p02(
            bt_results.get("trade_records", []), start=0.60, end=1.00,
        ),
    }
    summary_path = out_dir / f"PORTFOLIO_SUMMARY_{ts}{run_suffix}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    trades_path: Optional[Path] = None
    if export_trades:
        try:
            trades_path = out_dir / f"PORTFOLIO_TRADES_{ts}{run_suffix}.csv"
            _write_trade_ledger_csv(
                trades_path, "PORTFOLIO", bt_results.get("trade_records", []),
                min_confidence_used=float(cfg.min_confidence),
            )
        except Exception as e:
            print(f"[warning] Portfolio trade ledger export failed: {e}")

    if not quiet:
        print(f"\nSaved: {report_path}")
        print(f"Saved: {summary_path}")
        if trades_path is not None:
            print(f"Saved: {trades_path}")

    return {
        "ticker": "PORTFOLIO",
        "file": str(report_path),
        "summary_file": str(summary_path),
        "trades_file": (str(trades_path) if trades_path is not None else None),
        "performance": perf,
        "fills": bt_results.get("fills", []),
        "equity_curve": bt_results.get("equity_curve", []),
        "trade_records": bt_results.get("trade_records", []),
    }


def analyze(
    ticker: str, cfg: Optional[MeanReversionConfig] = None, quiet: bool = False,
    reports_dir: str = "reports", export_trades: bool = True,
    file_prefix: str = "", run_label: str = "",
    period: str = "500d", interval: str = "1d",
    price_field: str = "adjclose", require_min_bars: int = 400,
    use_adjclose_scaling: bool = True,
    universe_name: str = "", universe_asof: str = "", universe_source: str = "",
    survivorship_drag_ann: float = 0.0,
) -> Dict:
    ticker = ticker.upper()
    safe_ticker = _safe_ticker_for_filename(ticker)
    cfg = cfg or MeanReversionConfig()

    if cfg.debug:
        logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")

    version_tag = "v3.3 (Two-Layer)" if cfg.two_layer_mode else "v3.3"
    if not quiet:
        print(f"\n{'='*58}")
        print(f"Mean Reversion Backtester {version_tag} -- {ticker}")
        print("=" * 58)

    # 1. Data
    if not quiet:
        print("[1/7] Price data...")
    df, fetch_meta = fetch_price_data(
        ticker, period=period, interval=interval, price_field=price_field,
        use_adjclose_scaling=use_adjclose_scaling,
    )
    if df is None:
        print(f"  X No data for {ticker}")
        return {"ticker": ticker, "error": "no data"}
    raw_dup_raw = fetch_meta.get("raw_duplicate_timestamps_count")
    raw_dup_num = safe_number(raw_dup_raw, None)
    raw_dup_count = int(raw_dup_num) if raw_dup_num is not None else None
    data_quality = compute_data_quality_diagnostics(
        df,
        requested_price_field=str(fetch_meta.get("requested_price_field", price_field)),
        price_field_used=str(fetch_meta.get("price_field_used", "close")),
        price_field_warning=str(fetch_meta.get("price_field_warning", "")),
        raw_duplicate_timestamps_count=raw_dup_count,
        raw_is_monotonic_index=(
            bool(fetch_meta.get("raw_is_monotonic_index"))
            if fetch_meta.get("raw_is_monotonic_index") is not None else None
        ),
        has_splits=bool(fetch_meta.get("has_splits", False)),
        has_dividends=bool(fetch_meta.get("has_dividends", False)),
        has_split_events=bool(fetch_meta.get("has_split_events", False)),
        has_dividend_events=bool(fetch_meta.get("has_dividend_events", False)),
        ohlc_scaled_by_adjclose=bool(fetch_meta.get("ohlc_scaled_by_adjclose", False)),
        volume_inverse_scaled=bool(fetch_meta.get("volume_inverse_scaled", False)),
        interval=interval,
        gap_days_threshold_1d=int(cfg.gap_days_threshold),
    )
    bias_audit = _build_bias_audit_payload(
        cfg,
        data_quality,
        universe_name=universe_name,
        universe_asof=universe_asof,
        universe_source=universe_source,
    )
    config_hash = _build_config_hash(
        cfg,
        period=period,
        interval=interval,
        price_field=str(data_quality.get("price_field_used", price_field)),
        require_min_bars=require_min_bars,
    )
    if not quiet:
        print(f"  OK {len(df)} bars")
        warn_msg = str(data_quality.get("price_field_warning", "") or "")
        if warn_msg:
            print(f"  WARNING: {warn_msg}")

    out_dir = Path(reports_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = "TL" if cfg.two_layer_mode else "MR_v31"
    run_suffix = f"_{run_label}" if run_label else ""
    fpath = out_dir / f"{file_prefix}{safe_ticker}_{suffix}{run_suffix}_{ts}.txt"
    summary_path: Optional[Path] = None
    trades_path: Optional[Path] = None
    summary_suffix = f"_{ts}{run_suffix}" if run_suffix else f"_{ts}"

    survivorship_sensitivity = _build_survivorship_sensitivity_payload(
        df, cfg, None, survivorship_drag_ann,
    )

    # --- Universe filter (v3.4) ---
    if cfg.universe_filter_enabled:
        uf_passed, uf_reasons, uf_metrics = universe_filter(
            df, cfg, require_min_bars=int(require_min_bars), interval=interval,
        )
        if not uf_passed:
            if not quiet:
                print(f"  SKIPPED by universe filter: {'; '.join(uf_reasons)}")
            skip_report = _generate_skipped_report(
                ticker, cfg, uf_reasons, uf_metrics,
                data_quality=data_quality, bias_audit=bias_audit,
            )
            fpath.write_text(skip_report, encoding="utf-8")
            skip_summary = {
                "ticker": ticker,
                "reason": "SKIPPED BY UNIVERSE FILTER",
                "filter_reasons": uf_reasons,
                "filter_metrics": uf_metrics,
                "bars": int(len(df)),
                "bars_downloaded": int(len(df)),
                "period": str(period),
                "interval": str(interval),
                "price_field": str(data_quality.get("price_field_used", price_field)),
                "price_field_used": str(data_quality.get("price_field_used", price_field)),
                "price_field_warning": str(data_quality.get("price_field_warning", "")),
                "data_quality": data_quality,
                "bias_audit": bias_audit,
                "config_hash": config_hash,
            }
            summary_path = out_dir / f"{safe_ticker}_SUMMARY{summary_suffix}.json"
            summary_path.write_text(
                json.dumps(skip_summary, indent=2), encoding="utf-8",
            )
            if not quiet:
                print(f"\nSaved: {fpath}")
                print(f"Saved: {summary_path}")
            return {
                "ticker": ticker,
                "file": str(fpath),
                "summary_file": str(summary_path),
                "trades_file": None,
                "skipped_by_universe_filter": True,
                "reason": "SKIPPED BY UNIVERSE FILTER",
                "filter_reasons": uf_reasons,
                "filter_metrics": uf_metrics,
                "bars": int(len(df)),
                "bars_downloaded": int(len(df)),
            }

    if len(df) < int(require_min_bars):
        insufficient_report = _generate_insufficient_data_report(
            ticker=ticker,
            cfg=cfg,
            bars=len(df),
            require_min_bars=require_min_bars,
            data_quality=data_quality,
            bias_audit=bias_audit,
            survivorship_sensitivity=survivorship_sensitivity,
        )
        fpath.write_text(insufficient_report, encoding="utf-8")
        summary_payload = {
            "ticker": ticker,
            "reason": "INSUFFICIENT DATA",
            "bars": int(len(df)),
            "bars_downloaded": int(len(df)),
            "require_min_bars": int(require_min_bars),
            "period": str(period),
            "interval": str(interval),
            "price_field": str(data_quality.get("price_field_used", price_field)),
            "price_field_used": str(data_quality.get("price_field_used", price_field)),
            "price_field_warning": str(data_quality.get("price_field_warning", "")),
            "ohlc_scaled_by_adjclose": bool(data_quality.get("ohlc_scaled_by_adjclose", False)),
            "volume_inverse_scaled": bool(data_quality.get("volume_inverse_scaled", False)),
            "universe_name": str(bias_audit.get("universe_name", "")),
            "universe_asof": str(bias_audit.get("universe_asof", "")),
            "universe_source": str(bias_audit.get("universe_source", "")),
            "survivorship_drag_ann": float(safe_number(survivorship_sensitivity.get("survivorship_drag_ann"), 0.0) or 0.0),
            "heuristic_sensitivity_only": bool(survivorship_sensitivity.get("heuristic_sensitivity_only", True)),
            "core_weight": float(safe_number(survivorship_sensitivity.get("core_weight"), 0.0) or 0.0),
            "core_cagr": safe_number(survivorship_sensitivity.get("core_cagr"), None),
            "static_baseline_cagr": safe_number(survivorship_sensitivity.get("static_baseline_cagr"), None),
            "core_cagr_adj": safe_number(survivorship_sensitivity.get("core_cagr_adj"), None),
            "static_baseline_cagr_adj": safe_number(survivorship_sensitivity.get("static_baseline_cagr_adj"), None),
            "bias_audit": bias_audit,
            "survivorship_sensitivity": survivorship_sensitivity,
            "min_confidence_used": float(cfg.min_confidence),
            "core_pct": float(cfg.core_allocation_pct),
            "tactical_pct": float(cfg.tactical_allocation_pct),
            "cash_yield": float(cfg.cash_yield_annual_pct),
            "config_hash": config_hash,
            "data_first_dt": data_quality.get("data_first_dt", ""),
            "data_last_dt": data_quality.get("data_last_dt", ""),
            "calendar_span_days": int(data_quality.get("calendar_span_days", 0) or 0),
            "duplicate_timestamps_count": int(data_quality.get("duplicate_timestamps_count", 0) or 0),
            "is_monotonic_index": bool(data_quality.get("is_monotonic_index", True)),
            "gap_stats": data_quality.get("gap_stats", {"max_gap_days": 0, "n_gaps_over_3d": 0}),
            "pct_missing_est": float(data_quality.get("pct_missing_est", 0.0) or 0.0),
            "has_splits": bool(data_quality.get("has_splits", False)),
            "has_dividends": bool(data_quality.get("has_dividends", False)),
            "has_split_events": bool(data_quality.get("has_split_events", False)),
            "has_dividend_events": bool(data_quality.get("has_dividend_events", False)),
            "any_nonpositive_prices": int(data_quality.get("any_nonpositive_prices", 0) or 0),
            "any_nan_prices": int(data_quality.get("any_nan_prices", 0) or 0),
            "data_quality": data_quality,
        }
        summary_path = out_dir / f"{safe_ticker}_SUMMARY{summary_suffix}.json"
        summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
        if not quiet:
            print("\n" + insufficient_report)
            print(f"\nSaved: {fpath}")
            print(f"Saved: {summary_path}")
        return {
            "ticker": ticker,
            "file": str(fpath),
            "summary_file": str(summary_path),
            "trades_file": None,
            "insufficient_data": True,
            "reason": "INSUFFICIENT DATA",
            "bars": int(len(df)),
        }

    # 2. Technicals
    if not quiet:
        print("[2/7] Technicals...")
    technicals = get_technicals(df)

    # 3. Statistical metrics
    if not quiet:
        print("[3/7] Statistical tests...")
    strategy = MeanReversionStrategy(cfg)
    stat_metrics = strategy.get_metrics(df)
    if not quiet:
        print(f"  OK {stat_metrics['verdict']}")
        adf_note = stat_metrics.get("adf_note", "")
        if adf_note:
            print(f"  ADF: {adf_note}")

    # 4. Ratio + ratio z-score
    if not quiet:
        print("[4/7] Ratio anchor + z-scores...")
    ratio = strategy.compute_ratio(df)
    ratio_z = strategy.compute_ratio_z(ratio)
    if not quiet:
        rz_last = ratio_z.iloc[-1]
        print(f"  OK ratio_z(latest)={rz_last:.2f}" if not pd.isna(rz_last) else "  OK (warming up)")

    # 5. Regime classification
    if not quiet:
        print("[5/7] Regime classification...")
    regime_labels, regime_scores = strategy.classify_regime(df)
    regime_no_evidence_bars = int((regime_labels.astype(str) == "NO_EVIDENCE").sum())
    if not quiet:
        latest_regime = regime_labels.iloc[-1]
        print(f"  OK current regime: {latest_regime}")
    if regime_no_evidence_bars > 0:
        print(
            f"WARNING: regime tests produced no evidence for {regime_no_evidence_bars} bars; "
            "trading may be blocked due to missing stats or insufficient data."
        )

    # 6. Backtest
    if not quiet:
        print("[6/7] Backtesting...")

    tl_result = None
    if cfg.two_layer_mode:
        # --- Two-Layer mode ---
        if not quiet:
            print(f"  Two-Layer: Core {cfg.core_allocation_pct*100:.0f}% + "
                  f"Tactical {cfg.tactical_allocation_pct*100:.0f}%")
        tl_engine = TwoLayerPortfolioEngine(cfg)
        tl_result = tl_engine.run(df, ratio_z, regime_labels, regime_scores)
        bt_results = tl_result["tactical_bt"]
        perf = calculate_performance_metrics(
            bt_results["equity_curve"], bt_results["fills"],
            cash_yield_annual_pct=cfg.cash_yield_annual_pct,
        )
        if len(df) >= 2:
            bh = (float(df["close"].iloc[-1]) / float(df["close"].iloc[0]) - 1) * 100
            perf["buy_and_hold_return_pct"] = round(bh, 2)
        if not quiet:
            total_eq = tl_result["total_equity_curve"]
            total_ret = (total_eq[-1] / total_eq[0] - 1) * 100
            print(f"  OK Total return {total_ret:+.2f}% | "
                  f"Tactical fills {len(bt_results['fills'])}")
    else:
        # --- Legacy single-engine mode ---
        bt_results, perf = _run_backtest(df, ratio_z, regime_labels, regime_scores, cfg)
        if not quiet:
            n_fills = len(bt_results["fills"])
            sharpe_text = _format_sharpe(perf["sharpe_ratio"])
            print(f"  OK {n_fills} fills | Sharpe {sharpe_text} | "
                  f"Return {perf['total_return_pct']:+.2f}%")

    survivorship_sensitivity = _build_survivorship_sensitivity_payload(
        df, cfg, tl_result, survivorship_drag_ann,
    )

    # Analysis-only trade enrichments (no effect on execution/PnL accounting)
    trade_records = bt_results.get("trade_records", [])
    if isinstance(trade_records, list):
        _compute_trade_excursions(trade_records, df["close"], ratio_z)

    # 7. Baseline
    if not quiet:
        print("[7/7] Baseline backtest (no filter)...")
    cfg_base = cfg.copy_with(
        regime_filter_enabled=False,
        allowed_regimes=["MEAN_REVERTING", "SIDEWAYS", "TRENDING", "AMBIGUOUS"],
        ambiguous_policy="allow_small",
        two_layer_mode=False,
    )
    bt_baseline, perf_baseline = _run_backtest(
        df, ratio_z, regime_labels, regime_scores, cfg_base,
    )
    if not quiet:
        print(f"  OK Baseline return {perf_baseline['total_return_pct']:+.2f}%")

    # Info
    info = fetch_basic_info(ticker)

    # Report
    if cfg.two_layer_mode:
        assert tl_result is not None
        report = generate_two_layer_report(
            ticker, df, tl_result, perf, stat_metrics, bt_results, info, cfg,
            data_quality=data_quality,
            bias_audit=bias_audit,
            survivorship_sensitivity=survivorship_sensitivity,
        )
    else:
        report = generate_report(
            ticker, df, technicals, stat_metrics,
            bt_results, perf, bt_baseline, perf_baseline, info, cfg,
            data_quality=data_quality,
            bias_audit=bias_audit,
            survivorship_sensitivity=survivorship_sensitivity,
        )
    if not quiet:
        print("\n" + report)

    fpath.write_text(report, encoding="utf-8")
    try:
        summary_payload = _build_summary_payload(
            ticker, df, cfg, perf, bt_results, tl_result,
            period=period, interval=interval,
            price_field=str(data_quality.get("price_field_used", price_field)),
            require_min_bars=require_min_bars,
            data_quality=data_quality,
            config_hash=config_hash,
            survivorship_drag_ann=survivorship_drag_ann,
            bias_audit=bias_audit,
            survivorship_sensitivity=survivorship_sensitivity,
        )
        summary_path = out_dir / f"{safe_ticker}_SUMMARY{summary_suffix}.json"
        summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    except Exception as e:
        print(f"[warning] Summary export failed ({ticker}): {e}")

    if export_trades:
        try:
            trade_records = bt_results.get("trade_records", [])
            if not isinstance(trade_records, list):
                trade_records = []
            trades_path = out_dir / f"{safe_ticker}_TRADES{summary_suffix}.csv"
            _write_trade_ledger_csv(
                trades_path, ticker, trade_records, min_confidence_used=cfg.min_confidence,
            )
        except Exception as e:
            print(f"[warning] Trade ledger export failed ({ticker}): {e}")

    if not quiet:
        print(f"\nSaved: {fpath}")
        if summary_path is not None:
            print(f"Saved: {summary_path}")
        if trades_path is not None:
            print(f"Saved: {trades_path}")

    # --- Plot outputs ---
    plot_paths: Dict[str, str] = {}
    if cfg.plot:
        if not quiet:
            print("[plot] Generating charts + CSV...")
        equity_for_plot = (
            tl_result["total_equity_curve"] if tl_result
            else bt_results["equity_curve"]
        )
        plot_paths = generate_benchmark_outputs(
            equity_for_plot, df, cfg.starting_capital,
            safe_ticker, str(out_dir),
            two_layer_result=tl_result,
            bt_results=bt_results,
        )
        if plot_paths:
            if not quiet:
                for kind, p in plot_paths.items():
                    print(f"  Saved: {p}")
        else:
            if not quiet:
                print("  (matplotlib not available — skipped)")

    result = {
        "ticker": ticker, "file": str(fpath),
        "summary_file": (str(summary_path) if summary_path is not None else None),
        "trades_file": (str(trades_path) if trades_path is not None else None),
        "stat_metrics": stat_metrics, "performance": perf,
        "performance_baseline": perf_baseline,
        "fills": bt_results["fills"],
        "equity_curve": bt_results["equity_curve"],
        "blocked_count": bt_results["blocked_count"],
        "blocked_by_confidence": bt_results.get("blocked_by_confidence", 0),
        "blocked_by_regime": bt_results.get("blocked_by_regime", 0),
        "blocked_by_cost": bt_results.get("blocked_by_cost", 0),
        "regime_no_evidence_bars": bt_results.get("regime_no_evidence_bars", 0),
        "trade_records": bt_results.get("trade_records", []),
        "plot_paths": plot_paths,
    }
    if tl_result is not None:
        result["two_layer"] = tl_result
    return result


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Mean Reversion Backtester v3.3 — Long-Only Action Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python mean_reversion_standalone.py AAPL\n"
            "  python mean_reversion_standalone.py NVDA --window 30 --debug\n"
            "  python mean_reversion_standalone.py GOOG --no-regime --entry-at same_close\n"
            "  python mean_reversion_standalone.py AAPL --two-layer --plot\n"
            "  python mean_reversion_standalone.py AAPL --two-layer --core-pct 0.7 --tactical-pct 0.3\n"
        ),
    )
    parser.add_argument("tickers", nargs="*", help="Stock ticker(s)")
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--entry", type=float, default=-1.5, help="Entry ratio-z threshold")
    parser.add_argument("--entry-at", choices=["next_open", "same_close"], default="next_open")
    parser.add_argument(
        "--no-lag-signals-for-same-close",
        action="store_true",
        help="Disable lagging of same-close signals (not recommended; may introduce look-ahead)",
    )
    parser.add_argument("--capital", type=float, default=100_000)
    parser.add_argument("--pos-size", type=float, default=0.15, help="Max position pct")
    parser.add_argument("--stop-atr", type=float, default=3.0)
    parser.add_argument("--target-atr", type=float, default=3.0)
    parser.add_argument("--no-regime", action="store_true", help="Disable regime filter")
    parser.add_argument("--ratio-anchor", type=int, default=200)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--quiet", "-q", action="store_true")
    # v3.2: Two-Layer + Plot
    parser.add_argument("--two-layer", action="store_true",
                        help="Enable Core/Tactical two-layer portfolio mode")
    parser.add_argument("--core-pct", type=float, default=0.80,
                        help="Core (buy-hold) allocation fraction (default 0.80)")
    parser.add_argument("--tactical-pct", type=float, default=0.20,
                        help="Tactical (MR) allocation fraction (default 0.20)")
    parser.add_argument("--plot", action="store_true",
                        help="Save benchmark PNG, drawdown PNG, and curves CSV")
    parser.add_argument("--cash-yield", type=float, default=0.0,
                        help="Annual cash yield on idle tactical capital (%%)")
    parser.add_argument("--rebalance-freq", choices=["none", "M", "Q", "A"], default="Q",
                        help="Two-layer sleeve rebalance frequency (none, M, Q, A)")
    parser.add_argument("--rebalance-drift-threshold", type=float, default=0.05,
                        help="Rebalance when core weight drift exceeds this fraction (default 0.05)")
    parser.add_argument("--tactical-vol-targeting", action="store_true",
                        help="Enable tactical realized-vol targeting multiplier on entry sizing")
    parser.add_argument("--tactical-vol-target", type=float, default=0.15,
                        help="Tactical annualized volatility target (default 0.15, Spec v1)")
    parser.add_argument("--tactical-vol-window", type=int, default=20,
                        help="Rolling window (bars) for realized vol estimate (default 20)")
    parser.add_argument("--tactical-vol-floor", type=float, default=0.05,
                        help="Vol floor used in vol targeting (default 0.05)")
    parser.add_argument("--tactical-vol-cap", type=float, default=1.50,
                        help="Upper cap for vol targeting multiplier (default 1.50)")
    parser.add_argument("--cost-aware-entry", choices=["on", "off"], default="off",
                        help="Feature B: require expected-return proxy to clear costs (default off)")
    parser.add_argument("--cost-bps-est", type=float, default=15.0,
                        help="Feature B: estimated one-way/round-turn cost proxy in bps (default 15.0)")
    parser.add_argument("--cost-k", type=float, default=1.0,
                        help="Spec v1: safety multiple on cost threshold (default 1.0)")
    parser.add_argument("--better-exits", choices=["on", "off"], default="off",
                        help="Feature C: enable return-to-mean/time-stop next-open exits (default off)")
    parser.add_argument("--tactical-exit-z", type=float, default=-0.20,
                        help="Feature C: z threshold for return-to-mean exit (default -0.20)")
    parser.add_argument("--tactical-min-hold-days", type=int, default=3,
                        help="Feature C: minimum hold before z-based exit (default 3)")
    parser.add_argument("--tactical-max-hold-days", type=int, default=30,
                        help="Feature C: hard max hold before next-open exit (default 30)")
    parser.add_argument("--confidence-sizing", choices=["on", "off"], default="off",
                        help="Feature D: confidence-weighted sizing multiplier (default off)")
    parser.add_argument("--confidence-c0", type=float, default=0.60,
                        help="Feature D: baseline confidence where multiplier starts rising (default 0.60)")
    parser.add_argument("--confidence-gamma", type=float, default=1.0,
                        help="Feature D: curvature exponent for confidence multiplier (default 1.0)")
    parser.add_argument("--tactical-mode", choices=["single", "portfolio"], default="single",
                        help="Feature E: tactical run mode (default single)")
    parser.add_argument("--tactical-max-positions", type=int, default=5,
                        help="Feature E: max concurrent tactical positions in portfolio mode (default 5)")
    parser.add_argument("--tactical-entry-z", type=float, default=-1.25,
                        help="Feature E: entry z threshold used for portfolio candidate selection (default -1.25)")
    parser.add_argument("--tactical-weighting", choices=["equal", "inv_vol"], default="equal",
                        help="Feature E: portfolio tactical sizing scheme (default equal)")
    # v3.4: Core DCA Entry
    parser.add_argument("--core-entry", choices=["instant", "dca", "adaptive"], default="instant",
                        help="Core entry mode: instant (default), dca, or adaptive")
    parser.add_argument("--core-dca-days", type=int, default=40,
                        help="Number of bars to spread core DCA over (default 40)")
    parser.add_argument("--core-dca-start", type=int, default=0,
                        help="Bar offset to begin DCA (default 0)")
    parser.add_argument("--core-dca-commission", type=float, default=0.0,
                        help="Per-fill commission for core DCA buys (default 0.0)")
    parser.add_argument("--core-dca-slippage-pct", type=float, default=0.0,
                        help="Slippage for core DCA buys (default 0.0)")
    # Adaptive core entry
    parser.add_argument("--core-adaptive-base-days", type=int, default=60,
                        help="Adaptive core: NEUTRAL deploy pace in bars (default 60)")
    parser.add_argument("--core-adaptive-slow-days", type=int, default=120,
                        help="Adaptive core: CALM_UPTREND deploy pace (default 120)")
    parser.add_argument("--core-adaptive-fast-days", type=int, default=40,
                        help="Adaptive core: PULLBACK/HIGH_VOL base pace (default 40)")
    parser.add_argument("--core-adaptive-vol-window", type=int, default=20,
                        help="Adaptive core: realized vol window (default 20)")
    parser.add_argument("--core-adaptive-dd-window", type=int, default=252,
                        help="Adaptive core: peak drawdown window (default 252)")
    parser.add_argument("--core-adaptive-vol-target", type=float, default=0.12,
                        help="Adaptive core: annualized vol threshold (default 0.12)")
    parser.add_argument("--core-adaptive-vol-floor", type=float, default=0.05,
                        help="Adaptive core: vol floor for HIGH_VOL scaling (default 0.05)")
    parser.add_argument("--core-adaptive-max-deploy-pct", type=float, default=0.10,
                        help="Adaptive core: max deploy fraction per bar (default 0.10)")
    parser.add_argument("--core-adaptive-min-deploy-pct", type=float, default=0.002,
                        help="Adaptive core: min deploy fraction per bar (default 0.002)")
    parser.add_argument("--core-adaptive-start", type=int, default=0,
                        help="Adaptive core: bar offset to begin deployment (default 0)")
    # v3.4: Universe Filter
    parser.add_argument("--universe-filter", choices=["on", "off"], default="off",
                        help="Enable pre-run universe filter (default off)")
    parser.add_argument("--min-dollar-vol", type=float, default=20_000_000.0,
                        help="Min median daily dollar volume (default 20M)")
    parser.add_argument("--min-price", type=float, default=5.0,
                        help="Min median close price (default 5.0)")
    parser.add_argument("--max-gaps", type=int, default=10,
                        help="Max detected large date gaps (default 10)")
    parser.add_argument("--gap-days-threshold", type=int, default=7,
                        help="For interval=1d, count gaps >= this many calendar days (default 7)")
    parser.add_argument("--max-zero-vol-days", type=int, default=5,
                        help="Maximum tolerated zero-volume days in universe filter (default 5)")
    parser.add_argument("--min-price-tail", type=float, default=1.0,
                        help="Minimum median close over last 30 bars in universe filter (default 1.0)")
    parser.add_argument("--max-nan-frac", type=float, default=0.01,
                        help="Max fraction of rows with NaN close/volume in universe filter (default 0.01)")
    parser.add_argument("--enable-spec-v1", action="store_true",
                        help="Enable Tactical Overlay Upgrade Spec v1 (default off)")
    parser.add_argument("--regime-update-freq", type=int, default=5,
                        help="Run expensive regime stats every N bars (default 5)")
    parser.add_argument("--min-confidence", type=float, nargs="+", default=None,
                        help=("Minimum MR/SW confidence proxy to allow BUY entries. "
                              "Accepts one or multiple values (e.g., 0.0 0.6)."))
    parser.add_argument("--no-export-trades", action="store_true",
                        help="Disable per-run completed-trade ledger export")
    parser.add_argument("--reports-dir", type=str, default="reports",
                        help="Output directory for report artifacts")
    parser.add_argument("--period", type=str, default="500d",
                        help="Data history period (e.g., 500d, 4y, 10y, 20y, max)")
    parser.add_argument("--interval", type=str, default="1d",
                        help="Data interval (e.g., 1d, 1wk)")
    parser.add_argument("--price-field", choices=["adjclose", "close"], default="adjclose",
                        help="Price series source for calculations")
    parser.add_argument("--use-adjclose-scaling", dest="use_adjclose_scaling", action="store_true",
                        help="Scale OHLC by Adj Close/Close ratio and inverse-scale volume (default: enabled)")
    parser.add_argument("--no-use-adjclose-scaling", dest="use_adjclose_scaling", action="store_false",
                        help="Disable Adj Close scaling and use raw OHLCV")
    parser.set_defaults(use_adjclose_scaling=True)
    parser.add_argument("--require-min-bars", type=int, default=400,
                        help="Minimum downloaded bars required to run backtest")
    parser.add_argument("--universe-name", type=str, default="",
                        help="Universe label/provenance (audit metadata only)")
    parser.add_argument("--universe-asof", type=str, default="",
                        help="Universe as-of date (YYYY-MM-DD, audit metadata only)")
    parser.add_argument("--universe-source", type=str, default="",
                        help="Universe source/provenance text (audit metadata only)")
    parser.add_argument("--survivorship-drag-ann", type=float, default=0.0,
                        help="Annual survivorship drag sensitivity (%%, reporting only)")

    args = parser.parse_args()

    if not args.tickers:
        print("\nMean Reversion Backtester v3.3 — Long-Only")
        print("Usage: python mean_reversion_standalone.py AAPL\n")
        try:
            inp = input("Enter ticker(s): ").strip()
            args.tickers = inp.upper().split() if inp else []
        except (EOFError, KeyboardInterrupt):
            return 0

    if not args.tickers:
        return 1

    period = str(args.period).strip().lower()
    interval = str(args.interval).strip().lower()
    if not is_valid_period(period):
        print("Error: invalid --period. Use Nd/Nwk/Nmo/Ny (e.g., 500d, 4y) or 'max'.")
        return 2
    if not is_valid_interval(interval):
        allowed = ", ".join(sorted(_VALID_INTERVALS))
        print(f"Error: invalid --interval '{args.interval}'. Allowed: {allowed}")
        return 2
    price_field = str(args.price_field).strip().lower()
    if not is_valid_price_field(price_field):
        print("Error: invalid --price-field. Allowed: adjclose, close")
        return 2
    if int(args.require_min_bars) < 1:
        print("Error: invalid --require-min-bars. Must be >= 1.")
        return 2
    if not is_valid_iso_date_or_empty(args.universe_asof):
        print("Error: invalid --universe-asof. Expected YYYY-MM-DD or empty.")
        return 2
    if float(args.survivorship_drag_ann) < 0:
        print("Error: invalid --survivorship-drag-ann. Must be >= 0.")
        return 2
    if not (0.0 <= float(args.max_nan_frac) <= 1.0):
        print("Error: invalid --max-nan-frac. Must be in [0, 1].")
        return 2
    if int(args.gap_days_threshold) < 1:
        print("Error: invalid --gap-days-threshold. Must be >= 1.")
        return 2
    if int(args.max_zero_vol_days) < 0:
        print("Error: invalid --max-zero-vol-days. Must be >= 0.")
        return 2
    if float(args.min_price_tail) < 0:
        print("Error: invalid --min-price-tail. Must be >= 0.")
        return 2
    if int(args.regime_update_freq) < 1:
        print("Error: invalid --regime-update-freq. Must be >= 1.")
        return 2
    if float(args.rebalance_drift_threshold) < 0:
        print("Error: invalid --rebalance-drift-threshold. Must be >= 0.")
        return 2
    if float(args.tactical_vol_target) < 0:
        print("Error: invalid --tactical-vol-target. Must be >= 0.")
        return 2
    if int(args.tactical_vol_window) < 2:
        print("Error: invalid --tactical-vol-window. Must be >= 2.")
        return 2
    if float(args.tactical_vol_floor) <= 0:
        print("Error: invalid --tactical-vol-floor. Must be > 0.")
        return 2
    if float(args.tactical_vol_cap) <= 0:
        print("Error: invalid --tactical-vol-cap. Must be > 0.")
        return 2
    if float(args.cost_bps_est) < 0:
        print("Error: invalid --cost-bps-est. Must be >= 0.")
        return 2
    if float(args.cost_k) < 1.0:
        print("Error: invalid --cost-k. Must be >= 1.0.")
        return 2
    if int(args.tactical_min_hold_days) < 0:
        print("Error: invalid --tactical-min-hold-days. Must be >= 0.")
        return 2
    if int(args.tactical_max_hold_days) < 1:
        print("Error: invalid --tactical-max-hold-days. Must be >= 1.")
        return 2
    if not (0.0 <= float(args.confidence_c0) < 1.0):
        print("Error: invalid --confidence-c0. Must be in [0, 1).")
        return 2
    if float(args.confidence_gamma) < 1.0:
        print("Error: invalid --confidence-gamma. Must be >= 1.0.")
        return 2
    if int(args.tactical_max_positions) < 1:
        print("Error: invalid --tactical-max-positions. Must be >= 1.")
        return 2

    if args.min_confidence is None:
        min_conf_values = [0.0, 0.6] if args.two_layer else [0.0]
    else:
        min_conf_values = [float(v) for v in args.min_confidence]

    base_cfg = MeanReversionConfig(
        lookback_window=args.window,
        entry_z=-abs(args.entry),
        entry_at=args.entry_at,
        lag_signals_for_same_close=not bool(args.no_lag_signals_for_same_close),
        starting_capital=args.capital,
        max_position_pct=args.pos_size,
        stop_atr_multiple=args.stop_atr,
        target_atr_multiple=args.target_atr,
        regime_filter_enabled=not args.no_regime,
        regime_update_freq=args.regime_update_freq,
        ratio_anchor_window=args.ratio_anchor,
        debug=args.debug,
        two_layer_mode=args.two_layer,
        core_allocation_pct=args.core_pct,
        tactical_allocation_pct=args.tactical_pct,
        plot=args.plot,
        cash_yield_annual_pct=args.cash_yield,
        rebalance_freq=(None if args.rebalance_freq == "none" else args.rebalance_freq),
        rebalance_drift_threshold=args.rebalance_drift_threshold,
        tactical_vol_targeting_enabled=bool(args.tactical_vol_targeting),
        tactical_vol_target=float(args.tactical_vol_target),
        tactical_vol_window=int(args.tactical_vol_window),
        tactical_vol_floor=float(args.tactical_vol_floor),
        tactical_vol_cap=float(args.tactical_vol_cap),
        cost_aware_entry_enabled=(args.cost_aware_entry == "on"),
        cost_bps_est=float(args.cost_bps_est),
        cost_k=float(args.cost_k),
        better_exits_enabled=(args.better_exits == "on"),
        tactical_exit_z=float(args.tactical_exit_z),
        tactical_min_hold_days=int(args.tactical_min_hold_days),
        tactical_max_hold_days=int(args.tactical_max_hold_days),
        confidence_sizing_enabled=(args.confidence_sizing == "on"),
        confidence_c0=float(args.confidence_c0),
        confidence_gamma=float(args.confidence_gamma),
        tactical_mode=str(args.tactical_mode),
        tactical_max_positions=int(args.tactical_max_positions),
        tactical_entry_z=float(args.tactical_entry_z),
        tactical_weighting=str(args.tactical_weighting),
        min_confidence=min_conf_values[0],
        # v3.4: Core DCA
        core_entry_mode=args.core_entry,
        core_dca_days=args.core_dca_days,
        core_dca_start=args.core_dca_start,
        core_dca_commission=args.core_dca_commission,
        core_dca_slippage_pct=args.core_dca_slippage_pct,
        core_adaptive_base_days=args.core_adaptive_base_days,
        core_adaptive_slow_days=args.core_adaptive_slow_days,
        core_adaptive_fast_days=args.core_adaptive_fast_days,
        core_adaptive_vol_window=args.core_adaptive_vol_window,
        core_adaptive_dd_window=args.core_adaptive_dd_window,
        core_adaptive_vol_target=args.core_adaptive_vol_target,
        core_adaptive_vol_floor=args.core_adaptive_vol_floor,
        core_adaptive_max_deploy_pct=args.core_adaptive_max_deploy_pct,
        core_adaptive_min_deploy_pct=args.core_adaptive_min_deploy_pct,
        core_adaptive_start=args.core_adaptive_start,
        # v3.4: Universe Filter
        universe_filter_enabled=(args.universe_filter == "on"),
        min_dollar_vol=args.min_dollar_vol,
        min_price=args.min_price,
        max_gaps=args.max_gaps,
        gap_days_threshold=args.gap_days_threshold,
        max_zero_vol_days=args.max_zero_vol_days,
        min_price_tail=args.min_price_tail,
        max_nan_frac=args.max_nan_frac,
        # Spec v1: Tactical Overlay Upgrade
        enable_spec_v1_upgrades=bool(args.enable_spec_v1),
    )

    multi_conf_mode = len(min_conf_values) > 1
    if str(base_cfg.tactical_mode) == "portfolio":
        for min_conf in min_conf_values:
            run_cfg = base_cfg.copy_with(min_confidence=float(min_conf))
            run_label = _format_min_confidence_tag(min_conf) if multi_conf_mode else ""
            try:
                analyze_portfolio(
                    [t.upper() for t in args.tickers],
                    run_cfg,
                    args.quiet,
                    reports_dir=args.reports_dir,
                    export_trades=(not args.no_export_trades),
                    run_label=run_label,
                    period=period,
                    interval=interval,
                    price_field=price_field,
                    require_min_bars=int(args.require_min_bars),
                    use_adjclose_scaling=bool(args.use_adjclose_scaling),
                )
            except KeyboardInterrupt:
                print("\nInterrupted.")
                return 130
            except Exception as e:
                print(f"Error (portfolio, min_confidence={min_conf}): {e}")
                if args.debug:
                    import traceback
                    traceback.print_exc()
        return 0

    for t_idx, t in enumerate(args.tickers, start=1):
        file_prefix = f"{t_idx:02d}_" if (args.two_layer and multi_conf_mode) else ""
        for min_conf in min_conf_values:
            run_cfg = base_cfg.copy_with(min_confidence=float(min_conf))
            run_label = _format_min_confidence_tag(min_conf) if multi_conf_mode else ""
            try:
                analyze(
                    t.upper(), run_cfg, args.quiet,
                    reports_dir=args.reports_dir,
                    export_trades=(not args.no_export_trades),
                    file_prefix=file_prefix,
                    run_label=run_label,
                    period=period,
                    interval=interval,
                    price_field=price_field,
                    require_min_bars=int(args.require_min_bars),
                    use_adjclose_scaling=bool(args.use_adjclose_scaling),
                    universe_name=str(args.universe_name or ""),
                    universe_asof=str(args.universe_asof or ""),
                    universe_source=str(args.universe_source or ""),
                    survivorship_drag_ann=float(args.survivorship_drag_ann),
                )
            except KeyboardInterrupt:
                print("\nInterrupted.")
                return 130
            except Exception as e:
                print(f"Error ({t}, min_confidence={min_conf}): {e}")
                if args.debug:
                    import traceback
                    traceback.print_exc()

    return 0


if __name__ == "__main__":
    sys.exit(main())

# =============================================================================
# PHASE 2 DESIGN NOTES (future upgrade — NOT implemented here)
# =============================================================================
#
# Replace hard regime labels (MEAN_REVERTING / SIDEWAYS / TRENDING / AMBIGUOUS)
# with probabilistic state beliefs using a lightweight HMM-style framework:
#
#   1. Define K hidden states (e.g. K=3: MR-friendly, Trending, Ambiguous).
#   2. For each bar, compute observation likelihoods from the existing
#      statistical tests (ADF p-value, Hurst, VR, half-life) treated as
#      continuous features.  Model each state's emission as a multivariate
#      Gaussian (diagonal covariance is fine for a first cut).
#   3. Use a K×K transition matrix with a strong diagonal (persistence prior:
#      P(stay) ~ 0.95) to enforce regime stickiness and reduce whipsaw.
#   4. Run a forward pass (no smoothing needed for online use) to get
#      P(state | observations_1..t) at each bar — the "belief vector."
#   5. Instead of a binary regime gate (allow / block), scale tactical
#      position sizing by P(MR-friendly):
#          size_multiplier = P(MR) + 0.5 * P(Ambiguous) + 0.0 * P(Trending)
#      This removes the sharp threshold artefact.
#   6. Fit the emission parameters on a rolling training window (e.g. 504
#      bars) and refit monthly to allow structural change.
#   7. Transition matrix can either be fixed (calibrated once on broad
#      equity universe) or estimated with Baum-Welch on the training window.
#
# Dependencies: only numpy (no hmmlearn/scipy required).
# The forward pass and Gaussian likelihoods are <50 lines of code.
#
# Acceptance criteria for Phase 2:
#   - Belief vector per bar in [0,1]^K summing to 1.
#   - Regime-label outputs remain available (argmax of beliefs) for backward
#     compatibility with existing tests and reports.
#   - A/B comparison shows reduced whipsaw (fewer regime transitions) and
#     smoother tactical participation vs the current hard-label classifier.
# =============================================================================
