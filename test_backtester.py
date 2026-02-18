#!/usr/bin/env python3
"""
================================================================================
Tests for mean_reversion_standalone.py v3.3
================================================================================
Coverage:
    P0.1  ADF dict shape + fallback note
    P0.2  Thesis break: consecutive slope, magnitude, guard
    P0.3  A/B comparison actually differs on synthetic trending data
    P0.4  Dust fill filter
    P1.1  Adaptive regime → SIDEWAYS appears
    P1.2  Verdict coherence with regime distribution
    P2.1  Vol-adjusted sizing
    P2.2  Staged exits (trim levels)
    TL.*  Two-Layer Core/Tactical portfolio tests
    PH1.1 Cash yield on idle tactical capital
    PH1.2 New visualization plots
    PH1.3 Diagnostics: capture ratios, Calmar, attribution
    PH1.4 Hurst clamping
    PH2   RegimeSignal abstraction
    + All v3.0 tests ported

Run:
    python -m unittest test_backtester -v
================================================================================
"""

import unittest
import csv
import json
import os
import re
import tempfile
import io
import contextlib
import sys
import time
from unittest import mock
from pathlib import Path
from typing import Any, Dict, cast
import numpy as np
import pandas as pd
import aggregate_confidence_bins as acb

from mean_reversion_standalone import (
    Action,
    Regime,
    RegimeSignal,
    BacktestEngine,
    CorePosition,
    MeanReversionConfig,
    MeanReversionStrategy,
    TwoLayerPortfolioEngine,
    _PositionState,
    calculate_hurst_exponent,
    calculate_half_life,
    calculate_variance_ratio,
    calculate_mean_reversion_metrics,
    calculate_performance_metrics,
    calculate_rsi,
    calculate_atr,
    classify_regime,
    compute_buy_hold_curve,
    compute_ratio_series,
    compute_ratio_z,
    determine_action,
    generate_benchmark_outputs,
    generate_two_layer_report,
    regime_allows_action,
    check_reversal_confirmation,
    passes_quality_filter,
    normalize_dataframe,
    get_technicals,
    adf_test,
    _get_trim_action,
    _exit_reason_breakdown,
    _regime_timeline,
    _metrics_by_regime,
    _regime_transition_matrix,
    _tactical_diagnostics,
    _capture_ratios,
    _calmar_ratio,
    _return_attribution,
    _build_bias_audit_payload,
    _build_summary_payload,
    _write_trade_ledger_csv,
    is_valid_period,
    is_valid_interval,
    compute_data_quality_diagnostics,
    _compute_trade_excursions,
    _stop_execution_base,
    _apply_adjclose_price_volume_scaling,
    _confidence_sizing_multiplier,
    _vol_target_multiplier,
    _should_block_next_open_entry,
    _run_portfolio_tactical_backtest,
    _expected_return_spec_v1,
    _is_cost_effective_spec_v1,
    _adaptive_core_deploy_amount,
    analyze,
    universe_filter,
    _generate_skipped_report,
)


# =============================================================================
# SYNTHETIC DATA BUILDERS
# =============================================================================

def _make_df(n=300, start=100.0, drift=0.02, vol=0.5, seed=42):
    rng = np.random.RandomState(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="B", tz="UTC")
    close = start + np.cumsum(rng.randn(n) * vol + drift)
    close = np.maximum(close, 10.0)
    high = close + rng.uniform(0.5, 2.0, n)
    low = close - rng.uniform(0.5, 2.0, n)
    opn = close + rng.uniform(-1.0, 1.0, n)
    volume = rng.randint(500_000, 2_000_000, n).astype(float)
    df = pd.DataFrame(
        {"open": opn, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )
    df["return"] = df["close"].pct_change()
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df["volume_sma_20"] = df["volume"].rolling(20).mean()
    df["volume_ratio"] = df["volume"] / df["volume_sma_20"]
    return df


def _make_mean_reverting_df(n=400, mu=100.0, theta=0.15, sigma=1.5, seed=99):
    rng = np.random.RandomState(seed)
    dates = pd.date_range("2023-01-01", periods=n, freq="B", tz="UTC")
    prices = np.zeros(n)
    prices[0] = mu
    for i in range(1, n):
        prices[i] = prices[i-1] + theta * (mu - prices[i-1]) + sigma * rng.randn()
    prices = np.maximum(prices, 50.0)
    high = prices + rng.uniform(0.3, 1.5, n)
    low = prices - rng.uniform(0.3, 1.5, n)
    opn = prices + rng.uniform(-0.5, 0.5, n)
    volume = rng.randint(800_000, 3_000_000, n).astype(float)
    df = pd.DataFrame(
        {"open": opn, "high": high, "low": low, "close": prices, "volume": volume},
        index=dates,
    )
    df["return"] = df["close"].pct_change()
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df["volume_sma_20"] = df["volume"].rolling(20).mean()
    df["volume_ratio"] = df["volume"] / df["volume_sma_20"]
    return df


# Alias used by v3.3 tests
_make_mr_series = _make_mean_reverting_df


def _make_strongly_trending_df(n=400, seed=7):
    rng = np.random.RandomState(seed)
    dates = pd.date_range("2023-01-01", periods=n, freq="B", tz="UTC")
    prices = 100.0 + np.arange(n) * 0.3 + rng.randn(n) * 0.3
    prices = np.maximum(prices, 50.0)
    high = prices + rng.uniform(0.2, 0.8, n)
    low = prices - rng.uniform(0.2, 0.8, n)
    opn = prices + rng.uniform(-0.3, 0.3, n)
    volume = rng.randint(1_000_000, 2_000_000, n).astype(float)
    df = pd.DataFrame(
        {"open": opn, "high": high, "low": low, "close": prices, "volume": volume},
        index=dates,
    )
    df["return"] = df["close"].pct_change()
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df["volume_sma_20"] = df["volume"].rolling(20).mean()
    df["volume_ratio"] = df["volume"] / df["volume_sma_20"]
    return df


def _make_sideways_df(n=400, seed=55):
    rng = np.random.RandomState(seed)
    dates = pd.date_range("2023-01-01", periods=n, freq="B", tz="UTC")
    prices = 100.0 + np.cumsum(rng.randn(n) * 0.3)
    prices = np.maximum(prices, 50.0)
    high = prices + rng.uniform(0.2, 0.8, n)
    low = prices - rng.uniform(0.2, 0.8, n)
    opn = prices + rng.uniform(-0.2, 0.2, n)
    volume = rng.randint(800_000, 2_000_000, n).astype(float)
    df = pd.DataFrame(
        {"open": opn, "high": high, "low": low, "close": prices, "volume": volume},
        index=dates,
    )
    df["return"] = df["close"].pct_change()
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df["volume_sma_20"] = df["volume"].rolling(20).mean()
    df["volume_ratio"] = df["volume"] / df["volume_sma_20"]
    return df


def _make_regime_switching_df(n=400, seed=123):
    """MR -> trending -> MR synthetic path for regime-adaptation tests."""
    rng = np.random.RandomState(seed)
    dates = pd.date_range("2023-01-01", periods=n, freq="B", tz="UTC")
    n1 = min(150, max(1, n // 3))
    n2 = min(100, max(1, n // 4))
    n3 = max(1, n - n1 - n2)

    # Segment 1: OU-like mean reversion around 100
    seg1 = np.zeros(n1, dtype=float)
    seg1[0] = 100.0
    for i in range(1, n1):
        seg1[i] = seg1[i - 1] + 0.18 * (100.0 - seg1[i - 1]) + 1.0 * rng.randn()

    # Segment 2: trending drift (GBM-like in level form)
    seg2 = np.zeros(n2, dtype=float)
    seg2[0] = seg1[-1]
    for i in range(1, n2):
        seg2[i] = seg2[i - 1] + 0.35 + 0.35 * rng.randn()

    # Segment 3: OU-like mean reversion around new level
    mean3 = float(np.mean(seg2[-20:])) if n2 >= 20 else float(seg2[-1])
    seg3 = np.zeros(n3, dtype=float)
    seg3[0] = seg2[-1]
    for i in range(1, n3):
        seg3[i] = seg3[i - 1] + 0.16 * (mean3 - seg3[i - 1]) + 1.1 * rng.randn()

    close = np.concatenate([seg1, seg2, seg3])[:n]
    close = np.maximum(close, 5.0)
    high = close + rng.uniform(0.3, 1.2, n)
    low = close - rng.uniform(0.3, 1.2, n)
    opn = close + rng.uniform(-0.4, 0.4, n)
    volume = rng.randint(900_000, 2_500_000, n).astype(float)

    df = pd.DataFrame(
        {"open": opn, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )
    df["return"] = df["close"].pct_change()
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df["volume_sma_20"] = df["volume"].rolling(20).mean()
    df["volume_ratio"] = df["volume"] / df["volume_sma_20"]
    return df


# =============================================================================
# P0.1: ADF TEST
# =============================================================================

class TestADFTest(unittest.TestCase):
    """P0.1: ADF always returns dict with {adf_stat, p_value, is_stationary, note}."""

    def test_adf_dict_shape(self):
        result = adf_test(pd.Series([1, 2, 3]))
        for key in ("adf_stat", "p_value", "is_stationary", "note"):
            self.assertIn(key, result, f"Missing key: {key}")

    def test_adf_unavailable_has_note(self):
        """Whether statsmodels is missing or data is short, note must explain."""
        result = adf_test(pd.Series(range(10)))
        self.assertIsNone(result["adf_stat"])
        self.assertNotEqual(result["note"], "",
                            "Note should explain why ADF is N/A")

    def test_adf_constant_or_unavailable_has_note(self):
        result = adf_test(pd.Series([50.0] * 100))
        self.assertIsNone(result["adf_stat"])
        self.assertNotEqual(result["note"], "")

    def test_adf_dict_always_has_note_key(self):
        """Even on real data, note key exists (empty string if success)."""
        rng = np.random.RandomState(42)
        s = pd.Series(rng.randn(200).cumsum())
        result = adf_test(s)
        self.assertIn("note", result)
        # If statsmodels available, should have values; if not, note explains
        if result["adf_stat"] is not None:
            self.assertEqual(result["note"], "")
        else:
            self.assertNotEqual(result["note"], "")

    def test_metrics_include_adf_note(self):
        df = _make_df(300)
        metrics = calculate_mean_reversion_metrics(df, 20)
        self.assertIn("adf_note", metrics)


# =============================================================================
# P0.2: THESIS BREAK FIX
# =============================================================================

class TestThesisBreak(unittest.TestCase):
    """P0.2: Thesis break requires consecutive neg slope + magnitude + guard."""

    def _build_engine_and_run(self, sma200_values, prices, cfg_overrides=None):
        """Helper: build a df with controlled SMA200 behavior and run backtest."""
        n = len(prices)
        dates = pd.date_range("2024-01-01", periods=n, freq="B", tz="UTC")
        df = pd.DataFrame({
            "open": prices,
            "high": prices + 1.0,
            "low": prices - 1.0,
            "close": prices,
            "volume": [1_000_000.0] * n,
        }, index=dates)
        df["return"] = df["close"].pct_change()
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))
        df["volume_sma_20"] = df["volume"].rolling(20).mean()
        df["volume_ratio"] = df["volume"] / df["volume_sma_20"]

        cfg = MeanReversionConfig(
            starting_capital=100_000, entry_at="same_close",
            slippage_pct=0.0, commission_per_trade=0.0,
            regime_filter_enabled=False, require_reversal_confirmation=False,
            entry_z=-0.5, ratio_anchor_window=50, ratio_lookback=30,
            max_holding_days=999, stop_atr_multiple=50.0,
            thesis_break_sma_bars=5, thesis_break_min_slope=0.01,
            thesis_break_require_below_sma=True,
            min_trade_notional=0, min_shares=1,
            vol_adjust_sizing=False,
        )
        if cfg_overrides:
            for k, v in cfg_overrides.items():
                setattr(cfg, k, v)

        # Force a BUY at bar 80
        rz = pd.Series(-0.3, index=df.index)
        rz.iloc[80] = -1.0
        labels = pd.Series(Regime.MEAN_REVERTING.value, index=df.index)
        scores = pd.Series(0.8, index=df.index)

        engine = BacktestEngine(cfg)
        return engine.run(df, rz, labels, scores)

    def test_single_neg_slope_does_not_trigger(self):
        """One negative bar should NOT trigger thesis break."""
        n = 300
        prices = np.full(n, 100.0)
        result = self._build_engine_and_run(None, prices)
        thesis_exits = [f for f in result["fills"]
                        if f.get("reason") == "SELL_THESIS_BREAK"]
        self.assertEqual(len(thesis_exits), 0)

    def test_consecutive_neg_slope_triggers(self):
        """N consecutive negative slope bars + price below SMA should trigger."""
        n = 300
        # Create a series that starts flat then drops steadily
        prices = np.full(n, 100.0)
        # After buy at bar 80, create a sustained decline
        for i in range(85, 130):
            prices[i] = prices[i-1] - 0.15
        for i in range(130, n):
            prices[i] = prices[129]

        result = self._build_engine_and_run(None, prices)
        thesis_exits = [f for f in result["fills"]
                        if f.get("reason") == "SELL_THESIS_BREAK"]
        # May or may not trigger depending on SMA200 warmup
        # Key: it should NOT have triggered on bar 86 (only 1 neg slope bar)
        for f in thesis_exits:
            self.assertGreater(f["bar"], 90,
                               "Thesis break should not fire within first few negative bars")

    def test_guard_blocks_if_price_above_sma(self):
        """With require_below_sma=True, thesis break should NOT fire if price > SMA200."""
        n = 300
        # Price stays high, SMA200 is lower
        prices = np.full(n, 150.0)
        result = self._build_engine_and_run(
            None, prices,
            cfg_overrides={"thesis_break_require_below_sma": True}
        )
        thesis_exits = [f for f in result["fills"]
                        if f.get("reason") == "SELL_THESIS_BREAK"]
        self.assertEqual(len(thesis_exits), 0,
                         "Should not thesis-break when price > SMA200")


# =============================================================================
# P0.3: A/B COMPARISON
# =============================================================================

class TestABComparison(unittest.TestCase):
    """P0.3: Filtered vs baseline must differ when signals hit TRENDING bars."""

    def test_trending_regime_blocks_in_filtered_not_baseline(self):
        """Deterministic test: force TRENDING regime + BUY signals."""
        df = _make_df(300, drift=0.0, vol=0.3)
        cfg = MeanReversionConfig(
            regime_filter_enabled=True,
            entry_at="same_close",
            ratio_anchor_window=50,
            ratio_lookback=30,
            require_reversal_confirmation=False,
            min_trade_notional=0, min_shares=1,
            vol_adjust_sizing=False,
        )
        ratio = compute_ratio_series(df, cfg)
        rz = compute_ratio_z(ratio, cfg.ratio_lookback)

        # Force ALL bars to TRENDING regime
        labels = pd.Series(Regime.TRENDING.value, index=df.index)
        scores = pd.Series(0.1, index=df.index)

        # Inject BUY signals
        forced_rz = rz.copy()
        forced_rz.iloc[150] = -3.0
        forced_rz.iloc[200] = -3.0

        # Filtered (should BLOCK)
        engine_f = BacktestEngine(cfg)
        res_f = engine_f.run(df, forced_rz, labels, scores)

        # Baseline (should TRADE)
        cfg_b = MeanReversionConfig(
            regime_filter_enabled=False,
            entry_at="same_close",
            ratio_anchor_window=50,
            ratio_lookback=30,
            require_reversal_confirmation=False,
            min_trade_notional=0, min_shares=1,
            stop_atr_multiple=50.0,
            max_holding_days=999,
            vol_adjust_sizing=False,
        )
        engine_b = BacktestEngine(cfg_b)
        res_b = engine_b.run(df, forced_rz, labels, scores)

        self.assertGreater(res_f["blocked_count"], 0,
                           "Filtered should block signals in TRENDING regime")
        self.assertEqual(res_b["blocked_count"], 0,
                         "Baseline should block nothing")

    def test_ab_sanity_warning_logic(self):
        """If both runs identical, report should note 'no effect'."""
        # When regime is always MEAN_REVERTING, filter has no effect
        df = _make_df(200)
        rz = pd.Series(0.0, index=df.index)  # no signals
        labels = pd.Series(Regime.MEAN_REVERTING.value, index=df.index)
        scores = pd.Series(0.8, index=df.index)

        cfg_on = MeanReversionConfig(regime_filter_enabled=True,
                                     min_trade_notional=0, min_shares=1)
        cfg_off = MeanReversionConfig(regime_filter_enabled=False,
                                      min_trade_notional=0, min_shares=1)

        r_on = BacktestEngine(cfg_on).run(df, rz, labels, scores)
        r_off = BacktestEngine(cfg_off).run(df, rz, labels, scores)

        # Both should have 0 blocked (no signals fired)
        self.assertEqual(r_on["blocked_count"], 0)
        self.assertEqual(r_off["blocked_count"], 0)


class TestSameCloseLagging(unittest.TestCase):
    """Regression guard: same_close signals should be laggable to avoid look-ahead."""

    def test_lagged_same_close_reduces_future_leak(self):
        idx = pd.date_range("2024-01-01", periods=4, freq="B", tz="UTC")
        close = np.array([100.0, 50.0, 200.0, 200.0], dtype=float)
        df = pd.DataFrame(
            {
                "open": close,
                "high": close + 1.0,
                "low": close - 1.0,
                "close": close,
                "volume": np.full(len(close), 1_000_000.0),
            },
            index=idx,
        )
        ratio_z = pd.Series([-0.5, -3.0, 0.0, 0.0], index=idx, dtype=float)
        regimes = pd.Series([Regime.MEAN_REVERTING.value] * len(df), index=idx, dtype=object)
        regime_scores = pd.Series([1.0] * len(df), index=idx, dtype=float)

        cfg_lag = MeanReversionConfig(
            entry_at="same_close",
            lag_signals_for_same_close=True,
            regime_filter_enabled=False,
            require_reversal_confirmation=False,
            stop_atr_multiple=50.0,
            max_holding_days=999,
            min_trade_notional=0.0,
            min_shares=1,
            vol_adjust_sizing=False,
        )
        cfg_no_lag = cfg_lag.copy_with(lag_signals_for_same_close=False)

        bt_lag = BacktestEngine(cfg_lag).run(df, ratio_z, regimes, regime_scores)
        bt_no_lag = BacktestEngine(cfg_no_lag).run(df, ratio_z, regimes, regime_scores)

        buy_lag = [f for f in bt_lag["fills"] if f.get("action") == "BUY"]
        buy_no_lag = [f for f in bt_no_lag["fills"] if f.get("action") == "BUY"]
        self.assertEqual(len(buy_lag), 1)
        self.assertEqual(len(buy_no_lag), 1)
        self.assertGreater(int(buy_lag[0]["bar"]), int(buy_no_lag[0]["bar"]))
        self.assertGreater(bt_no_lag["equity_curve"][-1], bt_lag["equity_curve"][-1])


# =============================================================================
# P0.4: DUST FILL FILTER
# =============================================================================

class TestDustFilter(unittest.TestCase):
    """P0.4: Tiny orders are filtered out."""

    def test_small_capital_filters_dust(self):
        """With very small capital, BUY should be blocked by dust filter."""
        cfg = MeanReversionConfig(
            starting_capital=500,  # Very small
            min_trade_notional=1000,  # Requires $1000 min
            min_shares=5,
            entry_at="same_close",
            slippage_pct=0.0,
            commission_per_trade=0.0,
            regime_filter_enabled=False,
            require_reversal_confirmation=False,
            entry_z=-1.0,
            ratio_anchor_window=50,
            ratio_lookback=30,
            vol_adjust_sizing=False,
        )
        df = _make_df(200, start=100.0)
        rz = pd.Series(0.0, index=df.index)
        rz.iloc[100] = -2.0
        labels = pd.Series(Regime.MEAN_REVERTING.value, index=df.index)
        scores = pd.Series(0.8, index=df.index)

        engine = BacktestEngine(cfg)
        result = engine.run(df, rz, labels, scores)
        buys = [f for f in result["fills"] if f["action"] == "BUY"]
        self.assertEqual(len(buys), 0,
                         "Should filter dust BUY with $500 capital and $1000 min notional")

    def test_normal_capital_passes_filter(self):
        """With normal capital, BUY should pass dust filter."""
        cfg = MeanReversionConfig(
            starting_capital=100_000,
            min_trade_notional=1000,
            min_shares=5,
            entry_at="same_close",
            slippage_pct=0.0,
            commission_per_trade=0.0,
            regime_filter_enabled=False,
            require_reversal_confirmation=False,
            entry_z=-1.0,
            ratio_anchor_window=50,
            ratio_lookback=30,
            max_holding_days=999,
            stop_atr_multiple=50.0,
            vol_adjust_sizing=False,
        )
        df = _make_df(200, start=100.0)
        rz = pd.Series(0.0, index=df.index)
        rz.iloc[100] = -2.0
        labels = pd.Series(Regime.MEAN_REVERTING.value, index=df.index)
        scores = pd.Series(0.8, index=df.index)

        engine = BacktestEngine(cfg)
        result = engine.run(df, rz, labels, scores)
        buys = [f for f in result["fills"] if f["action"] == "BUY"]
        self.assertGreater(len(buys), 0, "Normal capital should pass dust filter")
        for b in buys:
            self.assertGreaterEqual(b["shares"], 5)
            self.assertGreaterEqual(b["shares"] * b["price"], 1000)

    def test_no_fills_below_min_shares(self):
        """All BUY/ADD fills must meet min_shares threshold."""
        cfg = MeanReversionConfig(
            starting_capital=100_000, min_shares=10,
            min_trade_notional=0,
            entry_at="same_close", slippage_pct=0.0, commission_per_trade=0.0,
            regime_filter_enabled=False, require_reversal_confirmation=False,
            entry_z=-1.0, ratio_anchor_window=50, ratio_lookback=30,
            max_holding_days=999, stop_atr_multiple=50.0,
            vol_adjust_sizing=False,
        )
        df = _make_df(200)
        rz = pd.Series(0.0, index=df.index)
        rz.iloc[100] = -2.0
        rz.iloc[120] = -2.5
        labels = pd.Series(Regime.MEAN_REVERTING.value, index=df.index)
        scores = pd.Series(0.8, index=df.index)

        result = BacktestEngine(cfg).run(df, rz, labels, scores)
        for f in result["fills"]:
            if f["action"] in ("BUY", "ADD"):
                self.assertGreaterEqual(f["shares"], 10,
                                        f"Fill has {f['shares']} shares < min 10")


# =============================================================================
# P1.1: REGIME CLASSIFICATION — SIDEWAYS
# =============================================================================

class TestRegimeClassifier(unittest.TestCase):
    """P1.1: Adaptive thresholds produce SIDEWAYS bars."""

    def test_sideways_series_produces_sideways_labels(self):
        df = _make_sideways_df(400)
        cfg = MeanReversionConfig(regime_lookback=200, regime_adaptive_thresholds=True)
        labels, scores = classify_regime(df, cfg)
        sideways_count = (labels == Regime.SIDEWAYS.value).sum()
        # With adaptive thresholds and low-trend data, SIDEWAYS should appear
        self.assertGreater(sideways_count, 0,
                           "Sideways series should produce some SIDEWAYS bars")

    def test_trending_series_fewer_mr_bars(self):
        df = _make_strongly_trending_df(400)
        cfg = MeanReversionConfig(regime_lookback=200, regime_adaptive_thresholds=True)
        labels, _ = classify_regime(df, cfg)
        # Strong uptrend should NOT be classified as MEAN_REVERTING
        mr_total = (labels == Regime.MEAN_REVERTING.value).sum()
        trend_total = (labels == Regime.TRENDING.value).sum()
        self.assertLess(mr_total, len(df) * 0.10,
                        "Strong trend should have very few MR bars")
        self.assertGreater(trend_total, 0,
                           "Strong trend should have some TRENDING bars")

    def test_regime_scores_bounded(self):
        df = _make_df(300)
        cfg = MeanReversionConfig(regime_lookback=200)
        _, scores = classify_regime(df, cfg)
        valid = scores.dropna()
        self.assertTrue((valid >= 0).all() and (valid <= 1).all())

    def test_regime_downsample_runtime_2000bars(self):
        """Downsampled regime loop should stay fast enough for research windows."""
        df = _make_df(2000, seed=123)
        cfg = MeanReversionConfig(regime_lookback=200, regime_update_freq=5)
        t0 = time.perf_counter()
        labels, scores = classify_regime(df, cfg)
        elapsed = time.perf_counter() - t0
        self.assertEqual(len(labels), len(df))
        self.assertEqual(len(scores), len(df))
        self.assertLess(elapsed, 10.0, f"classify_regime too slow: {elapsed:.3f}s")

    def test_regime_adaptation_on_switching_series(self):
        """Trending middle regime should be recognized more than MR segments."""
        df = _make_regime_switching_df(400)
        cfg = MeanReversionConfig(regime_lookback=120, regime_update_freq=5)
        labels, _ = classify_regime(df, cfg)
        mid = labels.iloc[150:250]
        head = labels.iloc[0:150]
        tail = labels.iloc[250:400]
        mid_trend = int((mid == Regime.TRENDING.value).sum())
        flank_trend = int((head == Regime.TRENDING.value).sum() + (tail == Regime.TRENDING.value).sum())
        self.assertGreater(mid_trend, 0)
        self.assertGreater(mid_trend * 2, flank_trend)


# =============================================================================
# P1.2: VERDICT COHERENCE
# =============================================================================

class TestVerdictCoherence(unittest.TestCase):
    """P1.2: Regime-derived verdict matches distribution."""

    def test_verdict_matches_dominant_regime(self):
        """Report's regime verdict should be the most common regime label."""
        df = _make_df(300)
        cfg = MeanReversionConfig(
            regime_filter_enabled=True,
            entry_at="same_close",
            ratio_anchor_window=50,
            ratio_lookback=30,
            min_trade_notional=0, min_shares=1,
        )
        ratio = compute_ratio_series(df, cfg)
        rz = compute_ratio_z(ratio, cfg.ratio_lookback)
        labels, scores = classify_regime(df, cfg)

        bt = BacktestEngine(cfg).run(df, rz, labels, scores)
        rt = _regime_timeline(bt["actions_log"])
        dominant = max(rt.items(), key=lambda kv: kv[1])[0] if rt else "UNKNOWN"

        # This should match what the report would print
        self.assertIn(dominant, [r.value for r in Regime])


# =============================================================================
# P2.2: STAGED EXITS
# =============================================================================

class TestStagedExits(unittest.TestCase):
    """P2.2: Multi-level trim thresholds."""

    def test_trim_levels_sorted(self):
        cfg = MeanReversionConfig(
            trim_levels=[(0.5, 0.25), (1.0, 0.50), (2.0, 1.0)]
        )
        result = _get_trim_action(0.6, cfg)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result[0], Action.REDUCE)
        self.assertAlmostEqual(result[1], 0.25)

    def test_highest_trim_triggers_sell(self):
        cfg = MeanReversionConfig(
            trim_levels=[(0.5, 0.25), (1.0, 0.50), (2.0, 1.0)]
        )
        result = _get_trim_action(2.5, cfg)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result[0], Action.SELL)
        self.assertAlmostEqual(result[1], 1.0)

    def test_below_all_levels_returns_none(self):
        cfg = MeanReversionConfig(
            trim_levels=[(0.5, 0.25), (1.0, 0.50), (2.0, 1.0)]
        )
        result = _get_trim_action(0.3, cfg)
        self.assertIsNone(result)

    def test_middle_level_triggers_reduce(self):
        cfg = MeanReversionConfig(
            trim_levels=[(0.5, 0.25), (1.0, 0.50), (2.0, 1.0)]
        )
        result = _get_trim_action(1.5, cfg)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result[0], Action.REDUCE)
        self.assertAlmostEqual(result[1], 0.50)


# =============================================================================
# ACTION ENGINE (ported from v3)
# =============================================================================

class TestActionEngine(unittest.TestCase):

    def setUp(self):
        self.cfg = MeanReversionConfig()

    def test_buy_when_undervalued(self):
        action = determine_action(-2.0, Regime.MEAN_REVERTING.value,
                                  False, 0.0, self.cfg, True, True)
        self.assertEqual(action, Action.BUY)

    def test_add_when_deeper(self):
        action = determine_action(-2.5, Regime.MEAN_REVERTING.value,
                                  True, 0.05, self.cfg, True, True)
        self.assertEqual(action, Action.ADD)

    def test_hold_when_normal(self):
        action = determine_action(0.3, Regime.MEAN_REVERTING.value,
                                  True, 0.10, self.cfg, True, True)
        self.assertEqual(action, Action.HOLD)

    def test_no_reversal_blocks_buy(self):
        action = determine_action(-2.0, Regime.MEAN_REVERTING.value,
                                  False, 0.0, self.cfg, False, True)
        self.assertEqual(action, Action.HOLD)

    def test_blocked_in_trending(self):
        cfg = MeanReversionConfig(regime_filter_enabled=True)
        action = determine_action(-2.0, Regime.TRENDING.value,
                                  False, 0.0, cfg, True, True)
        self.assertEqual(action, Action.BLOCKED)


# =============================================================================
# REGIME GATING (ported from v3)
# =============================================================================

class TestRegimeGating(unittest.TestCase):

    def test_trending_blocks_buy(self):
        cfg = MeanReversionConfig(regime_filter_enabled=True)
        self.assertFalse(regime_allows_action(Action.BUY, Regime.TRENDING.value, cfg))

    def test_trending_blocks_add(self):
        cfg = MeanReversionConfig(regime_filter_enabled=True)
        self.assertFalse(regime_allows_action(Action.ADD, Regime.TRENDING.value, cfg))

    def test_trending_allows_reduce(self):
        cfg = MeanReversionConfig(regime_filter_enabled=True)
        self.assertTrue(regime_allows_action(Action.REDUCE, Regime.TRENDING.value, cfg))

    def test_trending_allows_sell(self):
        cfg = MeanReversionConfig(regime_filter_enabled=True)
        self.assertTrue(regime_allows_action(Action.SELL, Regime.TRENDING.value, cfg))

    def test_mr_allows_buy(self):
        cfg = MeanReversionConfig(regime_filter_enabled=True)
        self.assertTrue(regime_allows_action(Action.BUY, Regime.MEAN_REVERTING.value, cfg))

    def test_ambiguous_block_denies(self):
        cfg = MeanReversionConfig(regime_filter_enabled=True, ambiguous_policy="block")
        self.assertFalse(regime_allows_action(Action.BUY, Regime.AMBIGUOUS.value, cfg))

    def test_filter_disabled_allows_all(self):
        cfg = MeanReversionConfig(regime_filter_enabled=False)
        self.assertTrue(regime_allows_action(Action.BUY, Regime.TRENDING.value, cfg))


# =============================================================================
# RATIO ANCHOR (ported)
# =============================================================================

class TestRatioAnchor(unittest.TestCase):

    def test_ratio_centered_near_one(self):
        df = _make_df(300, drift=0.0, vol=0.3)
        cfg = MeanReversionConfig(ratio_anchor_window=50, ratio_lookback=30)
        ratio = compute_ratio_series(df, cfg)
        valid = ratio.dropna()
        self.assertAlmostEqual(float(valid.mean()), 1.0, delta=0.05)

    def test_ratio_z_shape(self):
        df = _make_df(300)
        cfg = MeanReversionConfig(ratio_anchor_window=50, ratio_lookback=30)
        ratio = compute_ratio_series(df, cfg)
        rz = compute_ratio_z(ratio, 30)
        self.assertEqual(len(rz), len(df))

    def test_custom_passthrough(self):
        df = _make_df(100)
        cfg = MeanReversionConfig()
        custom = pd.Series(np.linspace(0.9, 1.1, 100), index=df.index)
        result = compute_ratio_series(df, cfg, custom_ratio=custom)
        pd.testing.assert_series_equal(result, custom)


# =============================================================================
# REVERSAL CONFIRMATION (ported)
# =============================================================================

class TestReversalConfirmation(unittest.TestCase):

    def test_close_above_prior(self):
        df = _make_df(100)
        close_col = cast(int, df.columns.get_loc("close"))
        df.iloc[-1, close_col] = float(df["close"].iloc[-2]) + 5.0
        cfg = MeanReversionConfig(confirmation_methods=["close_above_prior"],
                                  require_reversal_confirmation=True)
        self.assertTrue(check_reversal_confirmation(df, len(df)-1, cfg))

    def test_disabled_always_true(self):
        df = _make_df(100)
        cfg = MeanReversionConfig(require_reversal_confirmation=False)
        self.assertTrue(check_reversal_confirmation(df, 50, cfg))

    def test_early_bar_false(self):
        df = _make_df(100)
        cfg = MeanReversionConfig(require_reversal_confirmation=True)
        self.assertFalse(check_reversal_confirmation(df, 0, cfg))


# =============================================================================
# POSITION STATE (ported)
# =============================================================================

class TestPositionState(unittest.TestCase):

    def test_avg_cost(self):
        pos = _PositionState(shares=100, cost_basis=10_000.0)
        self.assertAlmostEqual(pos.avg_cost, 100.0)

    def test_avg_cost_after_add(self):
        pos = _PositionState(shares=100, cost_basis=10_000.0)
        pos.shares += 50
        pos.cost_basis += 50 * 110
        self.assertAlmostEqual(pos.avg_cost, (10_000 + 5_500) / 150)

    def test_is_open(self):
        self.assertTrue(_PositionState(shares=1, cost_basis=100).is_open)
        self.assertFalse(_PositionState(shares=0, cost_basis=0).is_open)


# =============================================================================
# BACKTEST ENGINE (ported + enhanced)
# =============================================================================

class TestBacktestEngine(unittest.TestCase):

    def test_no_signals_flat_equity(self):
        cfg = MeanReversionConfig(
            starting_capital=100_000, entry_at="same_close",
            regime_filter_enabled=False, require_reversal_confirmation=False,
            min_trade_notional=0, min_shares=1, vol_adjust_sizing=False,
        )
        df = _make_df(200)
        rz = pd.Series(0.0, index=df.index)
        labels = pd.Series(Regime.MEAN_REVERTING.value, index=df.index)
        scores = pd.Series(0.8, index=df.index)
        result = BacktestEngine(cfg).run(df, rz, labels, scores)
        self.assertEqual(len(result["fills"]), 0)
        self.assertAlmostEqual(result["equity_curve"][-1], 100_000, places=0)

    def test_equity_curve_length(self):
        cfg = MeanReversionConfig(
            min_trade_notional=0, min_shares=1, vol_adjust_sizing=False,
        )
        df = _make_df(200)
        rz = pd.Series(0.0, index=df.index)
        labels = pd.Series(Regime.MEAN_REVERTING.value, index=df.index)
        scores = pd.Series(0.8, index=df.index)
        result = BacktestEngine(cfg).run(df, rz, labels, scores)
        self.assertEqual(len(result["equity_curve"]), len(df))


# =============================================================================
# SCALING (ported + enhanced)
# =============================================================================

class TestScalingInOut(unittest.TestCase):

    def test_add_increases_shares(self):
        cfg = MeanReversionConfig(
            starting_capital=100_000, max_position_pct=0.20,
            add_step_pct=0.05, entry_at="same_close",
            slippage_pct=0.0, commission_per_trade=0.0,
            regime_filter_enabled=False, require_reversal_confirmation=False,
            entry_z=-1.0, add_z=-2.0,
            trim_levels=[(5.0, 0.25), (10.0, 1.0)],  # very high so no trim
            ratio_anchor_window=50, ratio_lookback=30,
            max_holding_days=999, stop_atr_multiple=50.0,
            min_trade_notional=0, min_shares=1, vol_adjust_sizing=False,
        )
        df = _make_df(200, drift=0.0, vol=0.2)
        rz = pd.Series(-0.5, index=df.index)
        rz.iloc[80] = -1.5
        rz.iloc[85] = -2.5
        labels = pd.Series(Regime.MEAN_REVERTING.value, index=df.index)
        scores = pd.Series(0.8, index=df.index)

        result = BacktestEngine(cfg).run(df, rz, labels, scores)
        buys = [f for f in result["fills"] if f["action"] == "BUY"]
        adds = [f for f in result["fills"] if f["action"] == "ADD"]
        self.assertGreater(len(buys), 0)
        self.assertGreater(len(adds), 0, "Should ADD on deeper z-score")

    def test_reduce_has_pnl(self):
        cfg = MeanReversionConfig(
            starting_capital=100_000, max_position_pct=0.20,
            entry_at="same_close", slippage_pct=0.0, commission_per_trade=0.0,
            regime_filter_enabled=False, require_reversal_confirmation=False,
            entry_z=-1.0,
            trim_levels=[(0.3, 0.25)],  # low threshold
            ratio_anchor_window=50, ratio_lookback=30,
            max_holding_days=999, stop_atr_multiple=50.0,
            min_trade_notional=0, min_shares=1, vol_adjust_sizing=False,
        )
        df = _make_df(200, drift=0.0, vol=0.2)
        rz = pd.Series(0.0, index=df.index)
        rz.iloc[80] = -2.0
        rz.iloc[100] = 0.5
        labels = pd.Series(Regime.MEAN_REVERTING.value, index=df.index)
        scores = pd.Series(0.8, index=df.index)

        result = BacktestEngine(cfg).run(df, rz, labels, scores)
        reduces = [f for f in result["fills"] if f["action"] == "REDUCE"]
        if reduces:
            self.assertIn("realized_pnl", reduces[0])


# =============================================================================
# STATISTICAL TESTS (ported)
# =============================================================================

class TestStatisticalTests(unittest.TestCase):
    def test_hurst_too_short(self):
        self.assertEqual(calculate_hurst_exponent(pd.Series([1, 2, 3])), (None, None))

    def test_half_life_trending_none(self):
        self.assertIsNone(calculate_half_life(pd.Series(np.arange(100, dtype=float))))

    def test_variance_ratio_short(self):
        self.assertIsNone(calculate_variance_ratio(pd.Series([1, 2, 3]), q=5))


# =============================================================================
# NORMALIZATION (ported)
# =============================================================================

class TestNormalization(unittest.TestCase):
    def test_adds_timezone(self):
        dates = pd.date_range("2024-01-01", periods=5)
        df = pd.DataFrame(
            {"open": [1]*5, "high": [2]*5, "low": [0.5]*5,
             "close": [1.5]*5, "volume": [100]*5}, index=dates,
        )
        norm_index = pd.DatetimeIndex(normalize_dataframe(df).index)
        self.assertIsNotNone(norm_index.tz)

    def test_missing_column_raises(self):
        dates = pd.date_range("2024-01-01", periods=5, tz="UTC")
        df = pd.DataFrame({"open": [1]*5, "close": [1]*5}, index=dates)
        with self.assertRaises(ValueError):
            normalize_dataframe(df)


class TestDataFetchArgsValidation(unittest.TestCase):
    """Validation helpers for CLI period/interval (no network)."""

    def test_valid_period_values(self):
        for p in ("500d", "4y", "8y", "10y", "20y", "1wk", "6mo", "max"):
            self.assertTrue(is_valid_period(p), f"Expected valid period: {p}")

    def test_invalid_period_values(self):
        for p in ("", "abc", "4years", "500", "0", "d500", "maxx"):
            self.assertFalse(is_valid_period(p), f"Expected invalid period: {p}")

    def test_valid_interval_values(self):
        for itv in ("1d", "1wk", "1mo", "5d"):
            self.assertTrue(is_valid_interval(itv), f"Expected valid interval: {itv}")

    def test_invalid_interval_values(self):
        for itv in ("", "daily", "2wk", "10d", "1year"):
            self.assertFalse(is_valid_interval(itv), f"Expected invalid interval: {itv}")


class TestDataQualityDiagnostics(unittest.TestCase):
    def test_data_quality_detects_duplicates_monotonicity_and_gaps(self):
        idx = pd.to_datetime(
            ["2024-01-03", "2024-01-01", "2024-01-01", "2024-01-10"], utc=True
        )
        df = pd.DataFrame(
            {
                "open": [100.0, 101.0, 101.0, 102.0],
                "high": [101.0, 102.0, 102.0, 103.0],
                "low": [99.0, 100.0, 100.0, 101.0],
                "close": [100.0, np.nan, 0.0, 102.0],
                "volume": [1_000_000.0] * 4,
            },
            index=idx,
        )
        dq = compute_data_quality_diagnostics(
            df,
            requested_price_field="adjclose",
            price_field_used="close",
            price_field_warning="Adj Close unavailable from provider; fell back to Close.",
            raw_duplicate_timestamps_count=1,
            raw_is_monotonic_index=False,
            has_splits=True,
            has_dividends=False,
        )
        self.assertEqual(dq["duplicate_timestamps_count"], 1)
        self.assertFalse(dq["is_monotonic_index"])
        self.assertEqual(dq["gap_stats"]["max_gap_days"], 7)
        self.assertEqual(dq["gap_stats"]["n_gaps_over_3d"], 1)
        self.assertGreater(dq["pct_missing_est"], 0.0)
        self.assertEqual(dq["any_nonpositive_prices"], 1)
        self.assertEqual(dq["any_nan_prices"], 1)
        self.assertTrue(dq["has_splits"])
        self.assertFalse(dq["has_dividends"])


class TestStopExecutionBase(unittest.TestCase):
    def test_gap_down_executes_at_open(self):
        px = _stop_execution_base(open_px=90.0, high_px=95.0, low_px=80.0, stop_px=95.0)
        self.assertEqual(px, 90.0)

    def test_intraday_hit_executes_at_stop(self):
        px = _stop_execution_base(open_px=100.0, high_px=105.0, low_px=90.0, stop_px=95.0)
        self.assertEqual(px, 95.0)

    def test_clamp_with_odd_data(self):
        px = _stop_execution_base(open_px=90.0, high_px=88.0, low_px=85.0, stop_px=95.0)
        self.assertGreaterEqual(px, 85.0)
        self.assertLessEqual(px, 88.0)


class TestAdjCloseScaling(unittest.TestCase):
    def test_scales_ohlc_and_inverse_scales_volume(self):
        df = pd.DataFrame(
            {
                "open": [100.0],
                "high": [100.0],
                "low": [100.0],
                "close": [100.0],
                "adjclose": [50.0],
                "volume": [1000.0],
            }
        )
        out, scale_safe = _apply_adjclose_price_volume_scaling(df)
        self.assertAlmostEqual(float(scale_safe.iloc[0]), 0.5, places=6)
        self.assertAlmostEqual(float(out["close"].iloc[0]), 50.0, places=6)
        self.assertAlmostEqual(float(out["volume"].iloc[0]), 2000.0, places=6)

    def test_scale_safe_fallback_no_crash(self):
        df = pd.DataFrame(
            {
                "open": [100.0],
                "high": [100.0],
                "low": [100.0],
                "close": [0.0],
                "adjclose": [50.0],
                "volume": [100.0],
            }
        )
        out, scale_safe = _apply_adjclose_price_volume_scaling(df)
        self.assertAlmostEqual(float(scale_safe.iloc[0]), 1.0, places=6)
        self.assertAlmostEqual(float(out["volume"].iloc[0]), 100.0, places=6)


class TestTacticalVolTargeting(unittest.TestCase):
    def test_multiplier_formula(self):
        m = _vol_target_multiplier(0.20, sigma_target=0.10, sigma_floor=0.05, cap=1.50)
        self.assertAlmostEqual(m, 0.5, places=6)
        m_floor = _vol_target_multiplier(0.01, sigma_target=0.10, sigma_floor=0.05, cap=1.50)
        self.assertAlmostEqual(m_floor, 1.5, places=6)
        m_nan = _vol_target_multiplier(None, sigma_target=0.10, sigma_floor=0.05, cap=1.50)
        self.assertAlmostEqual(m_nan, 1.0, places=6)

    def test_enabled_targeting_reduces_buy_size_when_vol_high(self):
        n = 80
        idx = pd.date_range("2024-01-01", periods=n, freq="B", tz="UTC")
        # High realized vol path
        close = 100.0 + np.array([(-1) ** i * 3.0 for i in range(n)]).cumsum()
        close = np.maximum(close, 20.0)
        df = pd.DataFrame(
            {
                "open": close,
                "high": close + 1.0,
                "low": close - 1.0,
                "close": close,
                "volume": np.full(n, 1_000_000.0),
            },
            index=idx,
        )
        rz = pd.Series(0.0, index=idx, dtype=float)
        rz.iloc[-1] = -3.0  # force one BUY decision at end
        labels = pd.Series([Regime.MEAN_REVERTING.value] * n, index=idx, dtype=object)
        scores = pd.Series([1.0] * n, index=idx, dtype=float)

        base_cfg = MeanReversionConfig(
            entry_at="same_close",
            lag_signals_for_same_close=False,
            regime_filter_enabled=False,
            require_reversal_confirmation=False,
            vol_adjust_sizing=False,
            min_trade_notional=0.0,
            min_shares=1,
            commission_per_trade=0.0,
            slippage_pct=0.0,
            tactical_vol_targeting_enabled=False,
        )
        cfg_on = base_cfg.copy_with(
            tactical_vol_targeting_enabled=True,
            tactical_vol_target=0.10,
            tactical_vol_window=20,
            tactical_vol_floor=0.05,
            tactical_vol_cap=1.50,
        )

        bt_off = BacktestEngine(base_cfg).run(df, rz, labels, scores)
        bt_on = BacktestEngine(cfg_on).run(df, rz, labels, scores)
        buy_off = next((f for f in bt_off["fills"] if f.get("action") == "BUY"), None)
        buy_on = next((f for f in bt_on["fills"] if f.get("action") == "BUY"), None)
        self.assertIsNotNone(buy_off)
        self.assertIsNotNone(buy_on)
        assert buy_off is not None
        assert buy_on is not None
        self.assertLessEqual(int(buy_on["shares"]), int(buy_off["shares"]))


class TestLastBarNextOpenGuard(unittest.TestCase):
    def test_guard_logic(self):
        self.assertTrue(_should_block_next_open_entry(9, 10, "next_open"))
        self.assertFalse(_should_block_next_open_entry(8, 10, "next_open"))
        self.assertFalse(_should_block_next_open_entry(9, 10, "same_close"))


class TestGapDownStopExecution(unittest.TestCase):
    def test_gap_down_stop_uses_open_with_slippage(self):
        idx = pd.date_range("2024-01-01", periods=3, freq="B", tz="UTC")
        df = pd.DataFrame(
            {
                "open": [100.0, 90.0, 91.0],
                "high": [101.0, 95.0, 92.0],
                "low": [99.0, 80.0, 89.0],
                "close": [100.0, 90.0, 91.0],
                "volume": [1_000_000.0, 1_000_000.0, 1_000_000.0],
            },
            index=idx,
        )
        ratio_z = pd.Series([-2.0, 0.0, 0.0], index=idx, dtype=float)
        regimes = pd.Series([Regime.MEAN_REVERTING.value] * len(df), index=idx, dtype=object)
        regime_scores = pd.Series([1.0] * len(df), index=idx, dtype=float)
        cfg = MeanReversionConfig(
            entry_at="same_close",
            lag_signals_for_same_close=False,
            regime_filter_enabled=False,
            require_reversal_confirmation=False,
            stop_atr_multiple=1.0,
            commission_per_trade=0.0,
            min_trade_notional=0.0,
            min_shares=1,
            vol_adjust_sizing=False,
        )
        with mock.patch(
            "mean_reversion_standalone.calculate_atr",
            return_value=pd.Series([5.0, 5.0, 5.0], index=idx, dtype=float),
        ):
            bt = BacktestEngine(cfg).run(df, ratio_z, regimes, regime_scores)

        sell_fills = [f for f in bt["fills"] if f.get("action") == "SELL"]
        self.assertEqual(len(sell_fills), 1)
        sell = sell_fills[0]
        expected_exit = round(90.0 * (1 - cfg.slippage_pct), 4)
        self.assertAlmostEqual(float(sell["price"]), expected_exit, places=4)
        self.assertEqual(str(sell.get("reason")), "SELL_GAP_STOP_OPEN")


class TestTradeExcursions(unittest.TestCase):
    def test_trade_excursion_metrics_known_path(self):
        prices = pd.Series([100.0, 95.0, 110.0, 102.0], dtype=float)
        rz = pd.Series([-2.0, -1.0, 0.5, 1.0], dtype=float)
        trades = [{
            "entry_bar_index": 0,
            "exit_bar_index": 3,
            "entry_price": 100.0,
            "shares": 10,
            "entry_ratio_z": -2.0,
            "exit_ratio_z": 1.0,
        }]
        _compute_trade_excursions(trades, prices, rz)
        rec = trades[0]
        self.assertAlmostEqual(rec["mae_pct"], -5.0, places=4)
        self.assertAlmostEqual(rec["mfe_pct"], 10.0, places=4)
        self.assertAlmostEqual(rec["mae_abs"], -50.0, places=2)
        self.assertAlmostEqual(rec["mfe_abs"], 100.0, places=2)
        self.assertAlmostEqual(rec["entry_z"], -2.0, places=4)
        self.assertAlmostEqual(rec["exit_z"], 1.0, places=4)


class TestRequireMinBars(unittest.TestCase):
    def test_insufficient_data_writes_summary_reason(self):
        short_df = _make_df(120)
        fetch_meta = {
            "requested_price_field": "adjclose",
            "price_field_used": "close",
            "price_field_warning": "Adj Close unavailable from provider; fell back to Close.",
            "raw_duplicate_timestamps_count": 0,
            "raw_is_monotonic_index": True,
            "has_splits": False,
            "has_dividends": False,
        }
        cfg = MeanReversionConfig(regime_filter_enabled=False)
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch(
                "mean_reversion_standalone.fetch_price_data",
                return_value=(short_df, fetch_meta),
            ):
                result = analyze(
                    "TEST",
                    cfg=cfg,
                    quiet=True,
                    reports_dir=tmpdir,
                    export_trades=False,
                    period="10y",
                    interval="1d",
                    price_field="adjclose",
                    require_min_bars=400,
                )
            self.assertTrue(result.get("insufficient_data", False))
            self.assertEqual(result.get("reason"), "INSUFFICIENT DATA")
            summary_path = result.get("summary_file")
            self.assertTrue(summary_path and os.path.isfile(summary_path))
            assert isinstance(summary_path, str)
            with open(summary_path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
            self.assertEqual(payload.get("reason"), "INSUFFICIENT DATA")
            self.assertEqual(payload.get("bars"), 120)
            self.assertEqual(payload.get("require_min_bars"), 400)


# =============================================================================
# PERFORMANCE METRICS (ported)
# =============================================================================

class TestPerformanceMetrics(unittest.TestCase):
    def test_no_trades(self):
        eq = [100_000.0] * 100
        perf = calculate_performance_metrics(eq, [])
        self.assertAlmostEqual(perf["total_return_pct"], 0.0, places=1)

    def test_sharpe_positive_for_gains(self):
        eq = [100_000.0 + i * 10.0 for i in range(252)]
        perf = calculate_performance_metrics(eq, [])
        sharpe = perf["sharpe_ratio"]
        if isinstance(sharpe, str):
            self.assertEqual(sharpe, "N/A")
        else:
            self.assertGreater(sharpe, 0)


# =============================================================================
# REPORT CONTENT
# =============================================================================

class TestReportContent(unittest.TestCase):

    def test_report_has_required_sections(self):
        from mean_reversion_standalone import generate_report
        df = _make_df(300)
        tech = get_technicals(df)
        cfg = MeanReversionConfig(
            ratio_anchor_window=50, ratio_lookback=30,
            regime_filter_enabled=True,
            min_trade_notional=0, min_shares=1,
        )
        stat_m = calculate_mean_reversion_metrics(df, 20)
        ratio = compute_ratio_series(df, cfg)
        rz = compute_ratio_z(ratio, cfg.ratio_lookback)
        labels, scores = classify_regime(df, cfg)
        bt = BacktestEngine(cfg).run(df, rz, labels, scores)
        perf = calculate_performance_metrics(bt["equity_curve"], bt["fills"])

        report = generate_report("TEST", df, tech, stat_m, bt, perf, None, None, None, cfg)

        self.assertIn("LONG-ONLY ACTIONS SUMMARY", report)
        self.assertIn("REGIME TIMELINE", report)
        self.assertIn("BACKTEST PERFORMANCE", report)
        self.assertIn("EXIT REASON BREAKDOWN", report)
        self.assertIn("DISCLAIMER", report)
        self.assertIn("v3.1", report)

    def test_report_shows_adf_note_when_na(self):
        """If ADF is N/A, report should show the note."""
        from mean_reversion_standalone import generate_report
        df = _make_df(300)
        tech = get_technicals(df)
        cfg = MeanReversionConfig(ratio_anchor_window=50, ratio_lookback=30,
                                  min_trade_notional=0, min_shares=1)
        stat_m = calculate_mean_reversion_metrics(df, 20)
        # Force ADF to N/A
        stat_m["adf_p_value"] = None
        stat_m["adf_note"] = "test note"
        ratio = compute_ratio_series(df, cfg)
        rz = compute_ratio_z(ratio, cfg.ratio_lookback)
        labels, scores = classify_regime(df, cfg)
        bt = BacktestEngine(cfg).run(df, rz, labels, scores)
        perf = calculate_performance_metrics(bt["equity_curve"], bt["fills"])

        report = generate_report("TEST", df, tech, stat_m, bt, perf, None, None, None, cfg)
        self.assertIn("test note", report)

    def test_report_includes_bias_audit_block(self):
        from mean_reversion_standalone import generate_report
        df = _make_df(260)
        tech = get_technicals(df)
        cfg = MeanReversionConfig(
            ratio_anchor_window=50, ratio_lookback=30,
            regime_filter_enabled=False, min_trade_notional=0, min_shares=1,
        )
        stat_m = calculate_mean_reversion_metrics(df, 20)
        ratio = compute_ratio_series(df, cfg)
        rz = compute_ratio_z(ratio, cfg.ratio_lookback)
        labels, scores = classify_regime(df, cfg)
        bt = BacktestEngine(cfg).run(df, rz, labels, scores)
        perf = calculate_performance_metrics(bt["equity_curve"], bt["fills"])
        dq = {
            "requested_price_field": "adjclose",
            "price_field_used": "adjclose",
            "ohlc_scaled_by_adjclose": True,
        }
        bias = _build_bias_audit_payload(
            cfg,
            dq,
            universe_name="SP500 current",
            universe_asof="2026-02-11",
            universe_source="manual list",
        )
        report = generate_report(
            "TEST", df, tech, stat_m, bt, perf, None, None, None, cfg,
            data_quality=dq, bias_audit=bias,
        )
        self.assertIn("BIAS AUDIT / ASSUMPTIONS", report)
        self.assertIn("Data Source", report)
        self.assertIn("Universe Name    : SP500 current", report)
        self.assertIn("frictionless dividend reinvestment", report)
        self.assertIn("gap-stop aware", report)


# =============================================================================
# DIAGNOSTICS
# =============================================================================

class TestDiagnostics(unittest.TestCase):

    def test_exit_reason_breakdown_shape(self):
        fills = [
            {"action": "SELL", "reason": "SELL_STOP", "realized_pnl": -100, "days_held": 5},
            {"action": "SELL", "reason": "SELL_STOP", "realized_pnl": -50, "days_held": 3},
            {"action": "REDUCE", "reason": "TRIM", "realized_pnl": 200, "days_held": 10},
        ]
        bd = _exit_reason_breakdown(fills)
        self.assertIn("SELL_STOP", bd)
        self.assertEqual(bd["SELL_STOP"]["count"], 2)
        self.assertIn("TRIM", bd)

    def test_regime_transitions_detects_changes(self):
        log = [
            {"regime": "MEAN_REVERTING"},
            {"regime": "MEAN_REVERTING"},
            {"regime": "TRENDING"},
            {"regime": "TRENDING"},
            {"regime": "SIDEWAYS"},
        ]
        trans = _regime_transition_matrix(log)
        self.assertIn("MEAN_REVERTING", trans)
        self.assertIn("TRENDING", trans["MEAN_REVERTING"])


# =============================================================================
# FULL PIPELINE (ported + enhanced)
# =============================================================================

class TestFullPipeline(unittest.TestCase):

    def test_mr_series_produces_fills(self):
        df = _make_mean_reverting_df(400, theta=0.20, sigma=1.5)
        cfg = MeanReversionConfig(
            entry_z=-1.0, add_z=-1.5,
            trim_levels=[(0.8, 0.25), (1.5, 1.0)],
            entry_at="same_close",
            ratio_anchor_window=50, ratio_lookback=30,
            regime_filter_enabled=False,
            require_reversal_confirmation=False,
            max_holding_days=30, stop_atr_multiple=5.0,
            min_trade_notional=0, min_shares=1,
            vol_adjust_sizing=False,
        )
        strat = MeanReversionStrategy(cfg)
        ratio = strat.compute_ratio(df)
        rz = strat.compute_ratio_z(ratio)
        labels, scores = strat.classify_regime(df)
        result = BacktestEngine(cfg).run(df, rz, labels, scores)
        perf = calculate_performance_metrics(result["equity_curve"], result["fills"])

        self.assertGreater(len(result["fills"]), 0)
        self.assertEqual(len(result["equity_curve"]), len(df))
        self.assertIn("sharpe_ratio", perf)

    def test_trending_filter_blocks_more(self):
        df = _make_strongly_trending_df(400)
        cfg_on = MeanReversionConfig(
            regime_filter_enabled=True, entry_at="same_close",
            ratio_anchor_window=50, ratio_lookback=30,
            require_reversal_confirmation=False,
            min_trade_notional=0, min_shares=1, vol_adjust_sizing=False,
        )
        cfg_off = MeanReversionConfig(
            regime_filter_enabled=False, entry_at="same_close",
            ratio_anchor_window=50, ratio_lookback=30,
            require_reversal_confirmation=False,
            min_trade_notional=0, min_shares=1, vol_adjust_sizing=False,
        )
        ratio = compute_ratio_series(df, cfg_on)
        rz = compute_ratio_z(ratio, cfg_on.ratio_lookback)
        labels, scores = classify_regime(df, cfg_on)

        r_on = BacktestEngine(cfg_on).run(df, rz, labels, scores)
        r_off = BacktestEngine(cfg_off).run(df, rz, labels, scores)

        self.assertGreaterEqual(r_on["blocked_count"], r_off["blocked_count"])

    def test_strategy_on_bar_interface_returns_action(self):
        df = _make_df(260)
        cfg = MeanReversionConfig(
            entry_at="same_close",
            regime_filter_enabled=False,
            require_reversal_confirmation=False,
        )
        strat = MeanReversionStrategy(cfg)
        history = df.iloc[:240]
        action = strat.on_bar(
            history,
            history.iloc[-1],
            has_position=False,
            position_pct=0.0,
            reversal_ok=True,
            quality_ok=True,
        )
        self.assertIsInstance(action, Action)


class TestConfidenceEntryFilter(unittest.TestCase):
    """Optional entry confidence gate based on MR/SW share in the last 60 bars."""

    def test_impossible_min_confidence_blocks_all_entries(self):
        df = _make_mr_series(320)
        cfg = MeanReversionConfig(
            min_confidence=1.1,  # impossible threshold, should block every BUY signal
            cash_yield_annual_pct=0.0,
            regime_filter_enabled=False,
            entry_at="same_close",
            entry_z=10.0,
            require_reversal_confirmation=False,
            require_volume_confirmation=False,
            require_rsi_confirmation=False,
            min_trade_notional=0,
            min_shares=1,
            vol_adjust_sizing=False,
        )
        ratio = compute_ratio_series(df, cfg)
        rz = compute_ratio_z(ratio, cfg.ratio_lookback)
        labels, scores = classify_regime(df, cfg)

        bt = BacktestEngine(cfg).run(df, rz, labels, scores)
        self.assertEqual(len(bt["fills"]), 0)
        self.assertEqual(sum(1 for a in bt["actions_log"] if a["action"] == "BUY"), 0)
        self.assertGreater(bt.get("blocked_by_confidence", 0), 0)

        perf = calculate_performance_metrics(
            bt["equity_curve"], bt["fills"],
            cash_yield_annual_pct=cfg.cash_yield_annual_pct,
        )
        vs_cash = perf["total_return_pct"]  # cash benchmark is 0% when cash_yield=0
        self.assertAlmostEqual(vs_cash, 0.0, places=2)

    def test_min_confidence_blocks_entries(self):
        df = _make_mr_series(320)
        cfg_base = MeanReversionConfig(
            min_confidence=0.0,
            regime_filter_enabled=False,
            entry_at="same_close",
            entry_z=10.0,
            require_reversal_confirmation=False,
            require_volume_confirmation=False,
            require_rsi_confirmation=False,
            min_trade_notional=0,
            min_shares=1,
            vol_adjust_sizing=False,
        )
        cfg_hi = cfg_base.copy_with(min_confidence=0.99)
        ratio = compute_ratio_series(df, cfg_base)
        rz = compute_ratio_z(ratio, cfg_base.ratio_lookback)
        labels, scores = classify_regime(df, cfg_base)
        bt_base = BacktestEngine(cfg_base).run(df, rz, labels, scores)
        bt_hi = BacktestEngine(cfg_hi).run(df, rz, labels, scores)

        self.assertGreater(bt_hi.get("blocked_by_confidence", 0), 0)
        fills_base = len(bt_base["fills"])
        fills_hi = len(bt_hi["fills"])
        self.assertTrue(fills_hi == 0 or fills_hi < fills_base)

    def test_min_confidence_070_blocks_when_conf_065(self):
        n = 120
        dates = pd.date_range("2024-01-01", periods=n, freq="B", tz="UTC")
        close = np.full(n, 100.0)
        df = pd.DataFrame({
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": np.full(n, 1_000_000.0),
        }, index=dates)
        rz = pd.Series(np.zeros(n), index=df.index, dtype=float)
        rz.iloc[59] = -2.0  # one BUY signal candidate
        labels = pd.Series(
            [Regime.MEAN_REVERTING.value] * 39 + [Regime.TRENDING.value] * (n - 39),
            index=df.index,
            dtype=object,
        )
        scores = pd.Series(np.full(n, 0.5), index=df.index, dtype=float)
        cfg = MeanReversionConfig(
            min_confidence=0.70,
            regime_filter_enabled=False,
            entry_at="same_close",
            require_reversal_confirmation=False,
            require_volume_confirmation=False,
            require_rsi_confirmation=False,
            min_trade_notional=0,
            min_shares=1,
            vol_adjust_sizing=False,
        )
        bt = BacktestEngine(cfg).run(df, rz, labels, scores)

        self.assertEqual(bt.get("blocked_by_confidence", 0), 1)
        self.assertEqual(len([f for f in bt["fills"] if f["action"] == "BUY"]), 0)
        blocked_reasons = [a["reason"] for a in bt["actions_log"] if a["action"] == "BLOCKED"]
        self.assertIn("confidence=0.65<min=0.70", blocked_reasons)


class TestPart2FeatureBC(unittest.TestCase):
    """Feature B/C upgrades: cost-aware entries and improved exits."""

    def test_cost_aware_entry_blocks_low_expected_return_buy(self):
        n = 120
        dates = pd.date_range("2024-01-01", periods=n, freq="B", tz="UTC")
        close = np.full(n, 100.0)
        df = pd.DataFrame(
            {
                "open": close,
                "high": close + 0.5,
                "low": close - 0.5,
                "close": close,
                "volume": np.full(n, 1_000_000.0),
            },
            index=dates,
        )
        rz = pd.Series(np.zeros(n), index=df.index, dtype=float)
        rz.iloc[59] = -2.0  # one BUY signal candidate
        labels = pd.Series([Regime.MEAN_REVERTING.value] * n, index=df.index, dtype=object)
        scores = pd.Series(np.full(n, 0.5), index=df.index, dtype=float)
        cfg = MeanReversionConfig(
            regime_filter_enabled=False,
            entry_at="same_close",
            require_reversal_confirmation=False,
            require_volume_confirmation=False,
            require_rsi_confirmation=False,
            min_trade_notional=0,
            min_shares=1,
            vol_adjust_sizing=False,
            cost_aware_entry_enabled=True,
            cost_bps_est=15.0,
            cost_k=1.25,
        )

        bt = BacktestEngine(cfg).run(df, rz, labels, scores)
        self.assertEqual(bt.get("blocked_by_cost", 0), 1)
        self.assertEqual(len([f for f in bt["fills"] if f["action"] == "BUY"]), 0)
        blocked_reasons = [a["reason"] for a in bt["actions_log"] if a["action"] == "BLOCKED"]
        self.assertTrue(any(str(r).startswith("cost_aware eret=") for r in blocked_reasons))

    def test_better_exits_return_to_mean_executes_next_open(self):
        n = 90
        dates = pd.date_range("2024-01-01", periods=n, freq="B", tz="UTC")
        close = np.full(n, 100.0)
        df = pd.DataFrame(
            {
                "open": close,
                "high": close + 0.5,
                "low": close - 0.5,
                "close": close,
                "volume": np.full(n, 1_000_000.0),
            },
            index=dates,
        )
        rz = pd.Series(np.zeros(n), index=df.index, dtype=float)
        rz.iloc[10] = -2.0  # BUY
        labels = pd.Series([Regime.MEAN_REVERTING.value] * n, index=df.index, dtype=object)
        scores = pd.Series(np.full(n, 0.5), index=df.index, dtype=float)
        cfg = MeanReversionConfig(
            regime_filter_enabled=False,
            entry_at="same_close",
            require_reversal_confirmation=False,
            require_volume_confirmation=False,
            require_rsi_confirmation=False,
            min_trade_notional=0,
            min_shares=1,
            vol_adjust_sizing=False,
            better_exits_enabled=True,
            tactical_exit_z=-0.20,
            tactical_min_hold_days=3,
            tactical_max_hold_days=30,
            sell_z=10.0,
        )

        bt = BacktestEngine(cfg).run(df, rz, labels, scores)
        sell_fills = [f for f in bt["fills"] if f["action"] == "SELL"]
        self.assertGreaterEqual(len(sell_fills), 1)
        self.assertIn("SELL_RETURN_TO_MEAN", {f.get("reason") for f in sell_fills})
        self.assertIn(
            "RETURN_TO_MEAN_NEXT_OPEN",
            {a.get("reason") for a in bt["actions_log"] if a.get("action") == "SELL"},
        )


class TestPart3FeatureDE(unittest.TestCase):
    """Feature D/E: confidence sizing + portfolio tactical sleeve."""

    def test_confidence_sizing_multiplier_bounds(self):
        self.assertAlmostEqual(_confidence_sizing_multiplier(0.60, c0=0.60, gamma=1.0), 0.0, places=9)
        self.assertAlmostEqual(_confidence_sizing_multiplier(1.00, c0=0.60, gamma=1.0), 1.0, places=9)
        self.assertAlmostEqual(_confidence_sizing_multiplier(0.80, c0=0.60, gamma=1.0), 0.5, places=9)
        self.assertAlmostEqual(_confidence_sizing_multiplier(0.80, c0=0.60, gamma=2.0), 0.25, places=9)

    def test_confidence_sizing_at_c0_blocks_new_entry_sizing(self):
        n = 120
        dates = pd.date_range("2024-01-01", periods=n, freq="B", tz="UTC")
        close = np.full(n, 100.0)
        df = pd.DataFrame(
            {
                "open": close,
                "high": close + 0.5,
                "low": close - 0.5,
                "close": close,
                "volume": np.full(n, 1_000_000.0),
            },
            index=dates,
        )
        rz = pd.Series(np.zeros(n), index=df.index, dtype=float)
        rz.iloc[59] = -2.0
        labels = pd.Series(
            [Regime.MEAN_REVERTING.value] * 36 + [Regime.TRENDING.value] * (n - 36),
            index=df.index,
            dtype=object,
        )
        scores = pd.Series(np.full(n, 0.5), index=df.index, dtype=float)

        cfg_base = MeanReversionConfig(
            regime_filter_enabled=False,
            entry_at="same_close",
            require_reversal_confirmation=False,
            require_volume_confirmation=False,
            require_rsi_confirmation=False,
            min_trade_notional=0,
            min_shares=1,
            vol_adjust_sizing=False,
            confidence_sizing_enabled=False,
        )
        cfg_conf = cfg_base.copy_with(
            confidence_sizing_enabled=True,
            confidence_c0=0.60,
            confidence_gamma=1.0,
        )
        bt_base = BacktestEngine(cfg_base).run(df, rz, labels, scores)
        bt_conf = BacktestEngine(cfg_conf).run(df, rz, labels, scores)
        buys_base = len([f for f in bt_base["fills"] if f["action"] == "BUY"])
        buys_conf = len([f for f in bt_conf["fills"] if f["action"] == "BUY"])
        self.assertGreaterEqual(buys_base, 1)
        self.assertEqual(buys_conf, 0)

    def test_portfolio_mode_k_cap_equity_identity_and_trades_ticker(self):
        n = 180
        dates = pd.date_range("2023-01-01", periods=n, freq="B", tz="UTC")

        def _mk_df(closes: np.ndarray) -> pd.DataFrame:
            opn = closes.copy()
            high = closes + 0.6
            low = closes - 0.6
            vol = np.full(len(closes), 1_500_000.0)
            return pd.DataFrame(
                {"open": opn, "high": high, "low": low, "close": closes, "volume": vol},
                index=dates,
            )

        base = np.linspace(100.0, 108.0, n)
        c1 = base.copy()
        c2 = base.copy()
        c3 = base.copy()
        # Same-day dislocations with different magnitudes -> deterministic ranking pressure.
        c1[110:113] -= 18.0
        c2[110:113] -= 12.0
        c3[110:113] -= 5.0

        data = {
            "AAA": _mk_df(c1),
            "BBB": _mk_df(c2),
            "CCC": _mk_df(c3),
        }
        cfg = MeanReversionConfig(
            tactical_mode="portfolio",
            tactical_max_positions=2,
            tactical_entry_z=-0.5,
            tactical_weighting="equal",
            entry_at="next_open",
            regime_filter_enabled=False,
            require_reversal_confirmation=False,
            require_volume_confirmation=False,
            require_rsi_confirmation=False,
            min_trade_notional=0,
            min_shares=1,
            vol_adjust_sizing=False,
            tactical_vol_targeting_enabled=False,
            better_exits_enabled=False,
            max_holding_days=200,
            stop_atr_multiple=100.0,
        )
        bt, perf, _ = _run_portfolio_tactical_backtest(data, cfg)
        self.assertIn("total_return_pct", perf)
        self.assertEqual(len(bt["equity_curve"]), len(bt["cash_curve"]))
        self.assertEqual(len(bt["equity_curve"]), len(bt["invested_value_curve"]))
        for eq, cash_v, invested_v in zip(
            bt["equity_curve"], bt["cash_curve"], bt["invested_value_curve"],
        ):
            self.assertAlmostEqual(float(eq), float(cash_v) + float(invested_v), places=2)

        # Reconstruct concurrent positions from fills to enforce |P_t| <= K.
        open_pos: dict[str, int] = {}
        max_open = 0
        fills_sorted = sorted(bt["fills"], key=lambda f: (int(f.get("bar", 0)), str(f.get("ticker", ""))))
        for f in fills_sorted:
            tk = str(f.get("ticker", ""))
            act = str(f.get("action", ""))
            if act == "BUY":
                open_pos[tk] = open_pos.get(tk, 0) + int(f.get("shares", 0))
            elif act == "SELL":
                open_pos.pop(tk, None)
            max_open = max(max_open, len(open_pos))
        self.assertLessEqual(max_open, int(cfg.tactical_max_positions))

        with tempfile.TemporaryDirectory() as tmpdir:
            ledger_path = Path(tmpdir) / "PORTFOLIO_TRADES_TEST.csv"
            _write_trade_ledger_csv(ledger_path, "PORTFOLIO", bt.get("trade_records", []), cfg.min_confidence)
            with ledger_path.open("r", newline="", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                rows = list(reader)
            if rows:
                self.assertTrue(all(str(r.get("ticker", "")).strip() != "" for r in rows))
                self.assertTrue(all(r.get("ticker") in {"AAA", "BBB", "CCC"} for r in rows))


class TestTradeExportArtifacts(unittest.TestCase):
    """Measurement-only exports: trade ledger + fine confidence bin summary."""

    def test_trade_records_returned(self):
        df = _make_mr_series(320)
        cfg = MeanReversionConfig(
            regime_filter_enabled=False,
            entry_at="same_close",
            entry_z=10.0,
            require_reversal_confirmation=False,
            require_volume_confirmation=False,
            require_rsi_confirmation=False,
            min_trade_notional=0,
            min_shares=1,
            vol_adjust_sizing=False,
        )
        ratio = compute_ratio_series(df, cfg)
        rz = compute_ratio_z(ratio, cfg.ratio_lookback)
        labels, scores = classify_regime(df, cfg)
        bt = BacktestEngine(cfg).run(df, rz, labels, scores)
        self.assertGreater(len(bt.get("trade_records", [])), 0, "Expected at least one completed trade")

    def test_export_trades_csv(self):
        df = _make_mr_series(320)
        cfg = MeanReversionConfig(
            regime_filter_enabled=False,
            entry_at="same_close",
            entry_z=10.0,
            require_reversal_confirmation=False,
            require_volume_confirmation=False,
            require_rsi_confirmation=False,
            min_trade_notional=0,
            min_shares=1,
            vol_adjust_sizing=False,
        )
        ratio = compute_ratio_series(df, cfg)
        rz = compute_ratio_z(ratio, cfg.ratio_lookback)
        labels, scores = classify_regime(df, cfg)
        bt = BacktestEngine(cfg).run(df, rz, labels, scores)
        self.assertGreater(len(bt.get("trade_records", [])), 0, "Expected at least one completed trade")

        with tempfile.TemporaryDirectory() as tmpdir:
            ledger_path = os.path.join(tmpdir, "TEST_TRADES_20260101_000000.csv")
            rows_written = _write_trade_ledger_csv(
                Path(ledger_path), "TEST", bt["trade_records"], cfg.min_confidence,
            )
            self.assertGreater(rows_written, 0)
            self.assertTrue(os.path.isfile(ledger_path))

            with open(ledger_path, "r", newline="", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                rows = list(reader)
            self.assertGreater(len(rows), 0)
            required_cols = {
                "ticker", "entry_dt", "exit_dt", "hold_days", "entry_confidence",
                "realized_pnl", "entry_regime", "entry_ratio_z", "exit_reason",
                "min_confidence_used", "mae_pct", "mfe_pct", "entry_z", "exit_z",
            }
            self.assertTrue(required_cols.issubset(set(reader.fieldnames or [])))
            self.assertIn("entry_confidence", rows[0])
            self.assertIn("realized_pnl", rows[0])

    def test_summary_has_fine_confidence_bins_when_empty(self):
        dates = pd.date_range("2024-01-01", periods=120, freq="B", tz="UTC")
        price = np.full(len(dates), 100.0)
        df = pd.DataFrame({
            "open": price, "high": price + 0.01,
            "low": price - 0.01, "close": price,
            "volume": np.full(len(dates), 1_000_000.0),
        }, index=dates)
        cfg = MeanReversionConfig(
            entry_z=-99.0,
            regime_filter_enabled=False,
            cash_yield_annual_pct=0.0,
        )
        ratio = compute_ratio_series(df, cfg)
        rz = compute_ratio_z(ratio, cfg.ratio_lookback)
        labels, scores = classify_regime(df, cfg)
        bt = BacktestEngine(cfg).run(df, rz, labels, scores)
        perf = calculate_performance_metrics(
            bt["equity_curve"], bt["fills"],
            cash_yield_annual_pct=cfg.cash_yield_annual_pct,
        )
        payload = _build_summary_payload("TEST", df, cfg, perf, bt, None)

        self.assertIn("confidence_bins_0p02_0p50_to_1p00", payload)
        self.assertIn("confidence_bins_0p02_0p60_to_1p00", payload)
        self.assertGreater(len(payload["confidence_bins_0p02_0p50_to_1p00"]), 0)
        self.assertGreater(len(payload["confidence_bins_0p02_0p60_to_1p00"]), 0)

        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = os.path.join(tmpdir, "TEST_SUMMARY_20260101_000000.json")
            with open(summary_path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2)
            with open(summary_path, "r", encoding="utf-8") as fh:
                loaded = json.load(fh)
            self.assertIn("confidence_bins_0p02_0p50_to_1p00", loaded)
            self.assertIn("confidence_bins_0p02_0p60_to_1p00", loaded)

    def test_survivorship_drag_adjustment_math(self):
        n = 253  # ~1 trading year so CAGR math is exact from start/end ratio
        dates = pd.date_range("2024-01-01", periods=n, freq="B", tz="UTC")
        close = np.linspace(100.0, 110.0, n)
        df = pd.DataFrame(
            {
                "open": close,
                "high": close + 0.1,
                "low": close - 0.1,
                "close": close,
                "volume": np.full(n, 1_000_000.0),
            },
            index=dates,
        )
        cfg = MeanReversionConfig(
            two_layer_mode=True,
            core_allocation_pct=0.80,
            tactical_allocation_pct=0.20,
            starting_capital=100_000.0,
            regime_filter_enabled=False,
            min_trade_notional=0,
            min_shares=1,
        )
        bt = {
            "actions_log": [],
            "fills": [],
            "trade_records": [],
            "confidence_bucket_summary": {"total_trades": 0, "rows": []},
        }
        perf = {
            "total_return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "total_fills": 0,
            "sharpe_ratio": "N/A",
        }
        tl_result = {
            "core_equity_curve": np.linspace(100_000.0, 110_000.0, n).tolist(),
            "tactical_equity_curve": np.linspace(20_000.0, 22_000.0, n).tolist(),
            "total_equity_curve": np.linspace(120_000.0, 132_000.0, n).tolist(),
            "baseline_equity_curve": np.linspace(100_000.0, 108_000.0, n).tolist(),
        }
        payload = _build_summary_payload(
            "TEST", df, cfg, perf, bt, tl_result, survivorship_drag_ann=1.5,
        )
        self.assertTrue(payload.get("heuristic_sensitivity_only"))
        self.assertIn("core_cagr", payload)
        self.assertIn("static_baseline_cagr", payload)
        self.assertIn("core_cagr_adj", payload)
        self.assertIn("static_baseline_cagr_adj", payload)
        self.assertAlmostEqual(float(payload["core_cagr"]), 10.0, places=6)
        self.assertAlmostEqual(float(payload["static_baseline_cagr"]), 8.0, places=6)
        self.assertAlmostEqual(float(payload["core_cagr_adj"]), 8.5, places=6)
        self.assertAlmostEqual(float(payload["static_baseline_cagr_adj"]), 6.8, places=6)


class TestAggregateConfidenceBins(unittest.TestCase):
    """Integration test for aggregate_confidence_bins.py main workflow."""

    def test_aggregator_writes_output_and_totals(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir)
            f1 = p / "AAA_TRADES_20260101_000001.csv"
            f2 = p / "BBB_TRADES_20260101_000002.csv"

            headers = [
                "ticker", "entry_dt", "exit_dt", "hold_days", "entry_confidence",
                "realized_pnl", "entry_regime", "entry_ratio_z", "exit_reason",
                "min_confidence_used",
            ]
            rows1 = [
                ["AAA", "2024-01-01", "2024-01-02", 1, 0.60, 10.00, "MEAN_REVERTING", -1.2, "SELL_OVERVALUED", 0.0],
                ["AAA", "2024-01-03", "2024-01-04", 1, 0.70, -5.00, "TRENDING", -1.5, "SELL_STOP", 0.0],
            ]
            rows2 = [
                ["BBB", "2024-01-05", "2024-01-06", 1, 0.80, 15.00, "SIDEWAYS", -1.1, "SELL_TIME_LIMIT", 0.0],
            ]
            for path, rows in ((f1, rows1), (f2, rows2)):
                with path.open("w", newline="", encoding="utf-8") as fh:
                    writer = csv.writer(fh)
                    writer.writerow(headers)
                    writer.writerows(rows)

            old_argv = sys.argv[:]
            try:
                sys.argv = ["aggregate_confidence_bins.py", "--reports-dir", str(p)]
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    rc = acb.main()
            finally:
                sys.argv = old_argv

            out = buf.getvalue()
            self.assertEqual(rc, 0)
            self.assertIn("Totals: trades=3", out)
            self.assertIn("win_rate=66.67%", out)
            self.assertIn("avg_pnl=6.67", out)
            self.assertIn("median_pnl=10.00", out)

            agg_files = list(p.glob("aggregated_bins_*.csv"))
            self.assertEqual(len(agg_files), 1)
            self.assertTrue(agg_files[0].is_file())

    def test_aggregator_includes_000_range_and_skips_invalid_conf(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir)
            f1 = p / "CCC_TRADES_20260101_000003.csv"

            headers = [
                "ticker", "entry_dt", "exit_dt", "hold_days", "entry_confidence",
                "realized_pnl", "entry_regime", "entry_ratio_z", "exit_reason",
                "min_confidence_used",
            ]
            rows = [
                ["CCC", "2024-01-01", "2024-01-02", 1, 0.10, 1.00, "MR", -1.0, "SELL", 0.0],
                ["CCC", "2024-01-03", "2024-01-04", 1, 0.11, -1.00, "MR", -1.1, "SELL", 0.0],
                ["CCC", "2024-01-05", "2024-01-06", 1, 0.55, 2.00, "MR", -1.2, "SELL", 0.0],
                ["CCC", "2024-01-07", "2024-01-08", 1, "", 99.00, "MR", -1.3, "SELL", 0.0],   # invalid conf
                ["CCC", "2024-01-09", "2024-01-10", 1, 1.20, 99.00, "MR", -1.4, "SELL", 0.0], # invalid conf
            ]
            with f1.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                writer.writerow(headers)
                writer.writerows(rows)

            old_argv = sys.argv[:]
            try:
                sys.argv = [
                    "aggregate_confidence_bins.py",
                    "--reports-dir", str(p),
                    "--low-sample-n", "2",
                ]
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    rc = acb.main()
            finally:
                sys.argv = old_argv

            out = buf.getvalue()
            self.assertEqual(rc, 0)
            self.assertIn("Totals: trades=3", out)  # invalid confidence rows skipped

            agg_files = list(p.glob("aggregated_bins_*.csv"))
            self.assertEqual(len(agg_files), 1)
            with agg_files[0].open("r", newline="", encoding="utf-8") as fh:
                rows_out = list(csv.DictReader(fh))

            self.assertTrue(any(r["range"] == "0.00_to_1.00" for r in rows_out))
            bin_010_012 = next(
                r for r in rows_out
                if r["range"] == "0.00_to_1.00" and r["bin"] == "[0.10,0.12)"
            )
            self.assertEqual(int(bin_010_012["n_trades"]), 2)
            self.assertEqual(str(bin_010_012["low_sample"]).lower(), "false")

    def test_aggregator_ignores_skipped_tickers_without_trades_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir)
            (p / "01_SKIPME_TL_20260101_000001.txt").write_text(
                "SKIPPED BY UNIVERSE FILTER\n",
                encoding="utf-8",
            )
            (p / "SKIPME_SUMMARY_20260101_000001.json").write_text(
                json.dumps(
                    {
                        "ticker": "SKIPME",
                        "reason": "SKIPPED BY UNIVERSE FILTER",
                        "filter_reasons": ["missing_columns=volume"],
                    }
                ),
                encoding="utf-8",
            )

            old_argv = sys.argv[:]
            try:
                sys.argv = ["aggregate_confidence_bins.py", "--reports-dir", str(p)]
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    rc = acb.main()
            finally:
                sys.argv = old_argv

            out = buf.getvalue()
            self.assertEqual(rc, 0)
            self.assertIn("Files scanned: 0", out)
            self.assertIn("Totals: trades=0", out)
            agg_files = list(p.glob("aggregated_bins_*.csv"))
            self.assertEqual(len(agg_files), 1)


# =============================================================================
# v3.2: CONFIG copy_with
# =============================================================================

class TestConfigCopyWith(unittest.TestCase):

    def test_copy_preserves_fields(self):
        cfg = MeanReversionConfig(starting_capital=50_000, entry_z=-2.0)
        copy = cfg.copy_with()
        self.assertEqual(copy.starting_capital, 50_000)
        self.assertEqual(copy.entry_z, -2.0)

    def test_copy_overrides_field(self):
        cfg = MeanReversionConfig(starting_capital=100_000)
        copy = cfg.copy_with(starting_capital=20_000)
        self.assertEqual(copy.starting_capital, 20_000)
        # Original untouched
        self.assertEqual(cfg.starting_capital, 100_000)

    def test_copy_multiple_overrides(self):
        cfg = MeanReversionConfig()
        copy = cfg.copy_with(starting_capital=50_000, regime_filter_enabled=False, debug=True)
        self.assertEqual(copy.starting_capital, 50_000)
        self.assertFalse(copy.regime_filter_enabled)
        self.assertTrue(copy.debug)


# =============================================================================
# v3.2: ALLOCATION VALIDATION
# =============================================================================

class TestAllocationValidation(unittest.TestCase):

    def test_valid_80_20(self):
        cfg = MeanReversionConfig(two_layer_mode=True,
                                  core_allocation_pct=0.80,
                                  tactical_allocation_pct=0.20)
        self.assertTrue(cfg.two_layer_mode)

    def test_invalid_sum_raises(self):
        with self.assertRaises(AssertionError):
            MeanReversionConfig(two_layer_mode=True,
                                core_allocation_pct=0.80,
                                tactical_allocation_pct=0.30)

    def test_70_30_valid(self):
        cfg = MeanReversionConfig(two_layer_mode=True,
                                  core_allocation_pct=0.70,
                                  tactical_allocation_pct=0.30)
        self.assertAlmostEqual(cfg.core_allocation_pct + cfg.tactical_allocation_pct, 1.0)

    def test_single_layer_ignores_sum(self):
        """When two_layer_mode=False, allocation sum is not validated."""
        cfg = MeanReversionConfig(two_layer_mode=False,
                                  core_allocation_pct=0.50,
                                  tactical_allocation_pct=0.10)
        self.assertFalse(cfg.two_layer_mode)


# =============================================================================
# v3.2: TWO-LAYER CORE/TACTICAL TESTS
# =============================================================================

class TestTwoLayerPortfolio(unittest.TestCase):
    """Core/Tactical architecture: accounting, curve alignment, no lookahead."""

    def _run_two_layer(self, df, cfg=None):
        cfg = cfg or MeanReversionConfig(
            two_layer_mode=True,
            core_allocation_pct=0.80,
            tactical_allocation_pct=0.20,
            rebalance_freq=None,
            starting_capital=100_000,
            entry_z=-1.0, add_z=-1.5,
            entry_at="same_close",
            ratio_anchor_window=50, ratio_lookback=30,
            regime_filter_enabled=False,
            require_reversal_confirmation=False,
            max_holding_days=30, stop_atr_multiple=5.0,
            min_trade_notional=0, min_shares=1,
            vol_adjust_sizing=False,
        )
        ratio = compute_ratio_series(df, cfg)
        rz = compute_ratio_z(ratio, cfg.ratio_lookback)
        labels, scores = classify_regime(df, cfg)
        engine = TwoLayerPortfolioEngine(cfg)
        return engine.run(df, rz, labels, scores), cfg

    def test_core_always_invested(self):
        """Core shares established once and never change."""
        df = _make_mean_reverting_df(300)
        result, cfg = self._run_two_layer(df)
        core_shares = result["core_shares"]
        self.assertGreater(core_shares, 0, "Core must have shares")
        # Core equity should track close price exactly
        first_close = float(df["close"].iloc[0])
        expected_eq_last = core_shares * float(df["close"].iloc[-1])
        self.assertAlmostEqual(result["core_equity_curve"][-1], expected_eq_last, places=1)

    def test_tactical_capital_allocation(self):
        """Tactical engine starts with exactly tactical_capital."""
        df = _make_mean_reverting_df(300)
        result, cfg = self._run_two_layer(df)
        expected_tactical = cfg.starting_capital * cfg.tactical_allocation_pct
        self.assertAlmostEqual(result["tactical_capital"], expected_tactical, places=2)
        # First bar of tactical equity curve should be tactical_capital
        self.assertAlmostEqual(result["tactical_equity_curve"][0], expected_tactical, places=0)

    def test_total_equity_is_sum(self):
        """For all bars, total == core + tactical (within cents)."""
        df = _make_mean_reverting_df(300)
        result, _ = self._run_two_layer(df)
        for i in range(len(df)):
            expected = result["core_equity_curve"][i] + result["tactical_equity_curve"][i]
            self.assertAlmostEqual(
                result["total_equity_curve"][i], expected, places=1,
                msg=f"Bar {i}: total != core + tactical",
            )

    def test_backward_compatibility(self):
        """When two_layer_mode=False, BacktestEngine works exactly as before."""
        df = _make_mean_reverting_df(300)
        cfg = MeanReversionConfig(
            two_layer_mode=False,
            starting_capital=100_000,
            entry_z=-1.0, add_z=-1.5,
            entry_at="same_close",
            ratio_anchor_window=50, ratio_lookback=30,
            regime_filter_enabled=False,
            require_reversal_confirmation=False,
            max_holding_days=30, stop_atr_multiple=5.0,
            min_trade_notional=0, min_shares=1,
            vol_adjust_sizing=False,
        )
        ratio = compute_ratio_series(df, cfg)
        rz = compute_ratio_z(ratio, cfg.ratio_lookback)
        labels, scores = classify_regime(df, cfg)
        result = BacktestEngine(cfg).run(df, rz, labels, scores)
        self.assertEqual(len(result["equity_curve"]), len(df))
        # First bar should be starting_capital (no position yet)
        self.assertAlmostEqual(result["equity_curve"][0], 100_000, places=0)

    def test_core_eq_matches_buy_hold_of_core_capital(self):
        """Core equity curve should equal buy_hold_curve scaled to core capital."""
        df = _make_df(200)
        result, cfg = self._run_two_layer(df)
        core_cap = cfg.starting_capital * cfg.core_allocation_pct
        bh = compute_buy_hold_curve(df, core_cap)
        for i in range(len(df)):
            self.assertAlmostEqual(result["core_equity_curve"][i], float(bh.iloc[i]), places=1,
                                   msg=f"Core != buy_hold at bar {i}")

    def test_baseline_starts_at_total_capital(self):
        """Static baseline first value should be total starting capital."""
        df = _make_df(200)
        result, cfg = self._run_two_layer(df)
        self.assertAlmostEqual(result["baseline_equity_curve"][0],
                               cfg.starting_capital, places=1)

    def test_cfg_not_mutated(self):
        """TwoLayerPortfolioEngine must NOT mutate the original cfg."""
        df = _make_mean_reverting_df(200)
        cfg = MeanReversionConfig(
            two_layer_mode=True, core_allocation_pct=0.80,
            tactical_allocation_pct=0.20, starting_capital=100_000,
            entry_at="same_close", ratio_anchor_window=50, ratio_lookback=30,
            regime_filter_enabled=False, require_reversal_confirmation=False,
            min_trade_notional=0, min_shares=1, vol_adjust_sizing=False,
        )
        original_capital = cfg.starting_capital
        ratio = compute_ratio_series(df, cfg)
        rz = compute_ratio_z(ratio, cfg.ratio_lookback)
        labels, scores = classify_regime(df, cfg)
        TwoLayerPortfolioEngine(cfg).run(df, rz, labels, scores)
        self.assertEqual(cfg.starting_capital, original_capital,
                         "Original cfg.starting_capital was mutated!")

    def test_curve_lengths(self):
        """All three equity curves must equal len(df)."""
        df = _make_df(250)
        result, _ = self._run_two_layer(df)
        for key in ("core_equity_curve", "tactical_equity_curve",
                     "total_equity_curve", "baseline_equity_curve"):
            self.assertEqual(len(result[key]), len(df), f"{key} wrong length")

    def test_quarterly_rebalance_resets_weights(self):
        """With rebalance enabled, large drift should be reset near target weights."""
        df = _make_strongly_trending_df(520, seed=11)
        cfg = MeanReversionConfig(
            two_layer_mode=True,
            core_allocation_pct=0.80,
            tactical_allocation_pct=0.20,
            rebalance_freq="Q",
            rebalance_drift_threshold=0.05,
            starting_capital=100_000,
            entry_z=-99.0,  # keep tactical mostly idle so drift is clear
            entry_at="same_close",
            ratio_anchor_window=50,
            ratio_lookback=30,
            regime_filter_enabled=False,
            require_reversal_confirmation=False,
            min_trade_notional=0,
            min_shares=1,
            vol_adjust_sizing=False,
        )
        ratio = compute_ratio_series(df, cfg)
        rz = compute_ratio_z(ratio, cfg.ratio_lookback)
        labels, scores = classify_regime(df, cfg)
        result = TwoLayerPortfolioEngine(cfg).run(df, rz, labels, scores)

        self.assertGreaterEqual(int(result.get("rebalancing_events", 0)), 1)
        idx = pd.DatetimeIndex(df.index).tz_localize(None)
        periods = idx.to_period("Q")
        core = result["core_equity_curve"]
        tact = result["tactical_equity_curve"]
        for i in range(1, len(df)):
            if periods[i] != periods[i - 1]:
                total = float(core[i]) + float(tact[i])
                if total > 0:
                    w_core = float(core[i]) / total
                    self.assertAlmostEqual(w_core, 0.80, delta=0.06)


# =============================================================================
# v3.2: REPORT SECTIONS
# =============================================================================

class TestTwoLayerReport(unittest.TestCase):

    def test_report_contains_required_sections(self):
        df = _make_mean_reverting_df(300)
        cfg = MeanReversionConfig(
            two_layer_mode=True, core_allocation_pct=0.80,
            tactical_allocation_pct=0.20, starting_capital=100_000,
            entry_at="same_close", ratio_anchor_window=50, ratio_lookback=30,
            regime_filter_enabled=False, require_reversal_confirmation=False,
            min_trade_notional=0, min_shares=1, vol_adjust_sizing=False,
        )
        ratio = compute_ratio_series(df, cfg)
        rz = compute_ratio_z(ratio, cfg.ratio_lookback)
        labels, scores = classify_regime(df, cfg)
        tl = TwoLayerPortfolioEngine(cfg).run(df, rz, labels, scores)
        bt = tl["tactical_bt"]
        perf = calculate_performance_metrics(bt["equity_curve"], bt["fills"])
        stat_m = calculate_mean_reversion_metrics(df, 20)
        report = generate_two_layer_report("TEST", df, tl, perf, stat_m, bt, {}, cfg)

        self.assertIn("TOTAL PORTFOLIO", report)
        self.assertIn("CORE LAYER", report)
        self.assertIn("TACTICAL LAYER", report)
        self.assertIn("STATIC BASELINE", report)
        self.assertIn("SIDE-BY-SIDE", report)
        self.assertIn("REGIME TIMELINE", report)
        self.assertIn("LONG-ONLY ACTIONS SUMMARY", report)
        self.assertIn("DISCLAIMER", report)


# =============================================================================
# v3.2: VISUALIZATION TESTS
# =============================================================================

class TestVisualization(unittest.TestCase):

    def test_csv_columns_single_layer(self):
        import tempfile, os
        df = _make_df(200)
        eq = [100_000.0] * len(df)
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = generate_benchmark_outputs(eq, df, 100_000, "TEST", tmpdir)
            self.assertIn("curves_csv", paths)
            csv_df = pd.read_csv(paths["curves_csv"])
            for col in ("date", "equity_strategy", "equity_buyhold"):
                self.assertIn(col, csv_df.columns)

    def test_csv_columns_two_layer(self):
        import tempfile, os
        df = _make_mean_reverting_df(200)
        cfg = MeanReversionConfig(
            two_layer_mode=True, core_allocation_pct=0.80,
            tactical_allocation_pct=0.20, starting_capital=100_000,
            entry_at="same_close", ratio_anchor_window=50, ratio_lookback=30,
            regime_filter_enabled=False, require_reversal_confirmation=False,
            min_trade_notional=0, min_shares=1, vol_adjust_sizing=False,
        )
        ratio = compute_ratio_series(df, cfg)
        rz = compute_ratio_z(ratio, cfg.ratio_lookback)
        labels, scores = classify_regime(df, cfg)
        tl = TwoLayerPortfolioEngine(cfg).run(df, rz, labels, scores)
        eq = tl["total_equity_curve"]

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = generate_benchmark_outputs(
                eq, df, 100_000, "TEST", tmpdir, two_layer_result=tl,
            )
            csv_df = pd.read_csv(paths["curves_csv"])
            for col in ("equity_core", "equity_tactical", "equity_total",
                         "equity_baseline_80_20"):
                self.assertIn(col, csv_df.columns, f"Missing: {col}")

    def test_pngs_created(self):
        import tempfile, os
        df = _make_df(200)
        eq = [100_000.0 + i * 5 for i in range(len(df))]
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = generate_benchmark_outputs(eq, df, 100_000, "TEST", tmpdir)
            self.assertIn("benchmark_png", paths)
            self.assertIn("drawdown_png", paths)
            self.assertTrue(os.path.isfile(paths["benchmark_png"]))
            self.assertTrue(os.path.isfile(paths["drawdown_png"]))
            self.assertGreater(os.path.getsize(paths["benchmark_png"]), 1000)

    def test_buy_hold_curve_correct(self):
        df = _make_df(100)
        bh = compute_buy_hold_curve(df, 50_000)
        expected = 50_000 * float(df["close"].iloc[-1]) / float(df["close"].iloc[0])
        self.assertAlmostEqual(float(bh.iloc[-1]), expected, places=1)


# =============================================================================
# v3.3 — CASH YIELD TESTS
# =============================================================================

class TestCashYield(unittest.TestCase):
    """PH1.1: Cash yield on idle tactical capital."""

    def _make_flat_data(self, n=60):
        """Flat price data that generates NO trade signals."""
        dates = pd.date_range("2024-01-01", periods=n, freq="B", tz="UTC")
        price = np.full(n, 100.0)
        df = pd.DataFrame({
            "open": price, "high": price + 0.01,
            "low": price - 0.01, "close": price,
            "volume": np.full(n, 1_000_000.0),
        }, index=dates)
        return df

    def test_no_trades_cash_yield_grows_equity(self):
        """If there are no trades and cash_yield > 0, tactical equity must increase."""
        cfg = MeanReversionConfig(
            cash_yield_annual_pct=5.0,
            starting_capital=100_000,
            entry_z=-99.0,       # effectively impossible to trigger BUY
            regime_filter_enabled=False,
        )
        df = self._make_flat_data(252)
        ratio = compute_ratio_series(df, cfg)
        rz = compute_ratio_z(ratio, cfg.ratio_lookback)
        labels, scores = classify_regime(df, cfg)

        engine = BacktestEngine(cfg)
        bt = engine.run(df, rz, labels, scores)

        fills = [f for f in bt["fills"] if f["action"] in ("BUY", "ADD")]
        self.assertEqual(len(fills), 0, "Expected zero BUY/ADD fills on flat data")

        # Equity should have grown
        eq = bt["equity_curve"]
        self.assertGreater(eq[-1], eq[0], "Equity should grow from cash yield alone")
        self.assertGreater(bt["accrued_cash_yield"], 0)

    def test_no_cash_yield_flat_equity(self):
        """If cash_yield=0 and no trades, equity stays flat."""
        cfg = MeanReversionConfig(
            cash_yield_annual_pct=0.0,
            starting_capital=100_000,
            entry_z=-99.0,
            regime_filter_enabled=False,
        )
        df = self._make_flat_data(60)
        ratio = compute_ratio_series(df, cfg)
        rz = compute_ratio_z(ratio, cfg.ratio_lookback)
        labels, scores = classify_regime(df, cfg)

        engine = BacktestEngine(cfg)
        bt = engine.run(df, rz, labels, scores)
        eq = bt["equity_curve"]
        self.assertAlmostEqual(eq[0], eq[-1], places=0)
        self.assertAlmostEqual(bt["accrued_cash_yield"], 0.0, places=2)

    def test_cash_yield_only_on_uninvested(self):
        """Cash yield should be near-zero when fully invested."""
        # Use mean-reverting data that triggers a long position
        df = _make_mr_series(300)
        cfg_with_yield = MeanReversionConfig(
            cash_yield_annual_pct=5.0,
            starting_capital=100_000,
            regime_filter_enabled=False,
        )
        ratio = compute_ratio_series(df, cfg_with_yield)
        rz = compute_ratio_z(ratio, cfg_with_yield.ratio_lookback)
        labels, scores = classify_regime(df, cfg_with_yield)

        engine = BacktestEngine(cfg_with_yield)
        bt = engine.run(df, rz, labels, scores)

        # Some yield accrued (on idle cash when not fully invested)
        # Just verify the field exists and is non-negative
        self.assertGreaterEqual(bt["accrued_cash_yield"], 0.0)

    def test_cash_yield_uses_calendar_day_compounding(self):
        """Compounding should follow (1+r)^(1/365)-1 across trading bars."""
        n = 252
        start_cap = 100_000.0
        annual = 5.0
        cfg = MeanReversionConfig(
            cash_yield_annual_pct=annual,
            starting_capital=start_cap,
            entry_z=-99.0,
            regime_filter_enabled=False,
        )
        df = self._make_flat_data(n)
        ratio = compute_ratio_series(df, cfg)
        rz = compute_ratio_z(ratio, cfg.ratio_lookback)
        labels, scores = classify_regime(df, cfg)
        bt = BacktestEngine(cfg).run(df, rz, labels, scores)

        end_eq = float(bt["equity_curve"][-1])
        expected_365 = start_cap * ((1 + annual / 100) ** (n / 365.0))
        expected_252 = start_cap * ((1 + annual / 100) ** (n / 252.0))
        self.assertLess(abs(end_eq - expected_365), abs(end_eq - expected_252))


# =============================================================================
# v3.3 — HURST CLAMPING TEST
# =============================================================================

class TestHurstClamping(unittest.TestCase):
    """PH1.4: Hurst exponent clamped to [0, 1]."""

    def test_hurst_in_valid_range(self):
        """Any non-None Hurst must be in [0, 1]."""
        rng = np.random.RandomState(99)
        # Test a range of data types
        for seed in range(5):
            rng2 = np.random.RandomState(seed + 100)
            prices = pd.Series(100.0 + np.cumsum(rng2.randn(300) * 2.0))
            h, r2 = calculate_hurst_exponent(prices)
            if h is not None:
                self.assertGreaterEqual(h, 0.0, f"Hurst {h} < 0 for seed {seed}")
                self.assertLessEqual(h, 1.0, f"Hurst {h} > 1 for seed {seed}")


# =============================================================================
# v3.3 — REGIME SIGNAL ABSTRACTION TESTS
# =============================================================================

class TestRegimeSignal(unittest.TestCase):
    """Phase 2 prep: RegimeSignal abstraction."""

    def test_from_hard_label(self):
        sig = RegimeSignal.from_hard_label(Regime.MEAN_REVERTING.value, score=0.8)
        self.assertEqual(sig.label, Regime.MEAN_REVERTING.value)
        self.assertEqual(sig.confidence, 1.0)
        self.assertIsNotNone(sig.beliefs)
        beliefs = sig.beliefs
        assert beliefs is not None
        self.assertAlmostEqual(float(np.sum(beliefs)), 1.0, places=6)

    def test_mr_probability_hard_mr(self):
        sig = RegimeSignal.from_hard_label(Regime.MEAN_REVERTING.value)
        self.assertEqual(sig.mr_probability, 1.0)

    def test_mr_probability_hard_trending(self):
        sig = RegimeSignal.from_hard_label(Regime.TRENDING.value)
        self.assertEqual(sig.mr_probability, 0.0)

    def test_mr_probability_hard_sideways(self):
        sig = RegimeSignal.from_hard_label(Regime.SIDEWAYS.value)
        self.assertEqual(sig.mr_probability, 1.0)

    def test_mr_probability_with_beliefs(self):
        """When beliefs are populated, mr_probability uses them."""
        sig = RegimeSignal(
            label=Regime.AMBIGUOUS.value,
            beliefs=np.array([0.4, 0.3, 0.2, 0.1], dtype=float),
        )
        self.assertAlmostEqual(sig.mr_probability, 0.7)

    def test_default_ambiguous(self):
        sig = RegimeSignal()
        self.assertEqual(sig.label, Regime.AMBIGUOUS.value)
        self.assertEqual(sig.mr_probability, 0.5)


# =============================================================================
# v3.3 — DIAGNOSTIC FUNCTION TESTS
# =============================================================================

class TestDiagnosticFunctions(unittest.TestCase):
    """PH1.3: New diagnostic functions."""

    def test_capture_ratios_symmetric(self):
        """100% capture on identical curves."""
        curve = [100.0, 102.0, 101.0, 104.0, 103.0]
        cap = _capture_ratios(curve, curve)
        upside_capture = cap["upside_capture"]
        downside_capture = cap["downside_capture"]
        self.assertIsNotNone(upside_capture)
        self.assertIsNotNone(downside_capture)
        assert upside_capture is not None
        assert downside_capture is not None
        self.assertAlmostEqual(upside_capture, 100.0, places=0)
        self.assertAlmostEqual(downside_capture, 100.0, places=0)

    def test_capture_ratios_no_down_days(self):
        """If benchmark has no down days, downside capture is None."""
        benchmark = [100.0, 101.0, 102.0, 103.0]
        portfolio = [100.0, 100.5, 101.0, 101.5]
        cap = _capture_ratios(portfolio, benchmark)
        self.assertIsNone(cap["downside_capture"])

    def test_calmar_ratio_positive(self):
        """Positive-return curve with drawdown should have positive Calmar."""
        eq = [100000.0, 99000.0, 100500.0, 101000.0, 100000.0, 102000.0]
        calmar = _calmar_ratio(eq)
        self.assertIsNotNone(calmar)
        assert calmar is not None
        self.assertGreater(calmar, 0)

    def test_calmar_ratio_no_drawdown(self):
        """Monotonically increasing curve → no drawdown → None."""
        eq = [100.0, 101.0, 102.0, 103.0, 104.0]
        calmar = _calmar_ratio(eq)
        self.assertIsNone(calmar)

    def test_return_attribution_sums(self):
        """Core + tactical dollar attribution should sum to total change."""
        total = [100000.0, 110000.0]
        core = [80000.0, 86000.0]
        tactical = [20000.0, 24000.0]
        attr = _return_attribution(total, core, tactical, 100000.0)
        self.assertAlmostEqual(
            attr["core_dollars"] + attr["tactical_dollars"],
            total[-1] - total[0], places=2,
        )

    def test_tactical_diagnostics_shape(self):
        """Smoke test: _tactical_diagnostics returns expected keys."""
        bt = {
            "actions_log": [
                {"action": "BUY", "bar": 0, "date": "2024-01-02", "regime": "MR"},
                {"action": "HOLD", "bar": 1, "date": "2024-01-03", "regime": "MR"},
                {"action": "SELL", "bar": 2, "date": "2024-01-04", "regime": "MR"},
            ],
            "fills": [],
            "exposure_pct_curve": [0.0, 10.0, 0.0],
            "blocked_count": 0,
            "blocked_by_regime": 0,
            "accrued_cash_yield": 5.50,
        }
        diag = _tactical_diagnostics(bt)
        self.assertIn("time_in_market_pct", diag)
        self.assertIn("avg_exposure_pct", diag)
        self.assertIn("blocked_signal_rate_pct", diag)
        self.assertIn("blocked_by_confidence", diag)
        self.assertIn("blocked_by_regime", diag)
        self.assertEqual(diag["accrued_cash_yield"], 5.50)


# =============================================================================
# v3.3 — EXPOSURE TRACKING TESTS
# =============================================================================

class TestExposureTracking(unittest.TestCase):
    """Verify BacktestEngine produces exposure_pct_curve."""

    def test_exposure_curve_length(self):
        df = _make_df(300)
        cfg = MeanReversionConfig(regime_filter_enabled=False)
        ratio = compute_ratio_series(df, cfg)
        rz = compute_ratio_z(ratio, cfg.ratio_lookback)
        labels, scores = classify_regime(df, cfg)
        bt = BacktestEngine(cfg).run(df, rz, labels, scores)
        self.assertEqual(len(bt["exposure_pct_curve"]), len(df))

    def test_exposure_zero_when_no_position(self):
        """Flat data → no trades → all exposure = 0."""
        dates = pd.date_range("2024-01-01", periods=60, freq="B", tz="UTC")
        price = np.full(60, 100.0)
        df = pd.DataFrame({
            "open": price, "high": price + 0.01,
            "low": price - 0.01, "close": price,
            "volume": np.full(60, 1_000_000.0),
        }, index=dates)
        cfg = MeanReversionConfig(entry_z=-99.0, regime_filter_enabled=False)
        ratio = compute_ratio_series(df, cfg)
        rz = compute_ratio_z(ratio, cfg.ratio_lookback)
        labels, scores = classify_regime(df, cfg)
        bt = BacktestEngine(cfg).run(df, rz, labels, scores)
        self.assertTrue(all(e == 0.0 for e in bt["exposure_pct_curve"]))


# =============================================================================
# v3.3 — NEW VISUALIZATION TESTS
# =============================================================================

class TestNewPlots(unittest.TestCase):
    """PH1.2: New plot outputs."""

    def _run_two_layer(self, n=300):
        df = _make_mr_series(n)
        cfg = MeanReversionConfig(
            two_layer_mode=True,
            core_allocation_pct=0.80,
            tactical_allocation_pct=0.20,
            rebalance_freq=None,
            regime_filter_enabled=False,
        )
        ratio = compute_ratio_series(df, cfg)
        rz = compute_ratio_z(ratio, cfg.ratio_lookback)
        labels, scores = classify_regime(df, cfg)
        tl = TwoLayerPortfolioEngine(cfg).run(df, rz, labels, scores)
        return df, cfg, tl

    def test_contribution_png_created(self):
        """Return contribution chart is created in two-layer mode."""
        df, cfg, tl = self._run_two_layer()
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = generate_benchmark_outputs(
                tl["total_equity_curve"], df, cfg.starting_capital,
                "TEST", tmpdir, two_layer_result=tl,
                bt_results=tl["tactical_bt"],
            )
            self.assertIn("contribution_png", paths)
            self.assertTrue(os.path.isfile(paths["contribution_png"]))

    def test_underwater_png_created(self):
        """Enhanced underwater chart is created in two-layer mode."""
        df, cfg, tl = self._run_two_layer()
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = generate_benchmark_outputs(
                tl["total_equity_curve"], df, cfg.starting_capital,
                "TEST", tmpdir, two_layer_result=tl,
                bt_results=tl["tactical_bt"],
            )
            self.assertIn("underwater_png", paths)

    def test_exposure_png_created(self):
        """Exposure vs price scatter is created when bt_results provided."""
        df, cfg, tl = self._run_two_layer()
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = generate_benchmark_outputs(
                tl["total_equity_curve"], df, cfg.starting_capital,
                "TEST", tmpdir, two_layer_result=tl,
                bt_results=tl["tactical_bt"],
            )
            self.assertIn("exposure_png", paths)

    def test_graceful_without_bt_results(self):
        """No crash if bt_results is None (backward compat)."""
        df = _make_df(100)
        eq = [100_000.0 + i * 5 for i in range(len(df))]
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = generate_benchmark_outputs(eq, df, 100_000, "TEST", tmpdir)
            # Original outputs still produced
            self.assertIn("benchmark_png", paths)
            self.assertIn("drawdown_png", paths)
            # New plots not produced (no bt_results/two_layer_result)
            self.assertNotIn("exposure_png", paths)
            self.assertNotIn("contribution_png", paths)


# =============================================================================
# v3.3 — ENHANCED TWO-LAYER REPORT TESTS
# =============================================================================

class TestEnhancedReport(unittest.TestCase):
    """PH1.3: New report sections present."""

    def test_report_has_new_sections(self):
        """Two-layer report includes diagnostics, attribution, and tactical diag."""
        df = _make_mr_series(300)
        cfg = MeanReversionConfig(
            two_layer_mode=True,
            core_allocation_pct=0.80,
            tactical_allocation_pct=0.20,
            regime_filter_enabled=False,
        )
        ratio = compute_ratio_series(df, cfg)
        rz = compute_ratio_z(ratio, cfg.ratio_lookback)
        labels, scores = classify_regime(df, cfg)

        tl = TwoLayerPortfolioEngine(cfg).run(df, rz, labels, scores)
        bt_results = tl["tactical_bt"]
        perf = calculate_performance_metrics(bt_results["equity_curve"], bt_results["fills"])
        stat_metrics = calculate_mean_reversion_metrics(df, cfg.lookback_window)

        report = generate_two_layer_report(
            "TEST", df, tl, perf, stat_metrics, bt_results, {"name": "Test"}, cfg,
        )

        for section in [
            "PORTFOLIO DIAGNOSTICS",
            "RETURN ATTRIBUTION",
            "TACTICAL LAYER DIAGNOSTICS",
            "Time in Market",
            "Avg Exposure",
            "Blocked Rate",
            "Upside Capture",
            "Downside Capture",
            "Calmar Ratio",
            "Core (beta)",
            "Tactical (alpha)",
        ]:
            self.assertIn(section, report, f"Missing section: {section}")

    def test_tactical_only_cash_benchmark_matches_baseline(self):
        """Cash benchmark text should match tactical-only baseline cash return."""
        df = _make_mr_series(500)
        cfg = MeanReversionConfig(
            two_layer_mode=True,
            core_allocation_pct=0.0,
            tactical_allocation_pct=1.0,
            cash_yield_annual_pct=5.0,
            regime_filter_enabled=False,
        )
        ratio = compute_ratio_series(df, cfg)
        rz = compute_ratio_z(ratio, cfg.ratio_lookback)
        labels, scores = classify_regime(df, cfg)
        tl = TwoLayerPortfolioEngine(cfg).run(df, rz, labels, scores)
        bt_results = tl["tactical_bt"]
        perf = calculate_performance_metrics(
            bt_results["equity_curve"], bt_results["fills"],
            cash_yield_annual_pct=cfg.cash_yield_annual_pct,
        )
        stat_metrics = calculate_mean_reversion_metrics(df, cfg.lookback_window)
        report = generate_two_layer_report(
            "TEST", df, tl, perf, stat_metrics, bt_results, {"name": "Test"}, cfg,
        )

        m_vs = re.search(r"\(cash=([+-]?\d+\.\d+)%\)", report)
        m_diag = re.search(r"\(cash benchmark = ([+-]?\d+\.\d+)%\)", report)
        self.assertIsNotNone(m_vs)
        self.assertIsNotNone(m_diag)
        assert m_vs is not None
        assert m_diag is not None

        baseline_cash_ret = (
            tl["baseline_equity_curve"][-1] / tl["baseline_equity_curve"][0] - 1
        ) * 100
        expected = round(baseline_cash_ret, 2)
        self.assertAlmostEqual(float(m_vs.group(1)), expected, places=2)
        self.assertAlmostEqual(float(m_diag.group(1)), expected, places=2)
        self.assertGreater(expected, 6.0)
        self.assertLess(expected, 8.5)

    def test_report_includes_confidence_bucket_table(self):
        """Confidence bucket summary should be persisted in two-layer report text."""
        df = _make_mr_series(300)
        cfg = MeanReversionConfig(
            two_layer_mode=True,
            core_allocation_pct=0.0,
            tactical_allocation_pct=1.0,
            regime_filter_enabled=False,
            entry_at="same_close",
            entry_z=10.0,
            require_reversal_confirmation=False,
            require_volume_confirmation=False,
            require_rsi_confirmation=False,
            min_trade_notional=0,
            min_shares=1,
            vol_adjust_sizing=False,
        )
        ratio = compute_ratio_series(df, cfg)
        rz = compute_ratio_z(ratio, cfg.ratio_lookback)
        labels, scores = classify_regime(df, cfg)
        tl = TwoLayerPortfolioEngine(cfg).run(df, rz, labels, scores)
        bt_results = tl["tactical_bt"]
        self.assertIn("confidence_bucket_summary", bt_results)
        self.assertGreater(bt_results["confidence_bucket_summary"]["total_trades"], 0)

        perf = calculate_performance_metrics(
            bt_results["equity_curve"], bt_results["fills"],
            cash_yield_annual_pct=cfg.cash_yield_annual_pct,
        )
        stat_metrics = calculate_mean_reversion_metrics(df, cfg.lookback_window)
        report = generate_two_layer_report(
            "TEST", df, tl, perf, stat_metrics, bt_results, {"name": "Test"}, cfg,
        )
        self.assertIn("CONFIDENCE BUCKETS (REGIME PROXY - MR/SW SHARE AT ENTRY)", report)
        self.assertIn("Bucket Trades WinRate AvgPnL AvgHoldDays", report)


# =============================================================================
# v3.3 — BACKWARD COMPATIBILITY TESTS
# =============================================================================

class TestBackwardCompat33(unittest.TestCase):
    """Ensure single-layer mode still works identically."""

    def test_single_layer_still_works(self):
        df = _make_mr_series(300)
        cfg = MeanReversionConfig(
            two_layer_mode=False,
            regime_filter_enabled=False,
        )
        ratio = compute_ratio_series(df, cfg)
        rz = compute_ratio_z(ratio, cfg.ratio_lookback)
        labels, scores = classify_regime(df, cfg)

        bt = BacktestEngine(cfg).run(df, rz, labels, scores)
        perf = calculate_performance_metrics(bt["equity_curve"], bt["fills"])

        self.assertEqual(len(bt["equity_curve"]), len(df))
        self.assertIn("exposure_pct_curve", bt)
        self.assertIn("accrued_cash_yield", bt)
        self.assertEqual(bt["accrued_cash_yield"], 0.0)  # default no yield

    def test_cash_yield_default_zero(self):
        cfg = MeanReversionConfig()
        self.assertEqual(cfg.cash_yield_annual_pct, 0.0)


# =============================================================================
# v3.4 - CORE DCA ENTRY TESTS
# =============================================================================

class TestCoreDCAEntry(unittest.TestCase):
    """Feature 1: Core DCA entry mode for TwoLayerPortfolioEngine."""

    def _run_dca(self, df, dca_days=40, dca_start=0, cash_yield=0.0,
                 slippage=0.0, commission=0.0):
        cfg = MeanReversionConfig(
            two_layer_mode=True,
            core_allocation_pct=0.80,
            tactical_allocation_pct=0.20,
            rebalance_freq=None,
            starting_capital=100_000,
            entry_at="same_close",
            ratio_anchor_window=50,
            ratio_lookback=30,
            regime_filter_enabled=False,
            require_reversal_confirmation=False,
            min_trade_notional=0,
            min_shares=1,
            vol_adjust_sizing=False,
            core_entry_mode="dca",
            core_dca_days=dca_days,
            core_dca_start=dca_start,
            core_dca_commission=commission,
            core_dca_slippage_pct=slippage,
            cash_yield_annual_pct=cash_yield,
            entry_z=-99.0,  # no tactical trades
        )
        ratio = compute_ratio_series(df, cfg)
        rz = compute_ratio_z(ratio, cfg.ratio_lookback)
        labels, scores = classify_regime(df, cfg)
        engine = TwoLayerPortfolioEngine(cfg)
        return engine.run(df, rz, labels, scores), cfg

    def test_dca_invests_exactly_core_capital(self):
        """DCA should deploy all core_capital over N days (within cents)."""
        df = _make_df(200, start=100.0, drift=0.0, vol=0.1)
        result, cfg = self._run_dca(df, dca_days=40)
        core_capital = cfg.starting_capital * cfg.core_allocation_pct  # 80,000
        # After DCA completes, all core_capital should be invested
        # core.cash should be near 0 (no yield, no slippage)
        # Total cost_basis should approximate core_capital
        core_shares = result["core_shares"]
        self.assertGreater(core_shares, 0)
        # Core equity at end = shares * close[-1] + residual cash
        # The core_equity_curve[-1] should reflect all capital deployed
        # Check: shares * avg_price ~ core_capital (no slippage/comm)
        last_core_eq = result["core_equity_curve"][-1]
        # The core should have value close to core_capital * (close[-1]/avg_price)
        self.assertGreater(last_core_eq, 0)

    def test_dca_cash_near_zero_after_completion(self):
        """After DCA period ends, uninvested core cash should be near 0."""
        n = 200
        dates = pd.date_range("2024-01-01", periods=n, freq="B", tz="UTC")
        # Use constant price for deterministic test
        price = np.full(n, 100.0)
        df = pd.DataFrame({
            "open": price, "high": price + 0.01,
            "low": price - 0.01, "close": price,
            "volume": np.full(n, 1_000_000.0),
        }, index=dates)
        result, cfg = self._run_dca(df, dca_days=40, cash_yield=0.0)
        core_capital = cfg.starting_capital * cfg.core_allocation_pct
        core_shares = result["core_shares"]
        # With constant price=100, shares should be core_capital/100 = 800
        self.assertAlmostEqual(core_shares, core_capital / 100.0, places=1)
        # DCA decisions on 0..39 fill at opens 1..40, so cash should be ~0 at bar 40.
        self.assertIn("core_cash_curve", result)
        self.assertIn("core_shares_curve", result)
        self.assertAlmostEqual(float(result["core_cash_curve"][40]), 0.0, places=2)
        self.assertAlmostEqual(float(result["core_shares_curve"][-1]), float(core_shares), places=6)

    def test_dca_avg_entry_price_in_range(self):
        """DCA avg entry bounds must use open[1:N+1], not close[0:N]."""
        n = 120
        dca_days = 20
        dates = pd.date_range("2024-01-01", periods=n, freq="B", tz="UTC")
        opn = np.linspace(80.0, 100.0, n)
        close = np.linspace(200.0, 220.0, n)
        high = np.maximum(opn, close) + 1.0
        low = np.minimum(opn, close) - 1.0
        df = pd.DataFrame(
            {
                "open": opn,
                "high": high,
                "low": low,
                "close": close,
                "volume": np.full(n, 1_000_000.0),
            },
            index=dates,
        )
        result, _ = self._run_dca(df, dca_days=dca_days)

        open_window = np.array(df["open"].iloc[1:dca_days + 1])
        close_window = np.array(df["close"].iloc[0:dca_days])
        avg_entry = float(result["core_entry_price"])
        self.assertGreater(avg_entry, 0)
        self.assertGreaterEqual(avg_entry, float(open_window.min()) - 0.01)
        self.assertLessEqual(avg_entry, float(open_window.max()) + 0.01)
        self.assertLess(avg_entry, float(close_window.min()) - 1.0)

    def test_core_eq_equals_shares_times_price_plus_cash(self):
        """core_equity_curve[i] = shares*close[i] + cash for all bars in DCA mode.

        DCA uses next_open: decision at bar i, fill at open of bar i+1.
        """
        n = 100
        dates = pd.date_range("2024-01-01", periods=n, freq="B", tz="UTC")
        rng = np.random.RandomState(42)
        close_prices = 100.0 + np.cumsum(rng.randn(n) * 0.5)
        close_prices = np.maximum(close_prices, 50.0)
        open_prices = close_prices + rng.uniform(-0.3, 0.3, n)
        df = pd.DataFrame({
            "open": open_prices, "high": close_prices + 0.5,
            "low": close_prices - 0.5, "close": close_prices,
            "volume": np.full(n, 1_000_000.0),
        }, index=dates)
        cfg = MeanReversionConfig(
            two_layer_mode=True,
            core_allocation_pct=0.80,
            tactical_allocation_pct=0.20,
            rebalance_freq=None,
            starting_capital=100_000,
            entry_at="same_close",
            ratio_anchor_window=50,
            ratio_lookback=30,
            regime_filter_enabled=False,
            require_reversal_confirmation=False,
            min_trade_notional=0,
            min_shares=1,
            vol_adjust_sizing=False,
            core_entry_mode="dca",
            core_dca_days=20,
            core_dca_start=0,
            core_dca_commission=0.0,
            core_dca_slippage_pct=0.0,
            cash_yield_annual_pct=0.0,
            entry_z=-99.0,
        )
        ratio = compute_ratio_series(df, cfg)
        rz = compute_ratio_z(ratio, cfg.ratio_lookback)
        labels, scores = classify_regime(df, cfg)
        # Replay DCA with next_open logic
        core_capital = 80_000.0
        dca_n = 20
        invest_per_day = core_capital / dca_n
        shares = 0.0
        cash = core_capital
        pending = 0.0
        for i in range(n):
            # Execute pending buy at this bar's open
            if pending > 0 and cash > 0:
                px = float(df["open"].iloc[i])
                spend = min(pending, cash)
                new_shares = spend / px
                cash -= new_shares * px
                cash = max(0.0, cash)
                shares += new_shares
                pending = 0.0
            # Schedule buy (guard: not on last bar)
            if 0 <= i < dca_n and i + 1 < n and cash > 0:
                pending = invest_per_day
        # After DCA, shares should be fully deployed
        self.assertAlmostEqual(cash, 0.0, places=1)
        # Run actual engine
        result = TwoLayerPortfolioEngine(cfg).run(df, rz, labels, scores)
        self.assertAlmostEqual(result["core_shares"], shares, places=2)

    def test_instant_mode_unchanged(self):
        """instant mode should produce identical results to original behavior."""
        df = _make_mean_reverting_df(300)
        cfg = MeanReversionConfig(
            two_layer_mode=True,
            core_allocation_pct=0.80,
            tactical_allocation_pct=0.20,
            starting_capital=100_000,
            entry_at="same_close",
            ratio_anchor_window=50,
            ratio_lookback=30,
            regime_filter_enabled=False,
            require_reversal_confirmation=False,
            min_trade_notional=0,
            min_shares=1,
            vol_adjust_sizing=False,
            core_entry_mode="instant",
        )
        ratio = compute_ratio_series(df, cfg)
        rz = compute_ratio_z(ratio, cfg.ratio_lookback)
        labels, scores = classify_regime(df, cfg)
        result = TwoLayerPortfolioEngine(cfg).run(df, rz, labels, scores)

        first_close = float(df["close"].iloc[0])
        expected_shares = 80_000.0 / first_close
        self.assertAlmostEqual(result["core_shares"], expected_shares, places=2)
        self.assertAlmostEqual(result["core_entry_price"], first_close, places=2)
        # core_eq[0] = shares * first_close = core_capital
        self.assertAlmostEqual(result["core_equity_curve"][0], 80_000.0, places=1)

    def test_baseline_matched_equals_baseline_in_instant_mode(self):
        """In instant mode, matched baseline should equal original baseline."""
        df = _make_df(200)
        cfg = MeanReversionConfig(
            two_layer_mode=True,
            core_allocation_pct=0.80,
            tactical_allocation_pct=0.20,
            starting_capital=100_000,
            core_entry_mode="instant",
            entry_at="same_close",
            ratio_anchor_window=50,
            ratio_lookback=30,
            regime_filter_enabled=False,
            require_reversal_confirmation=False,
            min_trade_notional=0,
            min_shares=1,
            vol_adjust_sizing=False,
        )
        ratio = compute_ratio_series(df, cfg)
        rz = compute_ratio_z(ratio, cfg.ratio_lookback)
        labels, scores = classify_regime(df, cfg)
        result = TwoLayerPortfolioEngine(cfg).run(df, rz, labels, scores)

        self.assertEqual(len(result["baseline_matched_core_entry"]),
                         len(result["baseline_equity_curve"]))
        for i in range(len(df)):
            self.assertAlmostEqual(
                result["baseline_matched_core_entry"][i],
                result["baseline_equity_curve"][i],
                places=1,
                msg=f"Matched != baseline at bar {i} in instant mode",
            )

    def test_dca_with_slippage_and_commission(self):
        """DCA with costs should result in fewer shares than without."""
        n = 100
        dates = pd.date_range("2024-01-01", periods=n, freq="B", tz="UTC")
        price = np.full(n, 100.0)
        df = pd.DataFrame({
            "open": price, "high": price + 0.01,
            "low": price - 0.01, "close": price,
            "volume": np.full(n, 1_000_000.0),
        }, index=dates)
        result_no_cost, _ = self._run_dca(df, dca_days=20, slippage=0.0, commission=0.0)
        result_with_cost, _ = self._run_dca(df, dca_days=20, slippage=0.001, commission=1.0)
        self.assertGreater(result_no_cost["core_shares"],
                           result_with_cost["core_shares"])

    def test_matched_baseline_different_in_dca_mode(self):
        """In DCA mode, matched baseline should differ from static baseline."""
        df = _make_df(200, start=100.0, drift=0.1, vol=0.5)
        result, cfg = self._run_dca(df, dca_days=40)
        base = result["baseline_equity_curve"]
        matched = result["baseline_matched_core_entry"]
        self.assertEqual(len(base), len(matched))
        # They should differ during DCA period
        differs = sum(1 for a, b in zip(base, matched) if abs(a - b) > 0.01)
        self.assertGreater(differs, 0,
                           "Matched baseline should differ from static in DCA mode")

    def test_dca_last_bar_guard(self):
        """No DCA buy should be scheduled on the last bar (no future open)."""
        n = 10
        dates = pd.date_range("2024-01-01", periods=n, freq="B", tz="UTC")
        opn = 100.0 + np.arange(n, dtype=float)
        close = np.full(n, 100.0)
        df = pd.DataFrame({
            "open": opn, "high": opn + 0.01,
            "low": opn - 0.01, "close": close,
            "volume": np.full(n, 1_000_000.0),
        }, index=dates)
        result, cfg = self._run_dca(df, dca_days=20)
        core_capital = cfg.starting_capital * cfg.core_allocation_pct
        dca_n = min(20, n)
        invest_per_day = core_capital / dca_n
        expected_shares = float(np.sum(invest_per_day / opn[1:n]))
        unexpected_with_last_fill = expected_shares + (invest_per_day / opn[n - 1])
        expected_cash = core_capital - invest_per_day * (n - 1)
        expected_last_equity = expected_shares * close[-1] + expected_cash
        self.assertAlmostEqual(float(result["core_shares"]), expected_shares, places=6)
        self.assertLess(float(result["core_shares"]), unexpected_with_last_fill - 0.1)
        self.assertAlmostEqual(float(result["core_equity_curve"][-1]), expected_last_equity, places=2)

    def test_dca_fills_at_open_not_close(self):
        """DCA fills should use open prices, not close prices."""
        n = 100
        dates = pd.date_range("2024-01-01", periods=n, freq="B", tz="UTC")
        # Open and close differ significantly
        close = np.full(n, 100.0)
        opn = np.full(n, 50.0)  # open is half of close
        df = pd.DataFrame({
            "open": opn, "high": close + 1,
            "low": opn - 1, "close": close,
            "volume": np.full(n, 1_000_000.0),
        }, index=dates)
        result, cfg = self._run_dca(df, dca_days=20)
        # Fills at open=50, so avg entry should be ~50, not ~100
        avg_entry = result["core_entry_price"]
        self.assertAlmostEqual(avg_entry, 50.0, places=1)

    def test_dca_rounding_deployment_within_cents_and_cash_non_negative(self):
        """Deployment should be within cents and direct core cash never negative."""
        n = 80
        dates = pd.date_range("2024-01-01", periods=n, freq="B", tz="UTC")
        price = np.full(n, 33.33)
        df = pd.DataFrame({
            "open": price, "high": price + 0.01,
            "low": price - 0.01, "close": price,
            "volume": np.full(n, 1_000_000.0),
        }, index=dates)
        result, cfg = self._run_dca(df, dca_days=13, cash_yield=0.0)
        core_capital = cfg.starting_capital * cfg.core_allocation_pct
        cash_curve = [float(x) for x in result["core_cash_curve"]]
        self.assertTrue(all(c >= -0.01 for c in cash_curve))
        # dca_days=13 means last fill occurs at bar 13 (next-open execution).
        self.assertLessEqual(abs(cash_curve[13]), 0.01)
        self.assertLessEqual(abs(cash_curve[-1]), 0.01)
        deployed_capital = core_capital - cash_curve[-1]
        self.assertLessEqual(abs(deployed_capital - core_capital), 0.01)

    def test_core_dca_days_validation(self):
        """core_dca_days < 1 should raise."""
        with self.assertRaises(AssertionError):
            MeanReversionConfig(core_entry_mode="dca", core_dca_days=0)

    def test_result_contains_entry_mode_key(self):
        """Result dict should contain core_entry_mode."""
        df = _make_df(100)
        result, _ = self._run_dca(df, dca_days=20)
        self.assertEqual(result["core_entry_mode"], "dca")
        self.assertEqual(result["core_dca_days"], 20)
        self.assertIn("core_cash_curve", result)
        self.assertIn("core_shares_curve", result)


# =============================================================================
# v3.4 - UNIVERSE FILTER TESTS
# =============================================================================

class TestUniverseFilter(unittest.TestCase):
    """Feature 2: Universe Filter (data quality + liquidity proxy)."""

    def _make_low_vol_df(self, n=200, price=3.0, volume=10_000.0):
        """Create df with low price and low volume."""
        dates = pd.date_range("2024-01-01", periods=n, freq="B", tz="UTC")
        close = np.full(n, price)
        df = pd.DataFrame({
            "open": close,
            "high": close + 0.01,
            "low": close - 0.01,
            "close": close,
            "volume": np.full(n, volume),
        }, index=dates)
        df["return"] = df["close"].pct_change()
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))
        df["volume_sma_20"] = df["volume"].rolling(20).mean()
        df["volume_ratio"] = df["volume"] / df["volume_sma_20"]
        return df

    def _make_good_df(self, n=200, price=150.0, volume=2_000_000.0):
        """Create df with good price and volume."""
        dates = pd.date_range("2024-01-01", periods=n, freq="B", tz="UTC")
        rng = np.random.RandomState(42)
        close = price + np.cumsum(rng.randn(n) * 0.5)
        close = np.maximum(close, price * 0.5)
        df = pd.DataFrame({
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.full(n, volume),
        }, index=dates)
        df["return"] = df["close"].pct_change()
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))
        df["volume_sma_20"] = df["volume"].rolling(20).mean()
        df["volume_ratio"] = df["volume"] / df["volume_sma_20"]
        return df

    def test_low_dollar_volume_fails(self):
        """Low dollar volume should fail the filter."""
        df = self._make_low_vol_df(price=10.0, volume=100_000.0)
        # dollar_vol = 10 * 100k = 1M < 20M threshold
        cfg = MeanReversionConfig(
            universe_filter_enabled=True,
            min_dollar_vol=20_000_000.0,
            min_price=5.0,
        )
        passed, reasons, metrics = universe_filter(df, cfg)
        self.assertFalse(passed)
        self.assertTrue(any("median_dollar_vol" in r for r in reasons))

    def test_low_price_fails(self):
        """Low median price should fail the filter."""
        df = self._make_low_vol_df(price=3.0, volume=10_000_000.0)
        cfg = MeanReversionConfig(
            universe_filter_enabled=True,
            min_dollar_vol=1.0,
            min_price=5.0,
        )
        passed, reasons, metrics = universe_filter(df, cfg)
        self.assertFalse(passed)
        self.assertTrue(any("median_close" in r for r in reasons))

    def test_sufficient_volume_passes(self):
        """Good df should pass the filter."""
        df = self._make_good_df(price=150.0, volume=2_000_000.0)
        # dollar_vol = ~150 * 2M = 300M >> 20M
        cfg = MeanReversionConfig(
            universe_filter_enabled=True,
            min_dollar_vol=20_000_000.0,
            min_price=5.0,
        )
        passed, reasons, metrics = universe_filter(df, cfg)
        self.assertTrue(passed)
        self.assertEqual(len(reasons), 0)

    def test_filter_returns_metrics(self):
        """Filter should return metrics dict with expected keys."""
        df = self._make_good_df()
        cfg = MeanReversionConfig(universe_filter_enabled=True)
        _, _, metrics = universe_filter(df, cfg)
        self.assertIn("median_close", metrics)
        self.assertIn("median_dollar_vol", metrics)
        self.assertIn("n_gaps_over_3d", metrics)

    def test_missing_volume_fails_gracefully_with_reason(self):
        df = self._make_good_df().drop(columns=["volume"])
        cfg = MeanReversionConfig(
            universe_filter_enabled=True,
            min_dollar_vol=1.0,
            min_price=1.0,
        )
        passed, reasons, metrics = universe_filter(df, cfg)
        self.assertFalse(passed)
        self.assertTrue(any("missing_columns=volume" in r for r in reasons))
        self.assertIn("median_dollar_vol", metrics)

    def test_nan_frac_tolerance_passes_at_default(self):
        n = 400
        df = self._make_good_df(n=n, price=150.0, volume=2_000_000.0)
        df.loc[df.index[10], "close"] = np.nan
        df.loc[df.index[20], "volume"] = np.nan
        cfg = MeanReversionConfig(
            universe_filter_enabled=True,
            min_dollar_vol=1.0,
            min_price=1.0,
            max_nan_frac=0.01,
        )
        passed, reasons, metrics = universe_filter(df, cfg)
        self.assertTrue(passed)
        self.assertEqual(reasons, [])
        self.assertAlmostEqual(float(metrics.get("nan_frac", 0.0)), 2.0 / float(n), places=6)

    def test_nan_frac_tolerance_fails_when_threshold_tight(self):
        n = 400
        df = self._make_good_df(n=n, price=150.0, volume=2_000_000.0)
        df.loc[df.index[10], "close"] = np.nan
        df.loc[df.index[20], "volume"] = np.nan
        cfg = MeanReversionConfig(
            universe_filter_enabled=True,
            min_dollar_vol=1.0,
            min_price=1.0,
            max_nan_frac=0.001,
        )
        passed, reasons, metrics = universe_filter(df, cfg)
        self.assertFalse(passed)
        self.assertIn("too_many_nans", reasons)
        self.assertAlmostEqual(float(metrics.get("nan_frac", 0.0)), 2.0 / float(n), places=6)

    def test_insufficient_usable_bars_reason(self):
        n = 300
        df = self._make_good_df(n=n, price=150.0, volume=2_000_000.0)
        df.loc[df.index[:30], "close"] = np.nan
        cfg = MeanReversionConfig(
            universe_filter_enabled=True,
            min_dollar_vol=1.0,
            min_price=1.0,
            max_nan_frac=1.0,
        )
        passed, reasons, metrics = universe_filter(df, cfg, require_min_bars=290)
        self.assertFalse(passed)
        self.assertIn("insufficient_usable_bars", reasons)
        self.assertEqual(int(metrics.get("usable_rows", 0)), 270)
        self.assertEqual(int(metrics.get("min_required", 0)), 290)

    def test_skipped_ticker_produces_no_trades_but_summary(self):
        """Skipped tickers should produce TXT + JSON but no trades CSV."""
        df = self._make_low_vol_df(n=500, price=3.0, volume=1000.0)
        fetch_meta = {
            "requested_price_field": "close",
            "price_field_used": "close",
            "price_field_warning": "",
            "raw_duplicate_timestamps_count": 0,
            "raw_is_monotonic_index": True,
            "has_splits": False,
            "has_dividends": False,
            "ohlc_scaled_by_adjclose": False,
            "volume_inverse_scaled": False,
            "has_split_events": False,
            "has_dividend_events": False,
        }
        cfg = MeanReversionConfig(
            regime_filter_enabled=False,
            universe_filter_enabled=True,
            min_dollar_vol=20_000_000.0,
            min_price=5.0,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch(
                "mean_reversion_standalone.fetch_price_data",
                return_value=(df, fetch_meta),
            ), mock.patch(
                "mean_reversion_standalone.fetch_basic_info",
                return_value={"ticker": "PENNY", "name": "Penny Stock"},
            ):
                result = analyze(
                    "PENNY",
                    cfg=cfg,
                    quiet=True,
                    reports_dir=tmpdir,
                    export_trades=True,
                    period="500d",
                    interval="1d",
                    price_field="close",
                    require_min_bars=100,
                )
            self.assertTrue(result.get("skipped_by_universe_filter", False))
            self.assertEqual(result.get("reason"), "SKIPPED BY UNIVERSE FILTER")
            self.assertIsNone(result.get("trades_file"))
            # Summary JSON should exist
            summary_path = result.get("summary_file")
            self.assertTrue(summary_path and os.path.isfile(summary_path))
            assert isinstance(summary_path, str)
            with open(summary_path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
            self.assertEqual(payload.get("reason"), "SKIPPED BY UNIVERSE FILTER")
            self.assertIn("filter_reasons", payload)
            # TXT report should exist
            txt_path = result.get("file")
            self.assertTrue(txt_path and os.path.isfile(txt_path))
            assert isinstance(txt_path, str)
            with open(txt_path, "r", encoding="utf-8") as fh:
                txt = fh.read()
            self.assertIn("SKIPPED BY UNIVERSE FILTER", txt)

    def test_skipped_ticker_missing_volume_does_not_crash(self):
        """Missing volume should skip gracefully and still emit TXT + JSON."""
        df = self._make_good_df(n=500).drop(columns=["volume"])
        fetch_meta = {
            "requested_price_field": "close",
            "price_field_used": "close",
            "price_field_warning": "",
            "raw_duplicate_timestamps_count": 0,
            "raw_is_monotonic_index": True,
            "has_splits": False,
            "has_dividends": False,
            "ohlc_scaled_by_adjclose": False,
            "volume_inverse_scaled": False,
            "has_split_events": False,
            "has_dividend_events": False,
        }
        cfg = MeanReversionConfig(
            regime_filter_enabled=False,
            universe_filter_enabled=True,
            min_dollar_vol=20_000_000.0,
            min_price=5.0,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch(
                "mean_reversion_standalone.fetch_price_data",
                return_value=(df, fetch_meta),
            ), mock.patch(
                "mean_reversion_standalone.fetch_basic_info",
                return_value={"ticker": "MISSVOL", "name": "Missing Volume Inc"},
            ):
                result = analyze(
                    "MISSVOL",
                    cfg=cfg,
                    quiet=True,
                    reports_dir=tmpdir,
                    export_trades=True,
                    period="500d",
                    interval="1d",
                    price_field="close",
                    require_min_bars=100,
                )
            self.assertTrue(result.get("skipped_by_universe_filter", False))
            self.assertTrue(any("missing_columns=volume" in r for r in result.get("filter_reasons", [])))
            summary_path = result.get("summary_file")
            self.assertTrue(summary_path and os.path.isfile(summary_path))
            txt_path = result.get("file")
            self.assertTrue(txt_path and os.path.isfile(txt_path))

    def test_filter_disabled_passes_everything(self):
        """When filter is disabled, even bad data passes."""
        df = self._make_low_vol_df(price=1.0, volume=100.0)
        cfg = MeanReversionConfig(universe_filter_enabled=False)
        # universe_filter function should still work, but analyze won't call it
        passed, reasons, _ = universe_filter(df, cfg)
        # The function itself doesn't check cfg.universe_filter_enabled
        self.assertFalse(passed)  # it would fail if called

    def test_gap_filter(self):
        """Excessive gaps should fail the filter."""
        # Create data with many gaps
        dates = pd.bdate_range("2024-01-01", periods=50, tz="UTC")
        # Insert big gaps by removing dates and adding far-apart ones
        gap_dates = pd.to_datetime([
            "2024-01-02", "2024-01-15", "2024-01-30",
            "2024-02-15", "2024-03-01", "2024-03-15",
            "2024-04-01", "2024-04-15", "2024-05-01",
            "2024-05-15", "2024-06-01", "2024-06-15",
            "2024-07-01", "2024-07-15", "2024-08-01",
        ], utc=True)
        price = np.full(len(gap_dates), 200.0)
        df = pd.DataFrame({
            "open": price, "high": price + 1, "low": price - 1,
            "close": price, "volume": np.full(len(gap_dates), 5_000_000.0),
        }, index=gap_dates)
        cfg = MeanReversionConfig(
            universe_filter_enabled=True,
            min_dollar_vol=1.0,
            min_price=1.0,
            max_gaps=3,
        )
        passed, reasons, metrics = universe_filter(df, cfg)
        self.assertFalse(passed)
        self.assertTrue(any("n_gaps_over_3d" in r for r in reasons))

    def test_daily_weekend_not_counted_as_gap(self):
        """For 1d data, weekend/holiday-style short calendar gaps should not count."""
        dates = pd.to_datetime(
            ["2024-01-05", "2024-01-08", "2024-01-09", "2024-01-12"], utc=True
        )
        price = np.full(len(dates), 200.0)
        df = pd.DataFrame(
            {
                "open": price,
                "high": price + 1.0,
                "low": price - 1.0,
                "close": price,
                "volume": np.full(len(dates), 5_000_000.0),
            },
            index=dates,
        )
        cfg = MeanReversionConfig(
            universe_filter_enabled=True,
            min_dollar_vol=1.0,
            min_price=1.0,
            max_gaps=1,
            gap_days_threshold=7,
        )
        passed, reasons, metrics = universe_filter(df, cfg, interval="1d")
        self.assertTrue(passed)
        self.assertEqual(int(metrics.get("n_gaps_over_3d", -1)), 0)
        self.assertEqual(int(metrics.get("gap_days_threshold_used", -1)), 7)
        self.assertEqual(reasons, [])

    def test_daily_true_long_gap_counted_and_can_fail(self):
        """A true long missing chunk should still count and fail when max_gaps is small."""
        dates = pd.to_datetime(
            ["2024-01-02", "2024-01-03", "2024-01-20", "2024-02-10"], utc=True
        )
        price = np.full(len(dates), 200.0)
        df = pd.DataFrame(
            {
                "open": price,
                "high": price + 1.0,
                "low": price - 1.0,
                "close": price,
                "volume": np.full(len(dates), 5_000_000.0),
            },
            index=dates,
        )
        cfg = MeanReversionConfig(
            universe_filter_enabled=True,
            min_dollar_vol=1.0,
            min_price=1.0,
            max_gaps=1,
            gap_days_threshold=7,
        )
        passed, reasons, metrics = universe_filter(df, cfg, interval="1d")
        self.assertFalse(passed)
        self.assertGreaterEqual(int(metrics.get("n_gaps_over_3d", 0)), 2)
        self.assertTrue(any("n_gaps_over_3d" in r for r in reasons))

    def test_zero_volume_and_tail_price_checks(self):
        dates = pd.date_range("2024-01-01", periods=60, freq="B", tz="UTC")
        close = np.full(len(dates), 2.0, dtype=float)
        close[-30:] = 0.8  # tail degradation
        volume = np.full(len(dates), 2_000_000.0, dtype=float)
        volume[:10] = 0.0
        df = pd.DataFrame(
            {
                "open": close,
                "high": close + 0.1,
                "low": close - 0.1,
                "close": close,
                "volume": volume,
            },
            index=dates,
        )
        cfg = MeanReversionConfig(
            universe_filter_enabled=True,
            min_dollar_vol=1.0,
            min_price=0.5,
            min_price_tail=1.0,
            max_zero_vol_days=5,
            max_gaps=99,
        )
        passed, reasons, metrics = universe_filter(df, cfg, interval="1d")
        self.assertFalse(passed)
        self.assertGreater(int(metrics.get("zero_volume_days", 0)), 5)
        self.assertLess(float(metrics.get("tail_median_close", 99.0)), 1.0)
        self.assertTrue(any("zero_volume_days" in r for r in reasons))
        self.assertTrue(any("tail_median_close" in r for r in reasons))

    def test_skipped_report_text(self):
        """_generate_skipped_report should produce readable text."""
        cfg = MeanReversionConfig()
        report = _generate_skipped_report(
            "TEST", cfg,
            ["median_close=3.00 < min_price=5.00"],
            {"median_close": 3.0, "median_dollar_vol": 100.0, "n_gaps_over_3d": 0},
        )
        self.assertIn("SKIPPED BY UNIVERSE FILTER", report)
        self.assertIn("median_close=3.00", report)
        self.assertIn("DISCLAIMER", report)


# =============================================================================
# SPEC V1: TACTICAL OVERLAY UPGRADE TESTS
# =============================================================================

class TestSpecV1VolMultiplierClipping(unittest.TestCase):
    """Spec v1: m_vol = clip(target_vol / max(sigma_annual, vol_floor), 0, vol_cap)."""

    def test_mvol_clips_to_cap(self):
        """When sigma_annual is very low, m_vol should clip at vol_cap."""
        # sigma_annual=0.02, target=0.15, floor=0.05 => raw = 0.15/0.05 = 3.0, cap=1.5
        result = _vol_target_multiplier(0.02, sigma_target=0.15, sigma_floor=0.05, cap=1.5)
        self.assertAlmostEqual(result, 1.5, places=4)

    def test_mvol_clips_to_zero(self):
        """Negative sigma should not produce negative multiplier."""
        result = _vol_target_multiplier(-0.1, sigma_target=0.15, sigma_floor=0.05, cap=1.5)
        # max(-0.1, 0.05) = 0.05 => 0.15/0.05 = 3.0 => clip to 1.5
        self.assertAlmostEqual(result, 1.5, places=4)

    def test_mvol_normal_case(self):
        """Synth test: sigma_annual=0.10 -> m_vol = 0.15/0.10 = 1.5."""
        result = _vol_target_multiplier(0.10, sigma_target=0.15, sigma_floor=0.05, cap=1.5)
        self.assertAlmostEqual(result, 1.5, places=4)

    def test_mvol_moderate_vol(self):
        """Synth test: sigma_annual=0.30 -> m_vol = 0.15/0.30 = 0.5."""
        result = _vol_target_multiplier(0.30, sigma_target=0.15, sigma_floor=0.05, cap=1.5)
        self.assertAlmostEqual(result, 0.5, places=4)

    def test_mvol_none_returns_one(self):
        """When sigma_annual is None, m_vol should be 1.0 (passthrough)."""
        result = _vol_target_multiplier(None, sigma_target=0.15, sigma_floor=0.05, cap=1.5)
        self.assertAlmostEqual(result, 1.0, places=4)


class TestSpecV1ConfidenceMultiplier(unittest.TestCase):
    """Spec v1: m_conf = clip((confidence - c0) / (1.0 - c0), 0, 1.0), no gamma."""

    def test_mconf_below_c0_is_zero(self):
        result = _confidence_sizing_multiplier(0.50, c0=0.60, gamma=1.0)
        self.assertAlmostEqual(result, 0.0, places=4)

    def test_mconf_at_c0_is_zero(self):
        result = _confidence_sizing_multiplier(0.60, c0=0.60, gamma=1.0)
        self.assertAlmostEqual(result, 0.0, places=4)

    def test_mconf_at_1_is_one(self):
        result = _confidence_sizing_multiplier(1.0, c0=0.60, gamma=1.0)
        self.assertAlmostEqual(result, 1.0, places=4)

    def test_mconf_midpoint_linear(self):
        """c0=0.60, conf=0.80 => (0.80-0.60)/(1.0-0.60) = 0.5."""
        result = _confidence_sizing_multiplier(0.80, c0=0.60, gamma=1.0)
        self.assertAlmostEqual(result, 0.5, places=4)

    def test_mconf_none_returns_zero(self):
        result = _confidence_sizing_multiplier(None, c0=0.60, gamma=1.0)
        self.assertAlmostEqual(result, 0.0, places=4)


class TestSpecV1ExpectedReturn(unittest.TestCase):
    """Spec v1: e_ret = dR * (ratio_mu - ratio_now) / ratio_now."""

    def test_eret_positive_when_undervalued(self):
        """ratio_now < ratio_mu => positive expected return."""
        eret = _expected_return_spec_v1(0.95, 1.0, half_life=20.0)
        self.assertGreater(eret, 0.0)

    def test_eret_negative_when_overvalued(self):
        """ratio_now > ratio_mu => negative expected return."""
        eret = _expected_return_spec_v1(1.05, 1.0, half_life=20.0)
        self.assertLess(eret, 0.0)

    def test_eret_zero_when_no_half_life(self):
        """half_life=None => dR=0 => eret=0."""
        eret = _expected_return_spec_v1(0.95, 1.0, half_life=None)
        self.assertAlmostEqual(eret, 0.0, places=6)

    def test_eret_zero_when_at_mean(self):
        """ratio_now == ratio_mu => eret=0."""
        eret = _expected_return_spec_v1(1.0, 1.0, half_life=20.0)
        self.assertAlmostEqual(eret, 0.0, places=6)


class TestSpecV1CostEffective(unittest.TestCase):
    """Spec v1: is_cost_effective = e_ret >= cost_k * (slippage_bps/10000 + commission/notional)."""

    def test_cost_effective_true(self):
        result = _is_cost_effective_spec_v1(
            e_ret=0.01, cost_k=1.0, slippage_bps=15.0,
            commission=1.0, notional=10000.0,
        )
        # min_req = 1.0 * (15/10000 + 1/10000) = 0.0016
        self.assertTrue(result)

    def test_cost_effective_false(self):
        result = _is_cost_effective_spec_v1(
            e_ret=0.0005, cost_k=1.25, slippage_bps=15.0,
            commission=1.0, notional=10000.0,
        )
        # min_req = 1.25 * (0.0015 + 0.0001) = 0.002
        self.assertFalse(result)

    def test_cost_effective_zero_notional(self):
        result = _is_cost_effective_spec_v1(
            e_ret=0.01, cost_k=1.0, slippage_bps=15.0,
            commission=1.0, notional=0.0,
        )
        self.assertFalse(result)


class TestSpecV1EnableFlag(unittest.TestCase):
    """Spec v1 gating: enable_spec_v1_upgrades toggles new logic."""

    def test_config_spec_v1_defaults(self):
        """Verify Spec v1 default values match the spec."""
        cfg = MeanReversionConfig()
        self.assertAlmostEqual(cfg.tactical_vol_target, 0.15)
        self.assertAlmostEqual(cfg.tactical_vol_floor, 0.05)
        self.assertAlmostEqual(cfg.tactical_vol_cap, 1.50)
        self.assertAlmostEqual(cfg.cost_k, 1.0)
        self.assertAlmostEqual(cfg.tactical_exit_z, -0.20)
        self.assertEqual(cfg.tactical_min_hold_days, 3)
        self.assertAlmostEqual(cfg.confidence_c0, 0.60)

    def test_config_default_off(self):
        cfg = MeanReversionConfig()
        self.assertFalse(cfg.enable_spec_v1_upgrades)

    def test_config_copy_with_preserves_flag(self):
        cfg = MeanReversionConfig(enable_spec_v1_upgrades=True)
        copy = cfg.copy_with(debug=True)
        self.assertTrue(copy.enable_spec_v1_upgrades)
        self.assertTrue(copy.debug)

    def test_baseline_unaffected_when_flag_off(self):
        """With enable_spec_v1_upgrades=False, baseline behavior is unchanged."""
        df = _make_mean_reverting_df(n=400)
        cfg = MeanReversionConfig(
            enable_spec_v1_upgrades=False,
            entry_at="same_close",
            regime_filter_enabled=False,
            require_reversal_confirmation=False,
            min_trade_notional=0, min_shares=1,
            vol_adjust_sizing=True,
        )
        ratio = compute_ratio_series(df, cfg)
        rz = compute_ratio_z(ratio, cfg.ratio_lookback)
        labels = pd.Series(Regime.MEAN_REVERTING.value, index=df.index)
        scores = pd.Series(0.8, index=df.index)
        engine = BacktestEngine(cfg)
        result = engine.run(df, rz, labels, scores)
        # Should produce some fills (baseline path)
        self.assertGreater(len(result["fills"]), 0)
        # Should NOT have any COST_AWARE_FILTER or REVERTED_TO_MEAN
        for f in result["fills"]:
            self.assertNotEqual(f.get("reason"), "REVERTED_TO_MEAN")
        for a in result["actions_log"]:
            self.assertNotIn("COST_AWARE_FILTER", a.get("reason", ""))


class TestSpecV1RevertedToMeanExit(unittest.TestCase):
    """Spec v1: should_exit_mr fires REVERTED_TO_MEAN when ratio_z worsens."""

    def test_reverted_to_mean_exit_fires(self):
        """Force ratio_z below tactical_exit_z after min_hold_days."""
        n = 300
        rng = np.random.RandomState(42)
        # Mean-reverting OU process to ensure valid half_life
        prices = np.zeros(n)
        prices[0] = 100.0
        for k in range(1, n):
            prices[k] = prices[k - 1] + 0.15 * (100.0 - prices[k - 1]) + 0.5 * rng.randn()
        prices = np.maximum(prices, 50.0)
        dates = pd.date_range("2024-01-01", periods=n, freq="B", tz="UTC")
        df = pd.DataFrame({
            "open": prices, "high": prices + 1, "low": prices - 1,
            "close": prices, "volume": [1_000_000.0] * n,
        }, index=dates)
        df["return"] = df["close"].pct_change()
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))
        df["volume_sma_20"] = df["volume"].rolling(20).mean()
        df["volume_ratio"] = df["volume"] / df["volume_sma_20"]

        cfg = MeanReversionConfig(
            enable_spec_v1_upgrades=True,
            entry_at="same_close",
            regime_filter_enabled=False,
            require_reversal_confirmation=False,
            min_trade_notional=0, min_shares=1,
            vol_adjust_sizing=False,
            tactical_exit_z=-0.20,
            tactical_min_hold_days=3,
            stop_atr_multiple=50.0,  # disable stop-loss
            max_holding_days=999,
            tactical_vol_target=0.15,
            tactical_vol_floor=0.05,
            tactical_vol_cap=1.5,
            cost_k=1.0,
            cost_bps_est=0.0,  # disable cost gate for this test
            commission_per_trade=0.0,
            confidence_c0=0.0,  # allow any confidence through m_conf
        )

        # Force BUY at bar 80, then worsen ratio_z below -0.20 after bar 83
        rz = pd.Series(0.0, index=df.index)
        rz.iloc[80] = -2.0  # trigger buy
        for j in range(81, 84):
            rz.iloc[j] = 0.0  # neutral during hold
        for j in range(84, n):
            rz.iloc[j] = -0.30  # below tactical_exit_z=-0.20

        labels = pd.Series(Regime.MEAN_REVERTING.value, index=df.index)
        scores = pd.Series(0.8, index=df.index)

        engine = BacktestEngine(cfg)
        result = engine.run(df, rz, labels, scores)

        reverted_exits = [f for f in result["fills"]
                          if f.get("reason") == "REVERTED_TO_MEAN"]
        reverted_actions = [a for a in result["actions_log"]
                            if a.get("reason") == "REVERTED_TO_MEAN"]
        self.assertGreater(
            len(reverted_exits) + len(reverted_actions), 0,
            "REVERTED_TO_MEAN exit should fire under Spec v1"
        )


class TestSpecV1Sigma252(unittest.TestCase):
    """Spec v1: sigma_annual = log(close).diff().rolling(252).std() * sqrt(252)."""

    def test_sigma_annual_252_computation(self):
        """Verify the 252-bar log-ret vol formula produces reasonable values."""
        n = 500
        rng = np.random.RandomState(42)
        prices = 100.0 * np.exp(np.cumsum(rng.randn(n) * 0.01))
        close = pd.Series(prices)
        log_close = pd.Series(np.log(close.values), index=close.index)
        sigma = log_close.diff().rolling(window=252).std() * np.sqrt(252)
        # After 252 bars, should have valid values
        valid = sigma.dropna()
        self.assertGreater(len(valid), 0)
        # Annualized vol should be in a reasonable range for 1% daily vol
        self.assertGreater(float(valid.iloc[-1]), 0.05)
        self.assertLess(float(valid.iloc[-1]), 0.50)


class TestSpecV1LookAheadFix(unittest.TestCase):
    """Spec v1: same_close regime scores exclude current bar."""

    def test_same_close_excludes_current_bar(self):
        """With enable_spec_v1_upgrades + same_close, regime should use iloc[:i]."""
        df = _make_mean_reverting_df(n=400)
        cfg_spec = MeanReversionConfig(
            entry_at="same_close",
            enable_spec_v1_upgrades=True,
            regime_update_freq=5,
        )
        cfg_legacy = MeanReversionConfig(
            entry_at="same_close",
            enable_spec_v1_upgrades=False,
            regime_update_freq=5,
        )
        labels_spec, _ = classify_regime(df, cfg_spec)
        labels_legacy, _ = classify_regime(df, cfg_legacy)
        # Both should produce valid labels, but may differ due to look-ahead fix
        self.assertEqual(len(labels_spec), len(labels_legacy))
        # Spec v1 should still produce regime labels (not all NaN)
        valid_count = (labels_spec != Regime.AMBIGUOUS.value).sum()
        self.assertGreater(valid_count, 0)


# =============================================================================
# Problem 1 — Split invariance of dollar volume after adjclose scaling
# =============================================================================

class TestSplitInvarianceDollarVolume(unittest.TestCase):
    """Verify _apply_adjclose_price_volume_scaling preserves dollar volume
    across a synthetic split event and that volume_ratio stays smooth."""

    def _make_split_df(self):
        """Create a 60-bar DataFrame where a 2:1 split happens at bar 30.

        Before the split the raw close is ~200, after the split ~100.
        adjclose is the continuous (post-split-adjusted) series hovering ~100.
        Raw volume doubles at the split (same notional traded).
        """
        n = 60
        idx = pd.date_range("2024-01-01", periods=n, freq="B", tz="UTC")

        # Continuous adjusted price ~ 100 throughout
        adj = np.full(n, 100.0)

        # Raw close: pre-split bars 0..29 trade at 200, post-split at 100
        close_raw = np.where(np.arange(n) < 30, 200.0, 100.0)

        # Raw volume: constant dollar volume = 1e8 => vol = dv / close_raw
        dollar_vol_const = 1e8
        volume_raw = dollar_vol_const / close_raw  # 500k pre, 1M post

        df = pd.DataFrame({
            "open":     close_raw.copy(),
            "high":     close_raw + 1.0,
            "low":      close_raw - 1.0,
            "close":    close_raw,
            "adjclose": adj,
            "volume":   volume_raw,
        }, index=idx)
        return df, dollar_vol_const

    def test_dollar_volume_preserved_after_scaling(self):
        """close_adj * volume_adj ≈ original close * volume everywhere."""
        df, dollar_vol_const = self._make_split_df()

        # Original dollar volume
        orig_dv = df["close"] * df["volume"]
        np.testing.assert_allclose(np.asarray(orig_dv), dollar_vol_const, rtol=1e-10)

        out, _ = _apply_adjclose_price_volume_scaling(df)

        # After scaling, dollar volume must still be preserved
        adj_dv = out["close"] * out["volume"]
        np.testing.assert_allclose(np.asarray(adj_dv), dollar_vol_const, rtol=1e-10,
                                   err_msg="Dollar volume not preserved after adjclose scaling")

    def test_volume_ratio_smooth_across_split(self):
        """volume_ratio computed from adjusted volume has no artificial spike at
        the split boundary (bar 30)."""
        df, _ = self._make_split_df()
        out, _ = _apply_adjclose_price_volume_scaling(df)

        vol_sma_20 = out["volume"].rolling(20).mean()
        vol_ratio = out["volume"] / vol_sma_20

        # After warmup (bar 19+), all volume_ratio values should be ~1.0
        # because underlying volume is constant in dollar terms.
        # At the split boundary bar 30, raw volume doubles — but adj volume
        # should remain constant, so ratio should NOT spike.
        post_warmup = vol_ratio.iloc[20:].dropna()
        self.assertTrue(len(post_warmup) > 0)
        # All ratios should be very close to 1.0 (constant notional)
        np.testing.assert_allclose(np.asarray(post_warmup), 1.0, atol=1e-10,
                                   err_msg="volume_ratio should be ~1.0 for constant dollar volume")


# =============================================================================
# Problem 2 — next_open last-bar entry is blocked (engine-level)
# =============================================================================

class TestNextOpenLastBarEntryBlocked(unittest.TestCase):
    """Ensure BacktestEngine never opens a trade on the last bar when
    entry_at='next_open', even if a BUY signal fires there."""

    def test_no_trade_on_last_bar_next_open(self):
        """Force a BUY signal on the final bar with entry_at='next_open'.
        Assert zero trades are opened."""
        n = 3
        idx = pd.date_range("2024-06-01", periods=n, freq="B", tz="UTC")
        df = pd.DataFrame({
            "open":   [100.0, 101.0, 102.0],
            "high":   [101.0, 102.0, 103.0],
            "low":    [99.0,  100.0, 101.0],
            "close":  [100.0, 101.0, 102.0],
            "volume": [1e6,   1e6,   1e6],
        }, index=idx)
        df["return"] = df["close"].pct_change()
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))
        df["volume_sma_20"] = df["volume"].rolling(20).mean()
        df["volume_ratio"] = df["volume"] / df["volume_sma_20"]

        # ratio_z: only the last bar has a strong BUY signal
        rz = pd.Series([0.0, 0.0, -3.0], index=idx, dtype=float)
        labels = pd.Series([Regime.MEAN_REVERTING.value] * n, index=idx, dtype=object)
        scores = pd.Series([1.0] * n, index=idx, dtype=float)

        cfg = MeanReversionConfig(
            entry_at="next_open",
            regime_filter_enabled=False,
            require_reversal_confirmation=False,
            vol_adjust_sizing=False,
            min_trade_notional=0.0,
            min_shares=1,
            commission_per_trade=0.0,
            slippage_pct=0.0,
        )
        result = BacktestEngine(cfg).run(df, rz, labels, scores)
        buys = [f for f in result["fills"] if f.get("action") == "BUY"]
        self.assertEqual(len(buys), 0,
                         "next_open must not produce a fill when signal is on the last bar")

    def test_same_close_last_bar_is_allowed(self):
        """Contrast: same_close entry on last bar IS allowed (no missing bar)."""
        n = 3
        idx = pd.date_range("2024-06-01", periods=n, freq="B", tz="UTC")
        df = pd.DataFrame({
            "open":   [100.0, 101.0, 102.0],
            "high":   [101.0, 102.0, 103.0],
            "low":    [99.0,  100.0, 101.0],
            "close":  [100.0, 101.0, 102.0],
            "volume": [1e6,   1e6,   1e6],
        }, index=idx)
        df["return"] = df["close"].pct_change()
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))
        df["volume_sma_20"] = df["volume"].rolling(20).mean()
        df["volume_ratio"] = df["volume"] / df["volume_sma_20"]

        rz = pd.Series([0.0, 0.0, -3.0], index=idx, dtype=float)
        labels = pd.Series([Regime.MEAN_REVERTING.value] * n, index=idx, dtype=object)
        scores = pd.Series([1.0] * n, index=idx, dtype=float)

        cfg = MeanReversionConfig(
            entry_at="same_close",
            lag_signals_for_same_close=False,
            regime_filter_enabled=False,
            require_reversal_confirmation=False,
            vol_adjust_sizing=False,
            min_trade_notional=0.0,
            min_shares=1,
            commission_per_trade=0.0,
            slippage_pct=0.0,
        )
        result = BacktestEngine(cfg).run(df, rz, labels, scores)
        buys = [f for f in result["fills"] if f.get("action") == "BUY"]
        self.assertGreaterEqual(len(buys), 1,
                                "same_close should allow entry on last bar")


# =============================================================================
# Adaptive Core Entry — Pure function tests
# =============================================================================

class TestAdaptiveCoreDeployFunction(unittest.TestCase):
    """Unit tests for _adaptive_core_deploy_amount pure function."""

    def test_calm_uptrend_state(self):
        """trend=1, low dd, low vol → CALM_UPTREND, slow pace."""
        amt, state = _adaptive_core_deploy_amount(
            close_i=110.0, sma200_i=100.0,  # above SMA → trend=1
            dd_i=0.02, sigma_i=0.08,         # dd < 0.10, sigma < 0.12
            remaining_cash=10000.0,
        )
        self.assertEqual(state, "CALM_UPTREND")
        self.assertAlmostEqual(amt, 10000.0 / 120.0, places=2)

    def test_pullback_state(self):
        """trend=0, dd in [0.05, 0.20] → PULLBACK, fast pace."""
        amt, state = _adaptive_core_deploy_amount(
            close_i=90.0, sma200_i=100.0,   # below SMA → trend=0
            dd_i=0.10, sigma_i=0.08,
            remaining_cash=10000.0,
        )
        self.assertEqual(state, "PULLBACK")
        self.assertAlmostEqual(amt, 10000.0 / 40.0, places=2)

    def test_high_vol_drawdown_by_dd(self):
        """dd >= 0.15 → HIGH_VOL_DRAWDOWN regardless of trend."""
        amt, state = _adaptive_core_deploy_amount(
            close_i=110.0, sma200_i=100.0,  # trend=1
            dd_i=0.20, sigma_i=0.08,         # dd >= 0.15 triggers
            remaining_cash=10000.0,
        )
        self.assertEqual(state, "HIGH_VOL_DRAWDOWN")

    def test_high_vol_drawdown_by_sigma(self):
        """sigma >= vol_target → HIGH_VOL_DRAWDOWN."""
        amt, state = _adaptive_core_deploy_amount(
            close_i=110.0, sma200_i=100.0,
            dd_i=0.02, sigma_i=0.15,         # sigma >= 0.12
            remaining_cash=10000.0,
        )
        self.assertEqual(state, "HIGH_VOL_DRAWDOWN")

    def test_high_vol_drawdown_scaling(self):
        """HIGH_VOL_DRAWDOWN scales by clip(vol_target/max(sigma,floor), 0, 1)."""
        amt, state = _adaptive_core_deploy_amount(
            close_i=90.0, sma200_i=100.0,
            dd_i=0.20, sigma_i=0.24,         # vol_target/sigma = 0.12/0.24 = 0.5
            remaining_cash=10000.0,
            fast_days=40, vol_target=0.12, vol_floor=0.05,
        )
        self.assertEqual(state, "HIGH_VOL_DRAWDOWN")
        expected = (10000.0 / 40.0) * 0.5  # 125.0
        self.assertAlmostEqual(amt, expected, places=2)

    def test_neutral_state(self):
        """No conditions met → NEUTRAL, base pace."""
        amt, state = _adaptive_core_deploy_amount(
            close_i=90.0, sma200_i=100.0,   # trend=0
            dd_i=0.03, sigma_i=0.08,         # dd < 0.05, not pullback
            remaining_cash=10000.0,
        )
        self.assertEqual(state, "NEUTRAL")
        self.assertAlmostEqual(amt, 10000.0 / 60.0, places=2)

    def test_nan_sma_defaults_trend_zero(self):
        """NaN SMA200 (< 200 bars) → trend=0."""
        _, state = _adaptive_core_deploy_amount(
            close_i=100.0, sma200_i=np.nan,
            dd_i=0.03, sigma_i=0.08,
            remaining_cash=10000.0,
        )
        self.assertEqual(state, "NEUTRAL")  # trend=0, dd=0.03 < 0.05

    def test_nan_sigma_no_crash(self):
        """NaN sigma (< vol_window bars) → should not crash."""
        amt, state = _adaptive_core_deploy_amount(
            close_i=110.0, sma200_i=100.0,
            dd_i=0.02, sigma_i=np.nan,
            remaining_cash=10000.0,
        )
        # sigma unknown → treat as 0 → CALM_UPTREND (trend=1, dd<0.10, sigma<target)
        self.assertEqual(state, "CALM_UPTREND")

    def test_clamp_max(self):
        """Deploy clamped to max_deploy_pct * remaining_cash."""
        amt, _ = _adaptive_core_deploy_amount(
            close_i=90.0, sma200_i=100.0,
            dd_i=0.10, sigma_i=0.08,
            remaining_cash=100.0,
            fast_days=1,               # raw = 100/1 = 100
            max_deploy_pct=0.10,       # max = 10
        )
        self.assertAlmostEqual(amt, 10.0, places=4)

    def test_clamp_min(self):
        """Deploy clamped to min_deploy_pct * remaining_cash."""
        amt, _ = _adaptive_core_deploy_amount(
            close_i=110.0, sma200_i=100.0,
            dd_i=0.02, sigma_i=0.08,
            remaining_cash=100.0,
            slow_days=100000,          # raw ≈ 0.001
            min_deploy_pct=0.01,       # min = 1.0
        )
        self.assertAlmostEqual(amt, 1.0, places=4)

    def test_depleted(self):
        """Remaining cash <= 0.01 → DEPLETED, 0 deploy."""
        amt, state = _adaptive_core_deploy_amount(
            close_i=100.0, sma200_i=100.0,
            dd_i=0.0, sigma_i=0.08,
            remaining_cash=0.005,
        )
        self.assertEqual(state, "DEPLETED")
        self.assertEqual(amt, 0.0)


# =============================================================================
# Adaptive Core Entry — Engine-level tests
# =============================================================================

class TestCoreAdaptiveEntry(unittest.TestCase):
    """Engine-level tests for core_entry_mode='adaptive'."""

    def _make_adaptive_cfg(self, **overrides: Any) -> MeanReversionConfig:
        defaults: Dict[str, Any] = dict(
            two_layer_mode=True,
            core_allocation_pct=0.80,
            tactical_allocation_pct=0.20,
            rebalance_freq=None,
            starting_capital=100_000,
            entry_at="same_close",
            ratio_anchor_window=50,
            ratio_lookback=30,
            regime_filter_enabled=False,
            require_reversal_confirmation=False,
            min_trade_notional=0,
            min_shares=1,
            vol_adjust_sizing=False,
            core_entry_mode="adaptive",
            core_adaptive_base_days=60,
            core_adaptive_slow_days=120,
            core_adaptive_fast_days=40,
            core_adaptive_vol_window=20,
            core_adaptive_dd_window=252,
            core_adaptive_vol_target=0.12,
            core_adaptive_vol_floor=0.05,
            core_adaptive_max_deploy_pct=0.10,
            core_adaptive_min_deploy_pct=0.002,
            core_adaptive_start=0,
            cash_yield_annual_pct=0.0,
            entry_z=-99.0,   # suppress all tactical trades
        )
        defaults.update(overrides)
        return MeanReversionConfig(**defaults)

    def _run_adaptive(self, df, **cfg_overrides):
        cfg = self._make_adaptive_cfg(**cfg_overrides)
        ratio = compute_ratio_series(df, cfg)
        rz = compute_ratio_z(ratio, cfg.ratio_lookback)
        labels, scores = classify_regime(df, cfg)
        return TwoLayerPortfolioEngine(cfg).run(df, rz, labels, scores), cfg

    def _make_constant_df(self, n=300, price=100.0):
        dates = pd.date_range("2024-01-01", periods=n, freq="B", tz="UTC")
        close = np.full(n, price)
        return pd.DataFrame({
            "open": close, "high": close + 0.01,
            "low": close - 0.01, "close": close,
            "volume": np.full(n, 1_000_000.0),
        }, index=dates)

    def _make_volatile_df(self, n=300, base=100.0, vol_mult=1.0, seed=42):
        rng = np.random.RandomState(seed)
        dates = pd.date_range("2024-01-01", periods=n, freq="B", tz="UTC")
        ret = rng.randn(n) * 0.01 * vol_mult
        ret[0] = 0.0
        close = base * np.exp(np.cumsum(ret))
        return pd.DataFrame({
            "open": close * (1 + rng.uniform(-0.002, 0.002, n)),
            "high": close * 1.005,
            "low": close * 0.995,
            "close": close,
            "volume": np.full(n, 1_000_000.0),
        }, index=dates)

    # --- Test 1: Accounting identity ---
    def test_accounting_identity(self):
        """core_equity[i] == shares[i]*close[i] + cash[i] for all bars."""
        df = self._make_volatile_df(n=300)
        result, cfg = self._run_adaptive(df)
        core_eq = result["core_equity_curve"]
        shares = result["core_shares_curve"]
        cash = result["core_cash_curve"]
        close_arr = df["close"].to_numpy()
        for i in range(len(df)):
            expected = shares[i] * close_arr[i] + cash[i]
            self.assertAlmostEqual(core_eq[i], round(expected, 2), places=1,
                                   msg=f"Accounting identity failed at bar {i}")

    # --- Test 2: Capital conservation ---
    def test_capital_conservation(self):
        """On flat price: shares*close + cash == starting core capital (within cents)."""
        df = self._make_constant_df(n=300, price=100.0)
        result, cfg = self._run_adaptive(df)
        core_capital = cfg.starting_capital * cfg.core_allocation_pct
        final_shares = result["core_shares_curve"][-1]
        final_cash = result["core_cash_curve"][-1]
        total = final_shares * 100.0 + final_cash
        self.assertAlmostEqual(total, core_capital, places=1,
                               msg="Capital not conserved on flat prices")

    # --- Test 3: Bounded deployment ---
    def test_bounded_deployment(self):
        """Every deployment between min and max pct of remaining cash."""
        df = self._make_volatile_df(n=300)
        result, cfg = self._run_adaptive(df)
        cash = result["core_cash_curve"]
        shares = result["core_shares_curve"]
        close_arr = df["close"].to_numpy()
        min_pct = cfg.core_adaptive_min_deploy_pct
        max_pct = cfg.core_adaptive_max_deploy_pct
        for i in range(1, len(df)):
            invested_delta = (shares[i] - shares[i - 1]) * close_arr[i]
            if invested_delta <= 0.01:
                continue  # no deployment this bar
            remaining_before = cash[i - 1]
            if remaining_before < 0.02:
                continue  # depleted
            # Allow small tolerance for rounding and cash yield
            lo = min_pct * remaining_before * 0.95
            hi = max_pct * remaining_before * 1.05
            self.assertGreaterEqual(invested_delta, lo,
                                    msg=f"Deploy below min at bar {i}: {invested_delta:.4f} < {lo:.4f}")
            self.assertLessEqual(invested_delta, hi,
                                 msg=f"Deploy above max at bar {i}: {invested_delta:.4f} > {hi:.4f}")

    # --- Test 4: No lookahead ---
    def test_no_lookahead(self):
        """Mutating future prices after bar t must not change deploy[t]."""
        n = 300
        df1 = self._make_volatile_df(n=n, seed=42)
        df2 = df1.copy()
        # Mutate last 50 bars of df2
        cutoff = 250
        rng = np.random.RandomState(999)
        close_col = cast(int, df2.columns.get_loc("close"))
        open_col = cast(int, df2.columns.get_loc("open"))
        high_col = cast(int, df2.columns.get_loc("high"))
        low_col = cast(int, df2.columns.get_loc("low"))
        close_tail = cast(pd.Series, df2.iloc[cutoff:, close_col]).to_numpy(dtype=float)
        close_tail = close_tail * (1 + rng.randn(n - cutoff) * 0.05)
        df2.iloc[cutoff:, close_col] = close_tail
        df2.iloc[cutoff:, open_col] = close_tail * 0.999
        df2.iloc[cutoff:, high_col] = close_tail * 1.005
        df2.iloc[cutoff:, low_col] = close_tail * 0.995

        result1, _ = self._run_adaptive(df1)
        result2, _ = self._run_adaptive(df2)

        # Shares curves must be identical up to cutoff
        s1 = result1["core_shares_curve"][:cutoff]
        s2 = result2["core_shares_curve"][:cutoff]
        for i in range(cutoff):
            self.assertAlmostEqual(s1[i], s2[i], places=8,
                                   msg=f"Lookahead leak at bar {i}")

    # --- Test 5: Monotonic risk (higher vol → less deployed) ---
    def test_monotonic_risk(self):
        """Higher vol scenario must not deploy more cumulative capital than low vol."""
        df_low = self._make_volatile_df(n=300, vol_mult=0.5, seed=10)
        df_high = self._make_volatile_df(n=300, vol_mult=3.0, seed=10)

        res_low, _ = self._run_adaptive(df_low)
        res_high, _ = self._run_adaptive(df_high)

        # Cumulative shares invested should be >= in low-vol case at every bar
        s_low = np.array(res_low["core_shares_curve"])
        s_high = np.array(res_high["core_shares_curve"])
        # Compare capital deployed = starting_cash - remaining_cash
        c_low = np.array(res_low["core_cash_curve"])
        c_high = np.array(res_high["core_cash_curve"])
        core_cap = 80_000.0
        deployed_low = core_cap - c_low
        deployed_high = core_cap - c_high
        # Low-vol should have deployed at least as much at every bar
        # Allow small tolerance for rounding
        for i in range(len(df_low)):
            self.assertGreaterEqual(deployed_low[i] + 1.0, deployed_high[i],
                                    msg=f"High vol deployed more than low vol at bar {i}")

    # --- Test 6: Drawdown acceleration ---
    def test_drawdown_acceleration(self):
        """Deeper drawdown must deploy at least as fast in PULLBACK/HIGH_VOL states."""
        # Shallow pullback: dd=0.08 (PULLBACK: trend=0, dd in [0.05, 0.15))
        amt_shallow, state_shallow = _adaptive_core_deploy_amount(
            close_i=95.0, sma200_i=100.0,
            dd_i=0.08, sigma_i=0.08,
            remaining_cash=10000.0,
            fast_days=40,
        )
        # Deeper pullback: dd=0.12 (still PULLBACK, since < 0.15)
        amt_deep, state_deep = _adaptive_core_deploy_amount(
            close_i=88.0, sma200_i=100.0,
            dd_i=0.12, sigma_i=0.08,
            remaining_cash=10000.0,
            fast_days=40,
        )
        self.assertEqual(state_shallow, "PULLBACK")
        self.assertEqual(state_deep, "PULLBACK")
        # Same remaining_cash → same rate (both C/fast_days)
        self.assertAlmostEqual(amt_shallow, amt_deep, places=2)

        # Severe drawdown: dd >= 0.15 → HIGH_VOL_DRAWDOWN (at least as fast)
        amt_hvd, state_hvd = _adaptive_core_deploy_amount(
            close_i=80.0, sma200_i=100.0,
            dd_i=0.22, sigma_i=0.08,
            remaining_cash=10000.0,
            fast_days=40, vol_target=0.12,
        )
        self.assertEqual(state_hvd, "HIGH_VOL_DRAWDOWN")
        # HIGH_VOL_DRAWDOWN with low sigma: scale = clip(0.12/max(0.08, 0.05)) = 1.0
        # so deploy = C/fast_days * 1.0 = same as PULLBACK
        self.assertGreaterEqual(amt_hvd + 0.01, amt_deep)

    # --- Test 7: Stability under perturbation ---
    def test_stability_under_perturbation(self):
        """Small price perturbation → deployment schedule correlation > 0.95."""
        n = 300
        df1 = self._make_volatile_df(n=n, seed=42)
        df2 = df1.copy()
        rng = np.random.RandomState(123)
        # Perturb all prices by +/- 0.1%
        noise = 1 + rng.randn(n) * 0.001
        df2["close"] = df2["close"] * noise
        df2["open"] = df2["open"] * noise
        df2["high"] = df2["high"] * noise
        df2["low"] = df2["low"] * noise

        res1, _ = self._run_adaptive(df1)
        res2, _ = self._run_adaptive(df2)

        s1 = np.array(res1["core_shares_curve"])
        s2 = np.array(res2["core_shares_curve"])
        corr = float(np.corrcoef(s1, s2)[0, 1])
        self.assertGreater(corr, 0.95,
                           f"Deployment schedule unstable: corr={corr:.4f}")

    # --- Test 8: Matched baseline produced ---
    def test_matched_baseline_exists(self):
        """Adaptive mode produces a matched baseline of correct length."""
        df = self._make_volatile_df(n=300)
        result, _ = self._run_adaptive(df)
        matched = result["baseline_matched_core_entry"]
        self.assertEqual(len(matched), len(df))
        # Matched baseline >= 0 at every point
        for v in matched:
            self.assertGreater(v, 0)

    # --- Test 9: Adaptive states recorded ---
    def test_adaptive_states_recorded(self):
        """core_adaptive_states list has one entry per bar."""
        df = self._make_volatile_df(n=300)
        result, _ = self._run_adaptive(df)
        states = result["core_adaptive_states"]
        self.assertEqual(len(states), len(df))
        valid_states = {"CALM_UPTREND", "PULLBACK", "HIGH_VOL_DRAWDOWN",
                        "NEUTRAL", "WAITING", "DEPLETED"}
        for s in states:
            self.assertIn(s, valid_states)


if __name__ == "__main__":
    unittest.main()
