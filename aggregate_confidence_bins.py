#!/usr/bin/env python3
"""
Aggregate per-run trade ledgers into pooled confidence-bin summaries.

Usage:
    python aggregate_confidence_bins.py --reports-dir reports
"""

import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _compute_confidence_bins_0p02(
    rows: List[Dict[str, float]],
    start: float,
    end: float = 1.00,
    low_sample_n: int = 10,
) -> List[Dict[str, Any]]:
    """Compute [start, start+0.02), ... bins; final bin is inclusive at 1.00."""
    valid: List[Tuple[float, float, float]] = []
    for rec in rows:
        conf = _safe_float(rec.get("entry_confidence"), None)
        pnl = _safe_float(rec.get("realized_pnl"), None)
        hold = _safe_float(rec.get("hold_days"), 0.0) or 0.0
        if conf is None or pnl is None:
            continue
        # Type assertion: conf is guaranteed to be float here after the None check
        conf_f: float = conf  # type: ignore[assignment]
        pnl_f: float = pnl  # type: ignore[assignment]
        if conf_f < start or conf_f > end:
            continue
        valid.append((conf_f, pnl_f, float(hold)))

    output: List[Dict[str, Any]] = []
    lo_i_start = int(round(start * 100))
    end_i = int(round(end * 100))
    step_i = 2  # 0.02
    low_sample_thr = max(int(low_sample_n), 1)

    for lo_i in range(lo_i_start, end_i, step_i):
        hi_i = min(lo_i + step_i, end_i)
        lo = lo_i / 100.0
        hi = hi_i / 100.0
        is_last = hi_i >= end_i
        bucket = [
            (c, p, h) for (c, p, h) in valid
            if ((lo <= c <= hi) if is_last else (lo <= c < hi))
        ]
        pnls = [p for _, p, _ in bucket]
        holds = [h for _, _, h in bucket]
        n = len(bucket)
        win_rate = (sum(1 for p in pnls if p > 0) / n * 100.0) if n > 0 else 0.0
        avg_pnl = (sum(pnls) / n) if n > 0 else 0.0
        med_pnl = float(np.median(pnls)) if n > 0 else 0.0
        avg_hold = (sum(holds) / n) if n > 0 else 0.0
        label = f"[{lo:.2f},{hi:.2f}]" if is_last else f"[{lo:.2f},{hi:.2f})"
        output.append({
            "bin": label,
            "bin_lo": round(lo, 2),
            "bin_hi": round(hi, 2),
            "n_trades": n,
            "win_rate": round(float(win_rate), 2),
            "avg_pnl": round(float(avg_pnl), 2),
            "median_pnl": round(float(med_pnl), 2),
            "avg_hold_days": round(float(avg_hold), 2),
            "low_sample": n < low_sample_thr,
        })
    return output


def _summarize_totals(rows: List[Dict[str, float]]) -> Dict[str, float]:
    pnls = [_safe_float(r.get("realized_pnl"), 0.0) or 0.0 for r in rows]
    n = len(pnls)
    wins = sum(1 for p in pnls if p > 0)
    return {
        "total_trades": n,
        "win_rate": round((wins / n * 100.0), 2) if n > 0 else 0.0,
        "avg_pnl": round((sum(pnls) / n), 2) if n > 0 else 0.0,
        "median_pnl": round((float(np.median(pnls)) if n > 0 else 0.0), 2),
    }


def _load_trade_rows(reports_dir: Path) -> Tuple[List[Path], List[Dict[str, float]]]:
    files = sorted(reports_dir.glob("*_TRADES_*.csv"))
    rows: List[Dict[str, float]] = []
    for path in files:
        try:
            with path.open("r", newline="", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                for rec in reader:
                    conf = _safe_float(rec.get("entry_confidence"), None)
                    if conf is None or conf < 0.0 or conf > 1.0:
                        # Skip invalid/missing confidence rows
                        continue
                    # Type assertion: conf is guaranteed to be float here after the None check
                    conf_f: float = conf  # type: ignore[assignment]
                    pnl_val = _safe_float(rec.get("realized_pnl"), 0.0)
                    hold_val = _safe_float(rec.get("hold_days"), 0.0)
                    rows.append({
                        "entry_confidence": conf_f,
                        "realized_pnl": pnl_val if pnl_val is not None else 0.0,
                        "hold_days": hold_val if hold_val is not None else 0.0,
                    })
        except Exception as exc:
            print(f"[warning] Failed to read {path.name}: {exc}")
            continue
    return files, rows


def _print_bins_table(title: str, rows: List[Dict[str, Any]]) -> None:
    print(title)
    print("  Bin            N   WinRate    AvgPnL   MedianPnL  AvgHold")
    for row in rows:
        low_sample = " LOW SAMPLE" if row["low_sample"] else ""
        print(
            f"  {row['bin']:<12} {row['n_trades']:>4d} "
            f"{row['win_rate']:>8.2f}% {row['avg_pnl']:>9.2f} "
            f"{row['median_pnl']:>10.2f} {row['avg_hold_days']:>8.2f}{low_sample}"
        )


def _write_aggregated_csv(
    out_path: Path,
    bins_000: List[Dict[str, Any]],
    bins_050: List[Dict[str, Any]],
    bins_060: List[Dict[str, Any]],
) -> None:
    fieldnames = [
        "range",
        "bin",
        "bin_lo",
        "bin_hi",
        "n_trades",
        "win_rate",
        "avg_pnl",
        "median_pnl",
        "avg_hold_days",
        "low_sample",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in bins_050:
            out = dict(row)
            out["range"] = "0.50_to_1.00"
            writer.writerow(out)
        for row in bins_060:
            out = dict(row)
            out["range"] = "0.60_to_1.00"
            writer.writerow(out)
        for row in bins_000:
            out = dict(row)
            out["range"] = "0.00_to_1.00"
            writer.writerow(out)


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate confidence bins from *_TRADES_*.csv files")
    parser.add_argument("--reports-dir", type=str, default="reports", help="Directory containing trade ledgers")
    parser.add_argument("--low-sample-n", type=int, default=10, help="Threshold for LOW SAMPLE bin marker")
    args = parser.parse_args()

    reports_dir = Path(args.reports_dir)
    if not reports_dir.exists():
        print(f"Directory not found: {reports_dir}")
        return 1

    files, rows = _load_trade_rows(reports_dir)
    totals = _summarize_totals(rows)
    bins_050 = _compute_confidence_bins_0p02(
        rows, start=0.50, end=1.00, low_sample_n=args.low_sample_n,
    )
    bins_060 = _compute_confidence_bins_0p02(
        rows, start=0.60, end=1.00, low_sample_n=args.low_sample_n,
    )
    bins_000 = _compute_confidence_bins_0p02(
        rows, start=0.00, end=1.00, low_sample_n=args.low_sample_n,
    )

    print(f"Files scanned: {len(files)}")
    print(
        f"Totals: trades={totals['total_trades']}  win_rate={totals['win_rate']:.2f}%  "
        f"avg_pnl={totals['avg_pnl']:.2f}  median_pnl={totals['median_pnl']:.2f}"
    )
    print("")
    _print_bins_table("Confidence bins 0.50-1.00 (0.02):", bins_050)
    print("")
    _print_bins_table("Confidence bins 0.60-1.00 (0.02):", bins_060)
    print("")
    _print_bins_table("Confidence bins 0.00-1.00 (0.02):", bins_000)

    out_path = reports_dir / f"aggregated_bins_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    _write_aggregated_csv(out_path, bins_000, bins_050, bins_060)
    print(f"\nSaved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
