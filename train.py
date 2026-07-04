"""
Master training script — NSE Intraday Trading System.

Runs the full pipeline in order, or a specific phase if requested.

Pipeline:
  Phase 0  feature_engineering.py  — compute all indicators, save parquets
  Phase 3  sac_trainer.py          — train SAC multi-output agent

Usage:
  python train.py                  # run all phases (0 → 3)
  python train.py --phase 0        # feature engineering only
  python train.py --phase 3        # SAC training only (needs phase 0)

Expected wall-clock times on CPU:
  Phase 0: ~15 min   (204k bars × 17 CSVs, rolling indicator computation)
  Phase 3: ~4–8 hr   (500k SAC timesteps; use --timesteps to shorten)

Outputs:
  features/NIFTY_50_features.parquet
  features/NIFTY_BANK_features.parquet
  models/sac_multi/best_model.zip
  models/sac_multi/final_model.zip
  models/sac_multi/metadata.json
"""

import argparse
import sys
import time
from pathlib import Path


def _banner(title: str) -> None:
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def _elapsed(start: float) -> str:
    s = int(time.time() - start)
    return f"{s // 60}m {s % 60}s"


def run_phase0():
    _banner("Phase 0 — Feature Engineering")
    t = time.time()
    from feature_engineering import run_feature_engineering
    run_feature_engineering()
    print(f"Phase 0 done in {_elapsed(t)}")


def run_phase3(timesteps: int | None = None):
    _banner("Phase 3 — SAC Multi-Output Agent Training")
    t = time.time()
    if timesteps:
        import sac_trainer
        sac_trainer.TOTAL_TIMESTEPS = timesteps
    from sac_trainer import run_sac_training
    run_sac_training()
    print(f"Phase 3 done in {_elapsed(t)}")


def check_prerequisites(phase: str) -> None:
    """Fail fast with a helpful message if required outputs are missing."""
    parquet_n50 = Path("features/NIFTY_50_features.parquet")

    if "0" not in phase and "3" in phase:
        if not parquet_n50.exists():
            print("ERROR: features/NIFTY_50_features.parquet not found.")
            print("       Run Phase 0 first:  python train.py --phase 0")
            sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="NSE Intraday Trading System — master trainer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--phase",
        type=str,
        default="all",
        help="Phase(s) to run: 0, 3, or 'all' (default).",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Override TOTAL_TIMESTEPS for SAC training (default: 500000).",
    )
    args = parser.parse_args()

    phase = args.phase.lower()
    if phase == "all":
        phase = "03"

    print(f"NSE Intraday Trading System — running phase(s): {phase}")
    check_prerequisites(phase)

    total_start = time.time()

    if "0" in phase:
        run_phase0()

    if "3" in phase:
        run_phase3(timesteps=args.timesteps)

    _banner("All done")
    print(f"Total time: {_elapsed(total_start)}")
    print("\nNext steps:")
    print("  1. Inspect validation metrics printed during SAC training")
    print("  2. Once Sharpe > 1.5 and MaxDD < 15% on val set:")
    print("     python backtest.py   # final evaluation on 2024 test set")
    print("  3. Paper trade ≥ 1 month before any live capital")


if __name__ == "__main__":
    main()
