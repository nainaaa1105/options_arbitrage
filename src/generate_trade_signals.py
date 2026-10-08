"""
generate_trade_signals.py
=========================
Converts ML convergence predictions and options pricing deviations into actionable
trade decisions: BUY, SELL, HOLD.

Rules:
------
- Only generate BUY/SELL when:
    1. Model confidence >= 0.20 (P >= 0.70 for convergence)
    2. Pricing deviation supports the trade direction:
       - Undervalued (market_price < theoretical_price or price_deviation < 0) -> BUY
       - Overvalued  (market_price > theoretical_price or price_deviation > 0) -> SELL
- Otherwise -> HOLD

Input:  data/processed/trading_signals.parquet
Output: data/processed/trade_signals.parquet
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# PATHS & CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_FILE  = PROJECT_ROOT / "data" / "processed" / "trading_signals.parquet"
DEFAULT_OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "trade_signals.parquet"
DEFAULT_CONF_THRESHOLD = 0.20


def generate_actionable_trade_signals(
    input_path: Path = DEFAULT_INPUT_FILE,
    output_path: Path = DEFAULT_OUTPUT_FILE,
    confidence_threshold: float = DEFAULT_CONF_THRESHOLD,
    min_abs_deviation: float = 0.0
) -> pd.DataFrame:
    """
    Converts ML predictions and pricing deviations into actionable BUY / SELL / HOLD signals.
    """
    print("=" * 80)
    print("GENERATING ACTIONABLE OPTIONS TRADE SIGNALS (BUY / SELL / HOLD)")
    print("=" * 80)
    print(f"Input file:           {input_path}")
    print(f"Confidence threshold: {confidence_threshold:.2f} (Required P >= {0.50 + confidence_threshold:.2f})")
    print(f"Output destination:   {output_path}")

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input trading signals not found at {input_path}. "
            f"Please run src/generate_signals.py first."
        )

    df = pd.read_parquet(input_path).copy()
    print(f"Loaded records:       {len(df):,} rows")

    # Validate required columns
    required_cols = [
        "option_type", "market_price", "theoretical_price",
        "price_deviation", "predicted_probability"
    ]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in input file: {missing}")

    # Recompute confidence if needed
    if "confidence" not in df.columns:
        df["confidence"] = (df["predicted_probability"] - 0.50).abs()

    prob = df["predicted_probability"]
    dev  = df["price_deviation"]
    conf = df["confidence"]

    # Convergence threshold (P >= 0.50 + confidence_threshold)
    min_prob_convergence = 0.50 + confidence_threshold

    # Conditions for Actionable Signals
    is_confident_convergence = (conf >= confidence_threshold) & (prob >= min_prob_convergence)
    is_valid_deviation       = dev.abs() >= min_abs_deviation

    is_buy  = is_confident_convergence & is_valid_deviation & (dev < 0)  # Undervalued -> BUY
    is_sell = is_confident_convergence & is_valid_deviation & (dev > 0)  # Overvalued  -> SELL

    # Assign trade_signal
    trade_signals = np.where(
        is_buy,
        "BUY",
        np.where(is_sell, "SELL", "HOLD")
    )

    df["trade_signal"]   = trade_signals
    df["is_actionable"]   = df["trade_signal"].isin(["BUY", "SELL"])
    df["trade_rationale"] = np.where(
        is_buy,
        f"Undervalued (Market < Fair) & High Conv. Prob (P={prob.round(3)})",
        np.where(
            is_sell,
            f"Overvalued (Market > Fair) & High Conv. Prob (P={prob.round(3)})",
            "Insufficient Confidence / Non-Converging"
        )
    )

    # Save to parquet
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"\nSaved trade signals to: {output_path}")

    # Counts & Metrics
    total_signals = len(df)
    buy_count     = int((df["trade_signal"] == "BUY").sum())
    sell_count    = int((df["trade_signal"] == "SELL").sum())
    hold_count    = int((df["trade_signal"] == "HOLD").sum())
    actionable    = buy_count + sell_count
    pct_actionable = (actionable / total_signals) * 100.0 if total_signals > 0 else 0.0

    print("\n" + "=" * 80)
    print("ACTIONABLE TRADE SIGNALS BREAKDOWN")
    print("=" * 80)
    print(f"Total signals:                    {total_signals:,}")
    print(f"BUY count:                        {buy_count:,}")
    print(f"SELL count:                       {sell_count:,}")
    print(f"HOLD count:                       {hold_count:,}")
    print(f"Total actionable (BUY + SELL):    {actionable:,}")
    print(f"Percentage of actionable signals: {pct_actionable:.2f}%")
    print("=" * 80)

    # Breakdown by Option Type
    if "option_type" in df.columns:
        print("\nActionable Signals by Option Type:")
        summary_table = pd.crosstab(
            df["option_type"],
            df["trade_signal"],
            margins=True,
            margins_name="Total"
        )
        print(summary_table.to_string())
        print("=" * 80)

    return df


# ============================================================
# CLI INTERFACE
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate actionable BUY / SELL / HOLD signals from options arbitrage predictions."
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default=str(DEFAULT_INPUT_FILE),
        help="Path to input trading_signals.parquet file."
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=str(DEFAULT_OUTPUT_FILE),
        help="Path to output trade_signals.parquet file."
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=DEFAULT_CONF_THRESHOLD,
        help=f"Minimum confidence margin (default: {DEFAULT_CONF_THRESHOLD:.2f})."
    )
    parser.add_argument(
        "--min-abs-deviation",
        type=float,
        default=0.0,
        help="Minimum absolute price deviation required to trade (default: 0.0)."
    )
    args = parser.parse_args()

    generate_actionable_trade_signals(
        input_path=Path(args.input_file),
        output_path=Path(args.output_file),
        confidence_threshold=args.confidence_threshold,
        min_abs_deviation=args.min_abs_deviation
    )


if __name__ == "__main__":
    main()
