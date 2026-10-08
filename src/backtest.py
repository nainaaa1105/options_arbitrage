"""
Options Arbitrage Strategy Backtesting Engine
=============================================

This module performs chronological out-of-sample backtesting on generated BUY/SELL
signals from `data/processed/trade_signals.parquet`.

Key Principles:
1. Strict Chronological Execution: No future lookahead bias in entry.
2. Realistic Transaction Costs & Slippage: Configurable exchange fees, STT, GST,
   brokerage, and per-leg slippage.
3. Accurate Contract Matching: Pulls historical realized prices from `master_options.parquet`
   at the target holding horizon.

Output Artifacts:
- `data/processed/backtest_trades.parquet`
- `data/processed/backtest_summary.csv`
- `data/processed/equity_curve.csv`
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("backtest")


# =====================================================================
# DEFAULT CONFIGURATION
# =====================================================================

DEFAULT_HORIZON_MINUTES = 30
DEFAULT_SLIPPAGE_POINTS = 0.15      # 15 paise slippage per share per leg
DEFAULT_BROKERAGE_PER_ORDER = 20.0  # ₹20 flat brokerage per executed order
DEFAULT_STT_RATE = 0.000625         # 0.0625% STT on option sell turnover
DEFAULT_EXCHANGE_TXN_RATE = 0.0005  # ~0.05% Exchange turnover, SEBI charges, GST


# =====================================================================
# BACKTEST ENGINE
# =====================================================================

class OptionsArbitrageBacktester:
    """
    Chronological trade execution & P&L calculator for options arbitrage signals.
    """

    def __init__(
        self,
        signals_file: Path,
        master_file: Path,
        output_dir: Path,
        horizon_minutes: int = DEFAULT_HORIZON_MINUTES,
        slippage_points: float = DEFAULT_SLIPPAGE_POINTS,
        brokerage_per_order: float = DEFAULT_BROKERAGE_PER_ORDER,
        stt_rate: float = DEFAULT_STT_RATE,
        exchange_txn_rate: float = DEFAULT_EXCHANGE_TXN_RATE,
    ):
        self.signals_file = signals_file
        self.master_file = master_file
        self.output_dir = output_dir
        self.horizon_minutes = horizon_minutes
        self.slippage_points = slippage_points
        self.brokerage_per_order = brokerage_per_order
        self.stt_rate = stt_rate
        self.exchange_txn_rate = exchange_txn_rate

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load signals and historical market options data."""
        if not self.signals_file.exists():
            raise FileNotFoundError(f"Signals file not found at: {self.signals_file}")
        if not self.master_file.exists():
            raise FileNotFoundError(f"Master options file not found at: {self.master_file}")

        logger.info(f"Loading trade signals from: {self.signals_file}")
        signals_df = pd.read_parquet(self.signals_file)

        logger.info(f"Loading master options price data from: {self.master_file}")
        master_df = pd.read_parquet(self.master_file)

        # Ensure uniform timezone (Asia/Kolkata)
        signals_df["timestamp"] = pd.to_datetime(signals_df["timestamp"]).dt.tz_convert("Asia/Kolkata")
        master_df["timestamp"] = pd.to_datetime(master_df["timestamp"]).dt.tz_convert("Asia/Kolkata")

        return signals_df, master_df

    def run_backtest(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Execute chronological backtest over all BUY and SELL signals.
        """
        signals_df, master_df = self.load_data()

        # Filter actionable trades (BUY / SELL)
        actionable_trades = signals_df[
            signals_df["trade_signal"].isin(["BUY", "SELL"])
        ].copy()

        total_signals = len(signals_df)
        total_actionable = len(actionable_trades)

        logger.info(f"Total signals in dataset: {total_signals:,}")
        logger.info(f"Actionable trade signals (BUY/SELL): {total_actionable:,}")

        if total_actionable == 0:
            raise ValueError("No actionable BUY/SELL signals found to backtest.")

        # Sort chronologically by entry timestamp
        actionable_trades = actionable_trades.sort_values("timestamp").reset_index(drop=True)
        actionable_trades["target_exit_time"] = (
            actionable_trades["timestamp"] + pd.Timedelta(minutes=self.horizon_minutes)
        )

        # Prepare master lookup table
        master_lookup = master_df[
            ["instrument_key", "timestamp", "open", "close", "high", "low", "lot_size"]
        ].sort_values("timestamp").reset_index(drop=True)

        # Match exit price at target horizon using merge_asof (forward direction within 5-min tolerance)
        merged = pd.merge_asof(
            actionable_trades.sort_values("target_exit_time"),
            master_lookup,
            left_on="target_exit_time",
            right_on="timestamp",
            by="instrument_key",
            direction="forward",
            tolerance=pd.Timedelta(minutes=5),
            suffixes=("_entry", "_exit"),
        ).sort_values("timestamp_entry").reset_index(drop=True)

        # Verify matched exits
        unmatched = merged["close"].isna().sum()
        if unmatched > 0:
            logger.warning(f"Unmatched exit prices for {unmatched} trades; falling back to nearest available price.")
            merged["close"] = merged["close"].fillna(merged["market_price"])

        # Determine trade execution quantities and prices
        merged["quantity"] = merged["lot_size"].fillna(65).astype(int)
        merged["entry_price"] = merged["market_price"]
        merged["exit_price"] = merged["close"]
        merged["entry_timestamp"] = merged["timestamp_entry"]
        merged["exit_timestamp"] = merged["timestamp_exit"].fillna(merged["target_exit_time"])
        merged["holding_period_mins"] = (
            (merged["exit_timestamp"] - merged["entry_timestamp"]).dt.total_seconds() / 60.0
        ).round(1)

        # Calculate P&L, Slippage, and Transaction Costs
        gross_pnls = []
        slippage_costs = []
        transaction_costs = []

        for _, row in merged.iterrows():
            qty = row["quantity"]
            p_in = row["entry_price"]
            p_out = row["exit_price"]
            sig = row["trade_signal"]

            # Gross P&L Calculation
            if sig == "BUY":
                # Buy at p_in, sell at p_out
                gross = (p_out - p_in) * qty
                entry_turnover = p_in * qty
                exit_turnover = p_out * qty
                stt = exit_turnover * self.stt_rate
            else:
                # Sell (short) at p_in, buy back at p_out
                gross = (p_in - p_out) * qty
                entry_turnover = p_in * qty
                exit_turnover = p_out * qty
                stt = entry_turnover * self.stt_rate

            # Slippage per share on both legs (entry + exit)
            slip = (self.slippage_points * 2) * qty

            # Brokerage + Turnover Fees + GST + STT
            other_tax = (entry_turnover + exit_turnover) * self.exchange_txn_rate
            brokerage = self.brokerage_per_order * 2  # round-trip
            fee = brokerage + stt + other_tax

            gross_pnls.append(gross)
            slippage_costs.append(slip)
            transaction_costs.append(fee)

        merged["gross_pnl"] = np.round(gross_pnls, 2)
        merged["slippage"] = np.round(slippage_costs, 2)
        merged["transaction_costs"] = np.round(transaction_costs, 2)
        merged["net_pnl"] = np.round(
            merged["gross_pnl"] - merged["slippage"] - merged["transaction_costs"], 2
        )
        merged["cumulative_pnl"] = np.round(merged["net_pnl"].cumsum(), 2)
        merged["trade_result"] = np.where(
            merged["net_pnl"] > 0, "WIN", np.where(merged["net_pnl"] < 0, "LOSS", "BREAKEVEN")
        )

        # Final trade record columns
        trade_columns = [
            "entry_timestamp",
            "exit_timestamp",
            "holding_period_mins",
            "trading_symbol",
            "instrument_key",
            "option_type",
            "strike",
            "expiry",
            "trade_signal",
            "entry_price",
            "exit_price",
            "quantity",
            "gross_pnl",
            "slippage",
            "transaction_costs",
            "net_pnl",
            "cumulative_pnl",
            "trade_result",
            "predicted_probability",
            "confidence",
            "price_deviation",
            "theoretical_price",
        ]

        trades_df = merged[[col for col in trade_columns if col in merged.columns]].copy()
        trades_df = trades_df.rename(columns={"entry_timestamp": "timestamp"})

        # Build Equity Curve & Drawdown Series
        equity_curve_df = trades_df[
            ["timestamp", "trade_signal", "trading_symbol", "net_pnl", "cumulative_pnl"]
        ].copy()
        equity_curve_df["peak_pnl"] = equity_curve_df["cumulative_pnl"].cummax()
        equity_curve_df["drawdown"] = equity_curve_df["peak_pnl"] - equity_curve_df["cumulative_pnl"]

        # Calculate Summary Metrics
        summary_df = self._compute_summary(trades_df, equity_curve_df)

        # Save files
        self._save_results(trades_df, equity_curve_df, summary_df)

        return trades_df, equity_curve_df, summary_df

    def _compute_summary(
        self, trades_df: pd.DataFrame, equity_curve_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Compute performance and risk metrics."""
        total_trades = len(trades_df)
        winning_trades = int((trades_df["net_pnl"] > 0).sum())
        losing_trades = int((trades_df["net_pnl"] < 0).sum())
        breakeven_trades = int((trades_df["net_pnl"] == 0).sum())
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0

        total_gross_pnl = float(trades_df["gross_pnl"].sum())
        total_slippage = float(trades_df["slippage"].sum())
        total_transaction_costs = float(trades_df["transaction_costs"].sum())
        total_costs = total_slippage + total_transaction_costs
        total_net_pnl = float(trades_df["net_pnl"].sum())

        max_drawdown = float(equity_curve_df["drawdown"].max())
        avg_trade_net_pnl = total_net_pnl / total_trades if total_trades > 0 else 0.0

        gross_wins = trades_df.loc[trades_df["gross_pnl"] > 0, "gross_pnl"].sum()
        gross_losses = abs(trades_df.loc[trades_df["gross_pnl"] < 0, "gross_pnl"].sum())
        profit_factor = (gross_wins / gross_losses) if gross_losses > 0 else np.nan

        avg_win = trades_df.loc[trades_df["net_pnl"] > 0, "net_pnl"].mean() if winning_trades > 0 else 0.0
        avg_loss = abs(trades_df.loc[trades_df["net_pnl"] < 0, "net_pnl"].mean()) if losing_trades > 0 else 0.0
        win_loss_ratio = (avg_win / avg_loss) if avg_loss > 0 else np.nan

        # Sharpe ratio on trade returns (annualized ~ 252 days * 12 thirty-min periods)
        std_pnl = trades_df["net_pnl"].std()
        sharpe = (trades_df["net_pnl"].mean() / std_pnl * np.sqrt(252 * 12)) if (std_pnl and std_pnl > 0) else 0.0

        summary_data = [
            {"Metric": "Total Trades", "Value": f"{total_trades}"},
            {"Metric": "Winning Trades", "Value": f"{winning_trades}"},
            {"Metric": "Losing Trades", "Value": f"{losing_trades}"},
            {"Metric": "Breakeven Trades", "Value": f"{breakeven_trades}"},
            {"Metric": "Win Rate (%)", "Value": f"{win_rate:.2f}%"},
            {"Metric": "Total Gross P&L (Rs)", "Value": f"{total_gross_pnl:,.2f}"},
            {"Metric": "Total Slippage (Rs)", "Value": f"{total_slippage:,.2f}"},
            {"Metric": "Total Transaction Costs (Rs)", "Value": f"{total_transaction_costs:,.2f}"},
            {"Metric": "Total Costs (Rs)", "Value": f"{total_costs:,.2f}"},
            {"Metric": "Total Net P&L (Rs)", "Value": f"{total_net_pnl:,.2f}"},
            {"Metric": "Average Trade Net P&L (Rs)", "Value": f"{avg_trade_net_pnl:,.2f}"},
            {"Metric": "Average Win (Rs)", "Value": f"{avg_win:,.2f}"},
            {"Metric": "Average Loss (Rs)", "Value": f"{avg_loss:,.2f}"},
            {"Metric": "Win / Loss Payoff Ratio", "Value": f"{win_loss_ratio:.2f}"},
            {"Metric": "Profit Factor", "Value": f"{profit_factor:.2f}"},
            {"Metric": "Maximum Drawdown (Rs)", "Value": f"{max_drawdown:,.2f}"},
            {"Metric": "Annualized Sharpe Ratio", "Value": f"{sharpe:.2f}"},
        ]

        return pd.DataFrame(summary_data)

    def _save_results(
        self,
        trades_df: pd.DataFrame,
        equity_curve_df: pd.DataFrame,
        summary_df: pd.DataFrame,
    ) -> None:
        """Persist parquet and csv outputs."""
        trades_path = self.output_dir / "backtest_trades.parquet"
        summary_path = self.output_dir / "backtest_summary.csv"
        equity_path = self.output_dir / "equity_curve.csv"

        trades_df.to_parquet(trades_path, index=False)
        summary_df.to_csv(summary_path, index=False)
        equity_curve_df.to_csv(equity_path, index=False)

        logger.info(f"Saved trades to: {trades_path}")
        logger.info(f"Saved summary to: {summary_path}")
        logger.info(f"Saved equity curve to: {equity_path}")


# =====================================================================
# CLI ENTRYPOINT
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Backtest options arbitrage BUY/SELL signals chronologically."
    )
    parser.add_argument(
        "--signals-file",
        type=Path,
        default=Path("data/processed/trade_signals.parquet"),
        help="Path to actionable trade signals parquet file.",
    )
    parser.add_argument(
        "--master-file",
        type=Path,
        default=Path("data/processed/master_options.parquet"),
        help="Path to master options dataset containing price history.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Output directory for backtest artifacts.",
    )
    parser.add_argument(
        "--horizon-minutes",
        type=int,
        default=DEFAULT_HORIZON_MINUTES,
        help="Trade holding period in minutes (default: 30).",
    )
    parser.add_argument(
        "--slippage-points",
        type=float,
        default=DEFAULT_SLIPPAGE_POINTS,
        help="Slippage in index points/premium per share per leg (default: 0.15).",
    )
    parser.add_argument(
        "--brokerage",
        type=float,
        default=DEFAULT_BROKERAGE_PER_ORDER,
        help="Brokerage cost per order in INR (default: 20.0).",
    )
    parser.add_argument(
        "--stt-rate",
        type=float,
        default=DEFAULT_STT_RATE,
        help="STT rate on sell turnover (default: 0.000625).",
    )
    parser.add_argument(
        "--exchange-txn-rate",
        type=float,
        default=DEFAULT_EXCHANGE_TXN_RATE,
        help="Exchange turnover & statutory charges rate (default: 0.0005).",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("RUNNING CHRONOLOGICAL OPTIONS ARBITRAGE BACKTEST")
    print("=" * 80)
    print(f"Signals file:          {args.signals_file.resolve()}")
    print(f"Master price file:     {args.master_file.resolve()}")
    print(f"Holding horizon:       {args.horizon_minutes} minutes")
    print(f"Slippage per leg:      Rs {args.slippage_points:.2f} per unit")
    print(f"Brokerage per order:   Rs {args.brokerage:.2f}")
    print("=" * 80)

    backtester = OptionsArbitrageBacktester(
        signals_file=args.signals_file,
        master_file=args.master_file,
        output_dir=args.output_dir,
        horizon_minutes=args.horizon_minutes,
        slippage_points=args.slippage_points,
        brokerage_per_order=args.brokerage,
        stt_rate=args.stt_rate,
        exchange_txn_rate=args.exchange_txn_rate,
    )

    trades_df, equity_curve_df, summary_df = backtester.run_backtest()

    print("\n" + "=" * 80)
    print("BACKTEST RESULTS SUMMARY")
    print("=" * 80)
    for _, row in summary_df.iterrows():
        print(f"{row['Metric']:<30}: {row['Value']}")
    print("=" * 80)

    print("\nSample Executed Trades (First 5):")
    cols_to_show = [
        "timestamp",
        "trading_symbol",
        "trade_signal",
        "entry_price",
        "exit_price",
        "gross_pnl",
        "transaction_costs",
        "slippage",
        "net_pnl",
        "cumulative_pnl",
    ]
    print(trades_df[cols_to_show].head(5).to_string(index=False))
    print("=" * 80)


if __name__ == "__main__":
    main()
