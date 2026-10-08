"""
Risk Analysis Module for Options Arbitrage Strategy
===================================================

This module calculates key quantitative risk, return, and drawdown metrics
from the completed backtest trade ledger and equity curve:

Key Metrics:
- Sharpe Ratio (Annualized Daily & Per-Trade)
- Sortino Ratio (Downside Volatility Adjusted)
- Maximum Drawdown (in ₹ and %)
- Drawdown Duration (Max & Average Duration to Recovery / Trough)
- Profit Factor (Gross and Net)
- Average Winning Trade
- Average Losing Trade
- Mathematical Expectancy (₹ per trade and per unit risk)
- Total Return (Net % on capital and absolute ₹ P&L)

Output Artifact:
- `data/processed/risk_metrics.csv`
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
logger = logging.getLogger("risk_analysis")


# =====================================================================
# DEFAULT CONFIGURATION
# =====================================================================

DEFAULT_INITIAL_CAPITAL = 250_000.0  # ₹250,000 default allocated margin capital
DEFAULT_ANNUAL_RISK_FREE_RATE = 0.06  # 6.0% Indian Risk-Free Rate
TRADING_DAYS_PER_YEAR = 252


# =====================================================================
# RISK ANALYZER
# =====================================================================

class StrategyRiskAnalyzer:
    """
    Computes professional quantitative risk and performance metrics
    from executed backtest results.
    """

    def __init__(
        self,
        trades_file: Path,
        equity_file: Path,
        output_file: Path,
        initial_capital: float = DEFAULT_INITIAL_CAPITAL,
        risk_free_rate: float = DEFAULT_ANNUAL_RISK_FREE_RATE,
    ):
        self.trades_file = trades_file
        self.equity_file = equity_file
        self.output_file = output_file
        self.initial_capital = initial_capital
        self.risk_free_rate = risk_free_rate

        self.output_file.parent.mkdir(parents=True, exist_ok=True)

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load trade ledger and equity curve."""
        if not self.trades_file.exists():
            raise FileNotFoundError(f"Trades file not found: {self.trades_file}")
        if not self.equity_file.exists():
            raise FileNotFoundError(f"Equity file not found: {self.equity_file}")

        trades_df = pd.read_parquet(self.trades_file)
        equity_df = pd.read_csv(self.equity_file)

        # Ensure datetime format
        trades_df["timestamp"] = pd.to_datetime(trades_df["timestamp"])
        equity_df["timestamp"] = pd.to_datetime(equity_df["timestamp"])

        return trades_df, equity_df

    def compute_drawdown_stats(
        self, equity_df: pd.DataFrame
    ) -> Tuple[float, float, float, float, float, int]:
        """
        Compute peak-to-trough maximum drawdown, max duration, and average duration.
        """
        equity = self.initial_capital + equity_df["cumulative_pnl"].values
        timestamps = equity_df["timestamp"].values

        peak = np.maximum.accumulate(equity)
        drawdown_rupees = peak - equity
        drawdown_pct = (drawdown_rupees / peak) * 100.0

        max_dd_rupees = float(np.max(drawdown_rupees))
        max_dd_pct = float(np.max(drawdown_pct))

        # Track Drawdown Periods
        in_drawdown = False
        start_time = None
        start_idx = None
        durations_days = []
        durations_trades = []

        for i in range(len(equity)):
            if equity[i] < peak[i]:
                if not in_drawdown:
                    in_drawdown = True
                    start_time = timestamps[i]
                    start_idx = i
            else:
                if in_drawdown:
                    in_drawdown = False
                    dur_days = (timestamps[i] - start_time) / np.timedelta64(1, "D")
                    dur_trades = i - start_idx
                    durations_days.append(dur_days)
                    durations_trades.append(dur_trades)

        # If backtest ends in drawdown, record active duration
        if in_drawdown:
            dur_days = (timestamps[-1] - start_time) / np.timedelta64(1, "D")
            dur_trades = len(equity) - start_idx
            durations_days.append(dur_days)
            durations_trades.append(dur_trades)

        max_duration_days = float(np.max(durations_days)) if durations_days else 0.0
        avg_duration_days = float(np.mean(durations_days)) if durations_days else 0.0
        max_duration_trades = int(np.max(durations_trades)) if durations_trades else 0

        return (
            max_dd_rupees,
            max_dd_pct,
            max_duration_days,
            avg_duration_days,
            max_duration_trades,
            len(durations_days),
        )

    def compute_daily_returns_and_ratios(
        self, trades_df: pd.DataFrame
    ) -> Tuple[float, float, float, float]:
        """
        Aggregate trade P&L into daily time series and compute annualized Sharpe & Sortino.
        """
        trades_df = trades_df.copy()
        trades_df["date"] = trades_df["timestamp"].dt.date

        daily_pnl = trades_df.groupby("date")["net_pnl"].sum()
        daily_returns = daily_pnl / self.initial_capital

        n_days = len(daily_returns)
        if n_days < 2:
            return 0.0, 0.0, 0.0, 0.0

        daily_rf = self.risk_free_rate / TRADING_DAYS_PER_YEAR
        excess_returns = daily_returns - daily_rf

        mean_excess = excess_returns.mean()
        std_returns = daily_returns.std()

        # Downside standard deviation (MAR = daily_rf)
        downside_diffs = np.minimum(0.0, daily_returns - daily_rf)
        downside_std = np.sqrt(np.mean(downside_diffs**2))

        # Annualized Sharpe & Sortino
        sharpe_ratio = (
            (mean_excess / std_returns) * np.sqrt(TRADING_DAYS_PER_YEAR)
            if (std_returns and std_returns > 0)
            else 0.0
        )
        sortino_ratio = (
            (mean_excess / downside_std) * np.sqrt(TRADING_DAYS_PER_YEAR)
            if (downside_std and downside_std > 0)
            else 0.0
        )

        return float(sharpe_ratio), float(sortino_ratio), float(std_returns * np.sqrt(TRADING_DAYS_PER_YEAR)), float(downside_std * np.sqrt(TRADING_DAYS_PER_YEAR))

    def run_analysis(self) -> pd.DataFrame:
        """
        Run full quantitative risk analysis and produce formatted results dataframe.
        """
        trades_df, equity_df = self.load_data()

        total_trades = len(trades_df)
        winning_trades_df = trades_df[trades_df["net_pnl"] > 0]
        losing_trades_df = trades_df[trades_df["net_pnl"] < 0]

        num_wins = len(winning_trades_df)
        num_losses = len(losing_trades_df)

        win_rate = (num_wins / total_trades) if total_trades > 0 else 0.0
        loss_rate = (num_losses / total_trades) if total_trades > 0 else 0.0

        # P&L Totals
        total_gross_pnl = float(trades_df["gross_pnl"].sum())
        total_slippage = float(trades_df["slippage"].sum())
        total_txn_costs = float(trades_df["transaction_costs"].sum())
        total_costs = total_slippage + total_txn_costs
        total_net_pnl = float(trades_df["net_pnl"].sum())

        # Total Return (%)
        total_return_pct = (total_net_pnl / self.initial_capital) * 100.0

        # Average Wins & Losses
        avg_winning_trade = (
            float(winning_trades_df["net_pnl"].mean()) if num_wins > 0 else 0.0
        )
        avg_losing_trade = (
            float(abs(losing_trades_df["net_pnl"].mean())) if num_losses > 0 else 0.0
        )

        # Profit Factor
        gross_wins = trades_df.loc[trades_df["gross_pnl"] > 0, "gross_pnl"].sum()
        gross_losses = abs(trades_df.loc[trades_df["gross_pnl"] < 0, "gross_pnl"].sum())
        profit_factor = (gross_wins / gross_losses) if gross_losses > 0 else np.nan

        net_wins = winning_trades_df["net_pnl"].sum()
        net_losses = abs(losing_trades_df["net_pnl"].sum())
        net_profit_factor = (net_wins / net_losses) if net_losses > 0 else np.nan

        # Expectancy
        # E = (Win Rate * Avg Win) - (Loss Rate * Avg Loss)
        expectancy_rupees = (win_rate * avg_winning_trade) - (loss_rate * avg_losing_trade)
        expectancy_ratio = (
            (win_rate * (avg_winning_trade / avg_losing_trade)) - loss_rate
            if avg_losing_trade > 0
            else np.nan
        )

        # Drawdown Metrics
        (
            max_dd_rupees,
            max_dd_pct,
            max_dd_duration_days,
            avg_dd_duration_days,
            max_dd_duration_trades,
            total_dd_periods,
        ) = self.compute_drawdown_stats(equity_df)

        # Sharpe & Sortino
        (
            sharpe_ratio,
            sortino_ratio,
            annualized_volatility,
            downside_deviation,
        ) = self.compute_daily_returns_and_ratios(trades_df)

        # Format output metrics table
        metrics = [
            {"Category": "Returns", "Metric": "Initial Capital", "Value": f"Rs {self.initial_capital:,.2f}"},
            {"Category": "Returns", "Metric": "Total Gross P&L", "Value": f"Rs {total_gross_pnl:,.2f}"},
            {"Category": "Returns", "Metric": "Total Transaction Costs & Slippage", "Value": f"Rs {total_costs:,.2f}"},
            {"Category": "Returns", "Metric": "Total Net P&L", "Value": f"Rs {total_net_pnl:,.2f}"},
            {"Category": "Returns", "Metric": "Total Return (%)", "Value": f"{total_return_pct:.2f}%"},
            {"Category": "Trade Performance", "Metric": "Total Trades", "Value": f"{total_trades}"},
            {"Category": "Trade Performance", "Metric": "Win Rate (%)", "Value": f"{win_rate * 100:.2f}%"},
            {"Category": "Trade Performance", "Metric": "Average Winning Trade", "Value": f"Rs {avg_winning_trade:,.2f}"},
            {"Category": "Trade Performance", "Metric": "Average Losing Trade", "Value": f"Rs {avg_losing_trade:,.2f}"},
            {"Category": "Trade Performance", "Metric": "Win / Loss Ratio", "Value": f"{(avg_winning_trade / avg_losing_trade):.2f}" if avg_losing_trade > 0 else "N/A"},
            {"Category": "Trade Performance", "Metric": "Profit Factor (Gross)", "Value": f"{profit_factor:.2f}"},
            {"Category": "Trade Performance", "Metric": "Profit Factor (Net)", "Value": f"{net_profit_factor:.2f}"},
            {"Category": "Trade Performance", "Metric": "Expectancy (₹ / trade)", "Value": f"Rs {expectancy_rupees:,.2f}"},
            {"Category": "Trade Performance", "Metric": "Expectancy Ratio", "Value": f"{expectancy_ratio:.2f}"},
            {"Category": "Risk & Volatility", "Metric": "Sharpe Ratio (Annualized)", "Value": f"{sharpe_ratio:.2f}"},
            {"Category": "Risk & Volatility", "Metric": "Sortino Ratio (Annualized)", "Value": f"{sortino_ratio:.2f}"},
            {"Category": "Risk & Volatility", "Metric": "Annualized Volatility (%)", "Value": f"{annualized_volatility * 100:.2f}%"},
            {"Category": "Risk & Volatility", "Metric": "Downside Deviation (%)", "Value": f"{downside_deviation * 100:.2f}%"},
            {"Category": "Drawdown Analysis", "Metric": "Maximum Drawdown (₹)", "Value": f"Rs {max_dd_rupees:,.2f}"},
            {"Category": "Drawdown Analysis", "Metric": "Maximum Drawdown (%)", "Value": f"{max_dd_pct:.2f}%"},
            {"Category": "Drawdown Analysis", "Metric": "Maximum Drawdown Duration", "Value": f"{max_dd_duration_days:.1f} days ({max_dd_duration_trades} trades)"},
            {"Category": "Drawdown Analysis", "Metric": "Average Drawdown Duration", "Value": f"{avg_dd_duration_days:.1f} days"},
        ]

        metrics_df = pd.DataFrame(metrics)
        metrics_df.to_csv(self.output_file, index=False)
        logger.info(f"Risk metrics successfully saved to: {self.output_file}")

        return metrics_df


# =====================================================================
# CLI ENTRYPOINT
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compute risk, return, and drawdown metrics for options arbitrage backtest."
    )
    parser.add_argument(
        "--trades-file",
        type=Path,
        default=Path("data/processed/backtest_trades.parquet"),
        help="Path to executed backtest trades parquet file.",
    )
    parser.add_argument(
        "--equity-file",
        type=Path,
        default=Path("data/processed/equity_curve.csv"),
        help="Path to equity curve CSV file.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("data/processed/risk_metrics.csv"),
        help="Path to output risk metrics CSV file.",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=DEFAULT_INITIAL_CAPITAL,
        help="Initial allocated margin capital in INR (default: 250,000).",
    )
    parser.add_argument(
        "--risk-free-rate",
        type=float,
        default=DEFAULT_ANNUAL_RISK_FREE_RATE,
        help="Annualized risk-free rate (default: 0.06).",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("OPTIONS ARBITRAGE STRATEGY - RISK & DRAWDOWN ANALYSIS")
    print("=" * 80)
    print(f"Trades file:        {args.trades_file.resolve()}")
    print(f"Equity curve file:  {args.equity_file.resolve()}")
    print(f"Initial Capital:    Rs {args.capital:,.2f}")
    print(f"Risk-free rate:     {args.risk_free_rate * 100:.1f}%")
    print(f"Output file:        {args.output_file.resolve()}")
    print("=" * 80)

    analyzer = StrategyRiskAnalyzer(
        trades_file=args.trades_file,
        equity_file=args.equity_file,
        output_file=args.output_file,
        initial_capital=args.capital,
        risk_free_rate=args.risk_free_rate,
    )

    metrics_df = analyzer.run_analysis()

    print("\n" + "=" * 80)
    print("FINAL RISK & PERFORMANCE METRICS")
    print("=" * 80)
    current_cat = None
    for _, row in metrics_df.iterrows():
        if row["Category"] != current_cat:
            current_cat = row["Category"]
            print(f"\n--- {current_cat.upper()} ---")
        print(f"  {row['Metric']:<36}: {row['Value']}")
    print("=" * 80)


if __name__ == "__main__":
    main()
