"""Risk and return metrics from the backtest ledger and full market calendar."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


MARKET_TZ = "Asia/Kolkata"
TRADING_DAYS_PER_YEAR = 252


class StrategyRiskAnalyzer:
    def __init__(self, trades_file, equity_file, output_file,
                 initial_capital=250_000.0, risk_free_rate=0.06,
                 calendar_file=None):
        self.trades_file = Path(trades_file)
        self.equity_file = Path(equity_file)
        self.output_file = Path(output_file)
        self.initial_capital = float(initial_capital)
        self.risk_free_rate = float(risk_free_rate)
        self.calendar_file = Path(calendar_file) if calendar_file else (
            self.trades_file.parent / "master_options.parquet"
        )

    @staticmethod
    def _dates(series):
        return pd.to_datetime(series, errors="coerce").dt.date

    def _calendar(self, equity_df):
        if self.calendar_file.exists():
            market = pd.read_parquet(self.calendar_file, columns=["trade_date"])
            dates = pd.to_datetime(market.trade_date, errors="coerce").dt.date.dropna().unique()
            return pd.Index(sorted(dates))
        if "trade_date" in equity_df:
            return pd.Index(sorted(self._dates(equity_df.trade_date).dropna().unique()))
        # Preserve known dates when no source calendar is available; do not
        # invent market sessions or fill weekends/holidays as trading days.
        return pd.Index(sorted(self._dates(equity_df.timestamp).dropna().unique()))

    def load_data(self):
        if not self.trades_file.exists():
            raise FileNotFoundError(self.trades_file)
        if not self.equity_file.exists():
            raise FileNotFoundError(self.equity_file)
        trades = pd.read_parquet(self.trades_file)
        equity = pd.read_csv(self.equity_file)
        if "timestamp" in trades:
            trades["timestamp"] = pd.to_datetime(trades.timestamp, errors="coerce")
        if "exit_timestamp" in trades:
            trades["exit_timestamp"] = pd.to_datetime(trades.exit_timestamp, errors="coerce")
        equity["timestamp"] = pd.to_datetime(equity.timestamp, errors="coerce")
        return trades, equity

    def build_daily_series(self, trades, equity_df):
        dates = self._calendar(equity_df)
        if len(trades):
            event_time = trades["exit_timestamp"] if "exit_timestamp" in trades else trades["timestamp"]
            pnl = pd.to_numeric(trades.net_pnl, errors="coerce").fillna(0.0)
            daily_pnl = pnl.groupby(event_time.dt.date).sum()
        else:
            daily_pnl = pd.Series(dtype=float)
        daily = pd.DataFrame({"trade_date": dates})
        daily["net_pnl"] = daily.trade_date.map(daily_pnl).fillna(0.0)
        daily["cumulative_pnl"] = daily.net_pnl.cumsum()
        daily["equity"] = self.initial_capital + daily.cumulative_pnl
        daily["daily_return"] = daily.net_pnl / daily.equity.shift(1).fillna(self.initial_capital)
        daily["peak_equity"] = daily.equity.cummax()
        daily["drawdown"] = daily.peak_equity - daily.equity
        daily["drawdown_pct"] = daily.drawdown / daily.peak_equity.replace(0, np.nan)
        daily["timestamp"] = pd.to_datetime(daily.trade_date.astype(str)).dt.tz_localize(MARKET_TZ) + pd.Timedelta(hours=15, minutes=30)
        return daily

    def run_analysis(self):
        trades, existing_equity = self.load_data()
        daily = self.build_daily_series(trades, existing_equity)
        n = len(trades)
        net = pd.to_numeric(trades.get("net_pnl", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
        gross = pd.to_numeric(trades.get("gross_pnl", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
        wins, losses = net[net > 0], net[net < 0]
        gross_wins, gross_losses = gross[gross > 0].sum(), abs(gross[gross < 0].sum())
        net_wins, net_losses = wins.sum(), abs(losses.sum())
        if "total_costs" in trades:
            costs = pd.to_numeric(trades["total_costs"], errors="coerce").fillna(0.0)
        else:
            costs = pd.to_numeric(
                trades.get("transaction_costs", pd.Series(index=trades.index, dtype=float)),
                errors="coerce",
            ).fillna(0.0)
            for component in ("spread_cost", "slippage"):
                if component in trades:
                    costs = costs + pd.to_numeric(trades[component], errors="coerce").fillna(0.0)
        rf_daily = self.risk_free_rate / TRADING_DAYS_PER_YEAR
        excess = daily.daily_return - rf_daily
        daily_std = daily.daily_return.std(ddof=1)
        downside_std = np.sqrt(np.mean(np.minimum(daily.daily_return - rf_daily, 0.0) ** 2)) if len(daily) else np.nan
        sharpe = excess.mean() / daily_std * np.sqrt(TRADING_DAYS_PER_YEAR) if daily_std and np.isfinite(daily_std) else np.nan
        sortino = excess.mean() / downside_std * np.sqrt(TRADING_DAYS_PER_YEAR) if downside_std > 0 else np.nan

        def drawdown_duration():
            in_drawdown = False
            start = None
            durations = []
            for idx, row in daily.iterrows():
                if row.equity < row.peak_equity:
                    if not in_drawdown:
                        in_drawdown, start = True, idx
                elif in_drawdown:
                    durations.append(idx - start)
                    in_drawdown = False
            if in_drawdown:
                durations.append(len(daily) - 1 - start)
            return max(durations, default=0), float(np.mean(durations)) if durations else 0.0

        max_duration, avg_duration = drawdown_duration()
        total_net = float(net.sum())
        metrics = [
            ("Initial Capital (INR)", self.initial_capital),
            ("Trading Days", len(daily)),
            ("Executed Trades", n),
            ("Win Rate (%)", float((net > 0).mean() * 100) if n else np.nan),
            ("Gross P&L (INR)", float(gross.sum())),
            ("Trading Costs (INR)", float(costs.sum())),
            ("Net P&L (INR)", total_net),
            ("Net Return (%)", total_net / self.initial_capital * 100),
            ("Average Trade Net P&L (INR)", float(net.mean()) if n else np.nan),
            ("Gross Profit Factor", float(gross_wins / gross_losses) if gross_losses else np.nan),
            ("Net Profit Factor", float(net_wins / net_losses) if net_losses else np.nan),
            ("Sharpe Ratio (Daily, annualized)", float(sharpe)),
            ("Sortino Ratio (Daily, annualized)", float(sortino)),
            ("Annualized Volatility (%)", float(daily_std * np.sqrt(252) * 100) if np.isfinite(daily_std) else np.nan),
            ("Downside Deviation (%)", float(downside_std * np.sqrt(252) * 100) if np.isfinite(downside_std) else np.nan),
            ("Maximum Drawdown (INR)", float(daily.drawdown.max()) if len(daily) else 0.0),
            ("Maximum Drawdown (%)", float(daily.drawdown_pct.max() * 100) if len(daily) else 0.0),
            ("Maximum Drawdown Duration (Trading Days)", max_duration),
            ("Average Drawdown Duration (Trading Days)", avg_duration),
        ]
        result = pd.DataFrame([{"Category": "Portfolio", "Metric": k, "Value": v} for k, v in metrics])
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(self.output_file, index=False)
        daily.to_csv(self.equity_file, index=False)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trades-file", type=Path, default=Path("data/processed/backtest_trades.parquet"))
    parser.add_argument("--equity-file", type=Path, default=Path("data/processed/equity_curve.csv"))
    parser.add_argument("--output-file", type=Path, default=Path("data/processed/risk_metrics.csv"))
    parser.add_argument("--calendar-file", type=Path, default=Path("data/processed/master_options.parquet"))
    parser.add_argument("--capital", type=float, default=250_000.0)
    parser.add_argument("--risk-free-rate", type=float, default=0.06)
    args = parser.parse_args()
    analyzer = StrategyRiskAnalyzer(args.trades_file, args.equity_file, args.output_file,
                                    args.capital, args.risk_free_rate, args.calendar_file)
    print(analyzer.run_analysis().to_string(index=False))


if __name__ == "__main__":
    main()
