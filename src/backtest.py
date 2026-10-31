"""Conservative, next-bar options backtest for 5-minute OHLC data.

The source has no bid/ask quotes or historical SPAN files.  Accordingly this
engine applies explicit half-spread and slippage assumptions and excludes
short-option signals unless a per-lot short margin proxy is supplied.
"""

import argparse
import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


logger = logging.getLogger("backtest")
MARKET_TZ = "Asia/Kolkata"
DEFAULT_HORIZON_MINUTES = 30
DEFAULT_INITIAL_CAPITAL = 250_000.0
DEFAULT_MAX_CONCURRENT_POSITIONS = 5
DEFAULT_CAPITAL_FRACTION = 0.95
DEFAULT_MAX_LOTS = 1
DEFAULT_HALF_SPREAD_POINTS = 0.15
DEFAULT_SLIPPAGE_POINTS = 0.15
DEFAULT_BROKERAGE_PER_ORDER = 20.0
DEFAULT_EXCHANGE_RATE = 0.0003553  # NSE equity options premium turnover, 2026 schedule
DEFAULT_SEBI_RATE = 0.000001       # 0.0001% on premium turnover
DEFAULT_STAMP_DUTY_RATE = 0.00003  # 0.003%, buyer side on premium turnover
DEFAULT_GST_RATE = 0.18
STT_RATE_BEFORE_2026_04_01 = 0.001
STT_RATE_FROM_2026_04_01 = 0.0015


def _market_timestamps(series):
    timestamps = pd.to_datetime(series, errors="coerce")
    if timestamps.dt.tz is None:
        return timestamps.dt.tz_localize(MARKET_TZ)
    return timestamps.dt.tz_convert(MARKET_TZ)


class OptionsArbitrageBacktester:
    """Run next-bar fills with explicit costs, capital, and overlap controls."""

    def __init__(
        self,
        signals_file: Path,
        master_file: Path,
        output_dir: Path,
        horizon_minutes: int = DEFAULT_HORIZON_MINUTES,
        slippage_points: float = DEFAULT_SLIPPAGE_POINTS,
        brokerage_per_order: float = DEFAULT_BROKERAGE_PER_ORDER,
        stt_rate: float | None = None,
        exchange_txn_rate: float = DEFAULT_EXCHANGE_RATE,
        *,
        initial_capital: float = DEFAULT_INITIAL_CAPITAL,
        max_concurrent_positions: int = DEFAULT_MAX_CONCURRENT_POSITIONS,
        capital_fraction: float = DEFAULT_CAPITAL_FRACTION,
        max_lots: int = DEFAULT_MAX_LOTS,
        half_spread_points: float = DEFAULT_HALF_SPREAD_POINTS,
        sebi_rate: float = DEFAULT_SEBI_RATE,
        stamp_duty_rate: float = DEFAULT_STAMP_DUTY_RATE,
        gst_rate: float = DEFAULT_GST_RATE,
        short_margin_per_lot: float = 0.0,
    ):
        self.signals_file = Path(signals_file)
        self.master_file = Path(master_file)
        self.output_dir = Path(output_dir)
        self.horizon_minutes = int(horizon_minutes)
        self.slippage_points = float(slippage_points)
        self.half_spread_points = float(half_spread_points)
        self.brokerage_per_order = float(brokerage_per_order)
        self.stt_rate_override = stt_rate
        self.exchange_txn_rate = float(exchange_txn_rate)
        self.sebi_rate = float(sebi_rate)
        self.stamp_duty_rate = float(stamp_duty_rate)
        self.gst_rate = float(gst_rate)
        self.initial_capital = float(initial_capital)
        self.max_concurrent_positions = int(max_concurrent_positions)
        self.capital_fraction = float(capital_fraction)
        self.max_lots = int(max_lots)
        self.short_margin_per_lot = float(short_margin_per_lot)
        if self.horizon_minutes <= 0 or self.initial_capital <= 0:
            raise ValueError("horizon_minutes and initial_capital must be positive")
        if not 0 < self.capital_fraction <= 1 or self.max_lots < 1:
            raise ValueError("capital_fraction must be (0,1] and max_lots >= 1")
        self.skipped = {}

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if not self.signals_file.exists() or not self.master_file.exists():
            raise FileNotFoundError("Signal or master price file is missing")
        return pd.read_parquet(self.signals_file), pd.read_parquet(self.master_file)

    def _stt_rate(self, timestamp):
        if self.stt_rate_override is not None:
            return float(self.stt_rate_override)
        cutoff = pd.Timestamp("2026-04-01", tz=MARKET_TZ)
        return STT_RATE_FROM_2026_04_01 if timestamp >= cutoff else STT_RATE_BEFORE_2026_04_01

    def simulate(self, signals_df: pd.DataFrame, master_df: pd.DataFrame):
        required_signals = {"timestamp", "instrument_key", "trade_signal"}
        required_market = {"timestamp", "instrument_key", "open", "lot_size"}
        if required_signals.difference(signals_df.columns):
            raise ValueError(f"Signals missing columns: {sorted(required_signals.difference(signals_df.columns))}")
        if required_market.difference(master_df.columns):
            raise ValueError(f"Master data missing columns: {sorted(required_market.difference(master_df.columns))}")

        signals = signals_df.copy()
        market = master_df.copy()
        signals["timestamp"] = _market_timestamps(signals["timestamp"])
        market["timestamp"] = _market_timestamps(market["timestamp"])
        market["open"] = pd.to_numeric(market["open"], errors="coerce")
        market["lot_size"] = pd.to_numeric(market["lot_size"], errors="coerce")
        market = market.dropna(subset=["timestamp", "instrument_key", "open", "lot_size"])
        market = market.loc[(market.open > 0) & (market.lot_size > 0)]
        market = market.sort_values("timestamp")
        bars = {
            key: part.sort_values("timestamp").reset_index(drop=True)
            for key, part in market.groupby("instrument_key", sort=False)
        }
        actionable = signals.loc[signals["trade_signal"].isin(["BUY", "SELL"])].sort_values("timestamp")
        active = []
        records = []
        self.skipped = {}

        def skip(reason):
            self.skipped[reason] = self.skipped.get(reason, 0) + 1

        for _, signal in actionable.iterrows():
            key = signal["instrument_key"]
            contract_bars = bars.get(key)
            if contract_bars is None:
                skip("contract_not_in_market_data")
                continue
            times = contract_bars["timestamp"].dt.as_unit("ns").array.asi8
            entry_ns = signal["timestamp"].value
            entry_idx = int(np.searchsorted(times, entry_ns, side="right"))
            if entry_idx >= len(contract_bars):
                skip("no_next_bar_entry")
                continue
            entry_bar = contract_bars.iloc[entry_idx]
            entry_time = entry_bar["timestamp"]
            exit_target = entry_time + pd.Timedelta(minutes=self.horizon_minutes)
            exit_idx = int(np.searchsorted(times, exit_target.value, side="left"))
            if exit_idx >= len(contract_bars):
                skip("missing_exit_bar")
                continue
            exit_bar = contract_bars.iloc[exit_idx]
            exit_time = exit_bar["timestamp"]
            raw_entry = float(entry_bar["open"])
            raw_exit = float(exit_bar["open"])
            lot_size = int(entry_bar["lot_size"])
            if lot_size <= 0 or not np.isfinite(raw_entry) or not np.isfinite(raw_exit):
                skip("invalid_fill_or_lot_size")
                continue

            # Release positions whose exits occur before this signal's entry.
            active = [position for position in active if position["exit_time"] > entry_time]
            if any(position["instrument_key"] == key for position in active):
                skip("overlapping_same_contract")
                continue
            if len(active) >= self.max_concurrent_positions:
                skip("max_concurrent_positions")
                continue

            direction = signal["trade_signal"]
            lots = self.max_lots
            if direction == "SELL":
                if self.short_margin_per_lot <= 0:
                    skip("short_margin_data_unavailable")
                    continue
                capital_per_lot = self.short_margin_per_lot
            else:
                capital_per_lot = raw_entry * lot_size
            capital_limit = self.initial_capital * self.capital_fraction
            allocated = sum(position["capital_required"] for position in active)
            affordable = int(max(0, (capital_limit - allocated) // capital_per_lot)) if capital_per_lot > 0 else 0
            lots = min(lots, affordable)
            if lots < 1:
                skip("capital_limit")
                continue
            quantity = lot_size * lots

            half_spread = self.half_spread_points
            slippage = self.slippage_points
            if direction == "BUY":
                fill_entry = raw_entry + half_spread + slippage
                fill_exit = raw_exit - half_spread - slippage
                buyer_turnover = fill_entry * quantity
                seller_turnover = fill_exit * quantity
                gross = (raw_exit - raw_entry) * quantity
            else:
                fill_entry = raw_entry - half_spread - slippage
                fill_exit = raw_exit + half_spread + slippage
                seller_turnover = fill_entry * quantity
                buyer_turnover = fill_exit * quantity
                gross = (raw_entry - raw_exit) * quantity
            if min(fill_entry, fill_exit, buyer_turnover, seller_turnover) <= 0:
                skip("nonpositive_executable_price")
                continue

            turnover = (fill_entry + fill_exit) * quantity
            exchange = turnover * self.exchange_txn_rate
            sebi = turnover * self.sebi_rate
            stt = seller_turnover * self._stt_rate(entry_time)
            stamp = buyer_turnover * self.stamp_duty_rate
            brokerage = 2.0 * self.brokerage_per_order
            gst = (brokerage + exchange + sebi) * self.gst_rate
            statutory_costs = exchange + sebi + stt + stamp + brokerage + gst
            spread_cost = 2.0 * half_spread * quantity
            slippage_cost = 2.0 * slippage * quantity
            net = gross - spread_cost - slippage_cost - statutory_costs
            capital_required = capital_per_lot * lots

            record = {
                "timestamp": entry_time,
                "signal_timestamp": signal["timestamp"],
                "exit_timestamp": exit_time,
                "holding_period_mins": (exit_time - entry_time).total_seconds() / 60.0,
                "instrument_key": key,
                "trading_symbol": signal.get("trading_symbol"),
                "option_type": signal.get("option_type"),
                "strike": signal.get("strike"),
                "expiry": signal.get("expiry"),
                "trade_signal": direction,
                "signal_price": signal.get("market_price"),
                "entry_reference_open": raw_entry,
                "exit_reference_open": raw_exit,
                "entry_price": fill_entry,
                "exit_price": fill_exit,
                "lot_size": lot_size,
                "lots": lots,
                "quantity": quantity,
                "capital_required": capital_required,
                "gross_pnl": gross,
                "spread_cost": spread_cost,
                "slippage": slippage_cost,
                "exchange_charges": exchange,
                "sebi_charges": sebi,
                "stt": stt,
                "stamp_duty": stamp,
                "brokerage": brokerage,
                "gst": gst,
                "transaction_costs": statutory_costs,
                "total_costs": spread_cost + slippage_cost + statutory_costs,
                "net_pnl": net,
                "predicted_probability": signal.get("predicted_probability"),
                "price_deviation": signal.get("price_deviation"),
            }
            records.append(record)
            active.append({"instrument_key": key, "exit_time": exit_time,
                           "capital_required": capital_required})

        trades = pd.DataFrame(records)
        if trades.empty:
            trades = pd.DataFrame(columns=["timestamp", "exit_timestamp", "net_pnl", "gross_pnl"])
        calendar = pd.Index(pd.Series(market.timestamp.dt.date.unique()).sort_values().unique())
        if len(calendar) == 0:
            raise ValueError("No valid market dates available for equity curve")
        daily_pnl = trades.groupby(trades.exit_timestamp.dt.date).net_pnl.sum() if len(trades) else pd.Series(dtype=float)
        daily = pd.DataFrame({"trade_date": calendar})
        daily["net_pnl"] = daily.trade_date.map(daily_pnl).fillna(0.0)
        daily["timestamp"] = pd.to_datetime(daily.trade_date.astype(str)).dt.tz_localize(MARKET_TZ) + pd.Timedelta(hours=15, minutes=30)
        daily["cumulative_pnl"] = daily.net_pnl.cumsum()
        daily["equity"] = self.initial_capital + daily.cumulative_pnl
        daily["daily_return"] = daily.net_pnl / (daily.equity.shift(1).fillna(self.initial_capital))
        daily["peak_equity"] = daily.equity.cummax()
        daily["drawdown"] = daily.peak_equity - daily.equity
        daily["drawdown_pct"] = daily.drawdown / daily.peak_equity.replace(0, np.nan)
        if len(trades):
            trades = trades.sort_values("exit_timestamp").reset_index(drop=True)
            trades["cumulative_pnl"] = trades.net_pnl.cumsum()
            trades["trade_result"] = np.where(trades.net_pnl > 0, "WIN", np.where(trades.net_pnl < 0, "LOSS", "BREAKEVEN"))
        summary = self._compute_summary(trades, daily)
        return trades, daily, summary

    def _compute_summary(self, trades, daily):
        net = trades.net_pnl if len(trades) else pd.Series(dtype=float)
        gross = trades.gross_pnl if len(trades) else pd.Series(dtype=float)
        net_wins = net[net > 0].sum()
        net_losses = abs(net[net < 0].sum())
        gross_wins = gross[gross > 0].sum()
        gross_losses = abs(gross[gross < 0].sum())
        rf_daily = 0.06 / 252.0
        excess = daily.daily_return - rf_daily
        std = daily.daily_return.std(ddof=1)
        downside = np.sqrt(np.mean(np.minimum(daily.daily_return - rf_daily, 0.0) ** 2))
        sharpe = excess.mean() / std * np.sqrt(252.0) if std and np.isfinite(std) else np.nan
        sortino = excess.mean() / downside * np.sqrt(252.0) if downside > 0 else np.nan
        values = [
            ("Initial Capital (INR)", self.initial_capital),
            ("Executed Trades", len(trades)),
            ("Skipped Signals", sum(self.skipped.values())),
            ("Skip Reasons", str(self.skipped)),
            ("Win Rate (%)", float((net > 0).mean() * 100) if len(net) else np.nan),
            ("Gross P&L (INR)", float(gross.sum())),
            ("Spread Cost (INR)", float(trades.spread_cost.sum()) if len(trades) else 0.0),
            ("Slippage (INR)", float(trades.slippage.sum()) if len(trades) else 0.0),
            ("Transaction Costs (INR)", float(trades.transaction_costs.sum()) if len(trades) else 0.0),
            ("Net P&L (INR)", float(net.sum())),
            ("Net Return (%)", float(net.sum() / self.initial_capital * 100)),
            ("Average Trade Net P&L (INR)", float(net.mean()) if len(net) else np.nan),
            ("Gross Profit Factor", float(gross_wins / gross_losses) if gross_losses else np.nan),
            ("Net Profit Factor", float(net_wins / net_losses) if net_losses else np.nan),
            ("Sharpe Ratio (Daily, annualized)", float(sharpe)),
            ("Sortino Ratio (Daily, annualized)", float(sortino)),
            ("Annualized Volatility (%)", float(std * np.sqrt(252.0) * 100) if np.isfinite(std) else np.nan),
            ("Maximum Drawdown (INR)", float(daily.drawdown.max())),
            ("Maximum Drawdown (%)", float(daily.drawdown_pct.max() * 100)),
        ]
        return pd.DataFrame([{"Metric": k, "Value": v} for k, v in values])

    def run_backtest(self):
        signals, market = self.load_data()
        trades, equity, summary = self.simulate(signals, market)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        trades.to_parquet(self.output_dir / "backtest_trades.parquet", index=False)
        summary.to_csv(self.output_dir / "backtest_summary.csv", index=False)
        equity.to_csv(self.output_dir / "equity_curve.csv", index=False)
        return trades, equity, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signals-file", type=Path, default=Path("data/processed/trade_signals.parquet"))
    parser.add_argument("--master-file", type=Path, default=Path("data/processed/master_options.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--horizon-minutes", type=int, default=DEFAULT_HORIZON_MINUTES)
    parser.add_argument("--capital", type=float, default=DEFAULT_INITIAL_CAPITAL)
    parser.add_argument("--max-concurrent-positions", type=int, default=DEFAULT_MAX_CONCURRENT_POSITIONS)
    parser.add_argument("--max-lots", type=int, default=DEFAULT_MAX_LOTS)
    parser.add_argument("--half-spread-points", type=float, default=DEFAULT_HALF_SPREAD_POINTS)
    parser.add_argument("--slippage-points", type=float, default=DEFAULT_SLIPPAGE_POINTS)
    parser.add_argument("--short-margin-per-lot", type=float, default=0.0,
                        help="Required to enable SELL trades; supply a conservative historical margin proxy.")
    args = parser.parse_args()
    engine = OptionsArbitrageBacktester(
        args.signals_file, args.master_file, args.output_dir,
        horizon_minutes=args.horizon_minutes,
        initial_capital=args.capital,
        max_concurrent_positions=args.max_concurrent_positions,
        max_lots=args.max_lots,
        half_spread_points=args.half_spread_points,
        slippage_points=args.slippage_points,
        short_margin_per_lot=args.short_margin_per_lot,
    )
    trades, equity, summary = engine.run_backtest()
    print(summary.to_string(index=False))
    print(f"\nExecuted trades: {len(trades)}; equity dates: {len(equity)}")


if __name__ == "__main__":
    main()
