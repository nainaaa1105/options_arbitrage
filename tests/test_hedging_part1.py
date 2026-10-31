import sys
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from datetime import date

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from deep_hedging.accounting import FuturesPosition
from deep_hedging.baselines import target_lots
from deep_hedging.config import HedgeConfig
from deep_hedging.costs import execution_price, transaction_fees
from deep_hedging.data import build_episodes, validate_sources
from deep_hedging.engine import run_episode, compare_baselines, margin_state
from deep_hedging.metrics import empirical_loss_risk, make_reports
from deep_hedging.collect import collect

BASE = pd.Timestamp("2026-01-06 09:20", tz="Asia/Kolkata")


def zero_config(**changes):
    c = HedgeConfig(initial_capital=100_000, option_half_spread=0, option_slippage=0,
                    future_half_spread=0, future_slippage=0, brokerage_per_order=0,
                    option_exchange_rate=0, future_exchange_rate=0, sebi_rate=0,
                    option_ipft_rate=0, future_ipft_rate=0, option_stamp_rate=0,
                    future_stamp_rate=0, option_stt_override=0, future_stt_override=0, gst_rate=0)
    return replace(c, **changes)


def fixture(delta=.5, option_prices=None):
    episode = {"episode_id": "test", "instrument_key": "OPT", "entry_time": BASE,
               "exit_time": BASE+pd.Timedelta(minutes=30), "option_lot_size": 25,
               "option_quantity": 100, "option_lots": 4}
    ts = pd.date_range(BASE, periods=7, freq="5min")
    path = pd.DataFrame({"episode_id": "test", "timestamp": ts, "observed_at": ts,
                         "option_mark": option_prices if option_prices is not None else np.arange(7)*5+100.,
                         "future_mark": np.arange(7)*10+1000., "delta": delta,
                         "future_spot_delta": 1., "future_lot_size": 25,
                         "future_tick_size": .05, "future_instrument_key": "NSE_FO|future"})
    return episode, path


def source_fixture():
    ts = pd.date_range(BASE-pd.Timedelta(minutes=5), periods=10, freq="5min")
    o = pd.DataFrame({"timestamp": ts, "instrument_key": "OPT", "expiry": pd.Timestamp("2026-01-13"),
                      "strike": 1000., "option_type": "CE", "lot_size": 25,
                      "open": np.arange(10)+100., "close": np.arange(10)+101., "volume": 10,
                      "open_interest": 100, "nifty_spot": np.arange(10)*10+1000., "delta": .5,
                      "feature_available_at": ts+pd.Timedelta(minutes=5),
                      "spot_available_at": ts+pd.Timedelta(minutes=5)})
    f = pd.DataFrame({"timestamp": ts, "instrument_key": "NSE_FO|future", "open": np.arange(10)*10+1000.,
                      "close": np.arange(10)*10+1002., "lot_size": 25, "tick_size": .1,
                      "expiry": pd.Timestamp("2026-01-27"), "asset_type": "FUT", "bar_minutes": 5})
    return o, f


class HedgingAccountingTests(unittest.TestCase):
    def test_futures_open_partial_close_and_reversal_do_not_purchase_notional(self):
        p = FuturesPosition()
        self.assertEqual(p.trade(100, 1000), 0)
        self.assertEqual(p.trade(-40, 1010), 400)
        self.assertEqual(p.quantity, 60)
        self.assertEqual(p.trade(-100, 990), -600)
        self.assertEqual(p.quantity, -40)
        self.assertEqual(p.average_fill, 990)
        self.assertEqual(p.unrealized(980), 400)
        self.assertEqual(p.trade(40, 980), 400)
        self.assertEqual(p.quantity, 0)

    def test_perfect_linear_call_hedge_cancels_option_pnl(self):
        episode, path = fixture()
        result, orders, marks = run_episode(episode, path, "dynamic_delta", 100000, zero_config())
        self.assertAlmostEqual(result["option_gross_pnl"], 3000)
        self.assertAlmostEqual(result["future_gross_pnl"], -3000)
        self.assertAlmostEqual(result["net_pnl"], 0)
        self.assertEqual(marks[0]["cash"], 90000)
        self.assertEqual(marks[0]["futures_margin"], 12500)
        self.assertEqual(marks[0]["equity"], 100000)
        self.assertEqual(marks[-1]["future_quantity"], 0)
        self.assertEqual(marks[-1]["option_quantity"], 0)

    def test_put_delta_hedge_buys_futures(self):
        episode, path = fixture(delta=-.5, option_prices=100.-np.arange(7)*5)
        result, orders, _ = run_episode(episode, path, "static_delta", 100000, zero_config())
        self.assertEqual([r for r in orders if r["asset"] == "future"][0]["quantity_change"], 50)
        self.assertAlmostEqual(result["net_pnl"], 0)

    def test_short_option_sign_and_explicit_margin(self):
        with self.assertRaises(ValueError):
            zero_config(option_side=-1)
        episode, path = fixture()
        episode["option_quantity"] = -100
        result, _, marks = run_episode(episode, path, "static_delta", 100000,
                                      zero_config(option_side=-1, short_option_margin_per_lot=5000))
        self.assertEqual(marks[0]["short_option_margin"], 20000)
        self.assertAlmostEqual(result["net_pnl"], 0)

    def test_constant_delta_static_dynamic_and_band_match(self):
        episode, path = fixture()
        values = [run_episode(episode, path, s, 100000, zero_config())[0]["net_pnl"]
                  for s in ("static_delta", "dynamic_delta", "band_delta")]
        np.testing.assert_allclose(values, 0)

    def test_unhedged_has_no_futures_orders(self):
        episode, path = fixture()
        result, orders, _ = run_episode(episode, path, "unhedged", 100000, zero_config())
        self.assertEqual(result["net_pnl"], 3000)
        self.assertFalse(any(r["asset"] == "future" for r in orders))

    def test_every_rebalance_and_terminal_close_pays_costs(self):
        episode, path = fixture(delta=[.25, .75, .25, .75, .25, .75, .25])
        c = zero_config(brokerage_per_order=20, future_brokerage_cap_rate=1)
        dynamic, orders, _ = run_episode(episode, path, "dynamic_delta", 100000, c)
        band, band_orders, _ = run_episode(episode, path, "band_delta", 100000, replace(c, no_trade_band=.6))
        self.assertGreater(len(orders), len(band_orders))
        self.assertAlmostEqual(dynamic["fees"], 20*len(orders))
        self.assertEqual(orders[-2]["asset"], "future")
        self.assertEqual(orders[-1]["asset"], "option")
        self.assertLess(band["fees"], dynamic["fees"])

    def test_tick_rounding_is_adverse_and_friction_is_not_double_counted(self):
        self.assertAlmostEqual(execution_price(100.01, 10, .05, .1, .1), 100.25)
        self.assertAlmostEqual(execution_price(100.01, -10, .05, .1, .1), 99.80)
        episode, path = fixture()
        result, orders, _ = run_episode(episode, path, "static_delta", 100000,
                                      zero_config(option_half_spread=.1, future_half_spread=.1))
        self.assertAlmostEqual(result["net_pnl"], -result["execution_impact"])
        self.assertAlmostEqual(result["execution_impact"], sum(r["execution_impact"] for r in orders))

    def test_stt_uses_correct_asset_turnover_side_and_effective_date(self):
        c = HedgeConfig()
        before = transaction_fees("future", -65, 25000, "2026-03-30", c)
        after = transaction_fees("future", -65, 25000, "2026-04-01", c)
        self.assertAlmostEqual(before["stt"], 325)
        self.assertAlmostEqual(after["stt"], 812.5)
        self.assertEqual(transaction_fees("future", 65, 25000, "2026-04-01", c)["stt"], 0)
        self.assertAlmostEqual(transaction_fees("option", -65, 100, "2026-04-01", c)["stt"], 9.75)

    def test_whole_lot_rounding_cap_and_half_lot_ties(self):
        state = {"delta": .5, "future_spot_delta": 1., "option_quantity": 25, "future_lot_size": 25}
        self.assertEqual(target_lots("dynamic_delta", state, 0, True, zero_config())[0], 0)
        state["option_quantity"] = 1000
        lots, reason = target_lots("dynamic_delta", state, 0, True, zero_config())
        self.assertEqual(lots, -4)
        self.assertEqual(reason, "exposure_capped")

    def test_future_observation_rejected(self):
        episode, path = fixture()
        path.loc[0, "observed_at"] += pd.Timedelta(minutes=5)
        with self.assertRaises(ValueError):
            run_episode(episode, path, "dynamic_delta", 100000, zero_config())

    def test_missing_delta_holds_inventory_instead_of_dropping_episode(self):
        episode, path = fixture()
        path.loc[2, "delta"] = np.nan
        result, _, _ = run_episode(episode, path, "dynamic_delta", 100000, zero_config())
        self.assertEqual(result["invalid_delta_events"], 1)
        self.assertAlmostEqual(result["net_pnl"], 0)

    def test_margin_breach_forces_both_legs_closed_and_retains_loss(self):
        episode, path = fixture(option_prices=np.full(7, 100.))
        path.loc[1:, "future_mark"] = 11000.
        result, _, marks = run_episode(episode, path, "static_delta", 100000, zero_config())
        self.assertEqual(result["forced_exits"], 1)
        self.assertEqual(len(marks), 2)
        self.assertEqual(result["net_pnl"], -500000)
        self.assertEqual(marks[-1]["future_quantity"], 0)

    def test_zero_bid_terminal_long_is_explicit_writeoff_not_removed(self):
        episode, path = fixture()
        path.loc[6, "option_mark"] = .05
        result, orders, _ = run_episode(episode, path, "unhedged", 100000,
                                      zero_config(option_half_spread=.15, option_slippage=.15))
        self.assertEqual(result["zero_recovery_writeoffs"], 1)
        self.assertEqual(orders[-1]["event"], "zero_recovery_writeoff")
        self.assertFalse(orders[-1]["executed_order"])
        self.assertAlmostEqual(result["net_pnl"], -10030)

    def test_positive_unrealized_gain_not_available_cash_and_margin_not_expense(self):
        account = margin_state(90000, 100, 100, FuturesPosition(50, 1000), 1010, 4, zero_config())
        self.assertEqual(account["future_unrealized"], 500)
        self.assertEqual(account["cash"], 90000)
        self.assertEqual(account["equity"], 100500)
        self.assertEqual(account["free_cash"], 90000-50*1010*.25)

    def test_common_capital_rejections_apply_to_every_baseline(self):
        episode, path = fixture()
        results, _, _, rejected = compare_baselines(pd.DataFrame([episode]), path, zero_config(initial_capital=100))
        self.assertTrue(results.empty)
        self.assertEqual(rejected.reason.iloc[0], "common_initial_capital_limit")

    def test_marked_daily_equity_includes_zero_days_and_initial_peak(self):
        episode, path = fixture()
        c = zero_config(brokerage_per_order=20, future_brokerage_cap_rate=1)
        results, orders, marks, _ = compare_baselines(pd.DataFrame([episode]), path, c)
        dates = [BASE.date(), (BASE+pd.Timedelta(days=1)).date()]
        report, daily = make_reports(results, orders, marks, dates, c)
        self.assertEqual(len(daily), 8)
        self.assertTrue(daily.loc[daily.trade_date.eq(dates[1]), "net_pnl"].eq(0).all())
        self.assertTrue(report.max_intraday_drawdown_rupees.gt(0).all())
        for name, group in daily.groupby("strategy"):
            expected = results.loc[results.strategy.eq(name), "net_pnl"].sum()
            self.assertAlmostEqual(group.net_pnl.sum(), expected)

    def test_empirical_cvar_handles_small_samples_and_tail_mass(self):
        var, cvar = empirical_loss_risk([0, -100], .95)
        self.assertEqual(var, 100)
        self.assertEqual(cvar, 100)
        self.assertAlmostEqual(empirical_loss_risk([0, -10, -100], .5)[1], 70)

    def test_funding_accrues_only_while_position_is_open(self):
        episode, path = fixture()
        result, _, _ = run_episode(episode, path, "unhedged", 100000, zero_config(funding_rate=365.))
        self.assertAlmostEqual(result["funding"], 10000*30/1440)
        self.assertAlmostEqual(result["net_pnl"], 3000-result["funding"])

    def test_fractional_option_units_are_rejected(self):
        episode, path = fixture()
        episode["option_quantity"] = 100.5
        with self.assertRaises(ValueError):
            run_episode(episode, path, "static_delta", 100000, zero_config())

    def test_changing_future_observations_cannot_change_earlier_hedges(self):
        episode, path = fixture()
        _, first, _ = run_episode(episode, path, "dynamic_delta", 100000, zero_config())
        changed = path.copy()
        changed.loc[4:, "delta"] = -.9
        changed.loc[4:, "future_mark"] += 100
        _, second, _ = run_episode(episode, changed, "dynamic_delta", 100000, zero_config())
        a = [r for r in first if r["timestamp"] < path.timestamp.iloc[4]]
        b = [r for r in second if r["timestamp"] < path.timestamp.iloc[4]]
        self.assertEqual(a, b)


class HedgingDataTests(unittest.TestCase):
    def test_next_open_uses_previous_completed_delta_and_observed_basis(self):
        o, f = source_fixture()
        o.loc[1, "delta"] = .9
        episodes, paths, skipped, _ = build_episodes(o, f, zero_config(), ("09:15",))
        self.assertEqual(len(episodes), 1)
        self.assertEqual(episodes.entry_time.iloc[0], BASE)
        self.assertEqual(paths.delta.iloc[0], .5)
        self.assertEqual(paths.delta.iloc[1], .9)
        self.assertEqual(paths.option_mark.iloc[0], 101)
        self.assertAlmostEqual(paths.future_spot_delta.iloc[0], 1002/1000)

    def test_missing_exact_hedge_bar_rejects_common_episode(self):
        o, f = source_fixture()
        f = f.drop(index=4)
        episodes, _, skipped, _ = build_episodes(o, f, zero_config(), ("09:15",))
        self.assertTrue(episodes.empty)
        self.assertEqual(skipped.reason.iloc[0], "missing_exact_option_or_future_bar")

    def test_unsynchronized_source_and_index_as_hedge_are_rejected(self):
        o, f = source_fixture()
        wrong = o.copy()
        wrong["spot_available_at"] -= pd.Timedelta(minutes=4)
        with self.assertRaises(ValueError):
            validate_sources(wrong, f)
        f["instrument_key"] = "NSE_INDEX|Nifty 50"
        with self.assertRaises(ValueError):
            validate_sources(o, f)

    def test_missing_later_greek_does_not_select_away_future_difficulty(self):
        o, f = source_fixture()
        o.loc[2, "delta"] = np.nan
        episodes, paths, _, _ = build_episodes(o, f, zero_config(), ("09:15",))
        self.assertEqual(len(episodes), 1)
        self.assertTrue(paths.delta.isna().any())

    def test_hedge_only_date_is_retained_in_equity_calendar(self):
        o, f = source_fixture()
        extra = f.iloc[:1].copy()
        extra["timestamp"] += pd.Timedelta(days=1)
        f = pd.concat([f, extra], ignore_index=True)
        episodes, _, skipped, dates = build_episodes(o, f, zero_config(), ("09:15",))
        self.assertEqual(len(episodes), 1)
        self.assertEqual(len(dates), 2)
        self.assertTrue(skipped.reason.eq("no_decision_bar").any())


class HedgingCollectionTests(unittest.TestCase):
    def metadata(self):
        return {"data": [{"instrument_type": "FUT", "underlying_key": "NSE_INDEX|Nifty 50",
                          "instrument_key": "NSE_FO|test|28-04-2026", "lot_size": 65,
                          "tick_size": 10., "trading_symbol": "NIFTY FUT APR26"}]}

    def test_closed_calendar_window_is_distinguished_and_tick_units_converted(self):
        with TemporaryDirectory() as tmp:
            output, calendar = Path(tmp)/"out", Path(tmp)/"spot.parquet"
            pd.DataFrame({"timestamp": [pd.Timestamp("2026-04-01 09:15", tz="Asia/Kolkata")]}).to_parquet(calendar)
            candle = [["2026-04-01T09:15:00+05:30", 100, 101, 99, 100, 10, 1]]
            with patch("deep_hedging.collect.get_expiries", return_value=["2026-03-30", "2026-04-28"]), \
                 patch("deep_hedging.collect.make_request", return_value=self.metadata()), \
                 patch("deep_hedging.collect.get_expired_candles", side_effect=[[], candle]):
                result = collect(date(2026,3,31), date(2026,4,1), output, calendar)
            self.assertEqual(len(result), 1)
            self.assertEqual(result.tick_size.iloc[0], .1)
            self.assertEqual(result.lot_size.iloc[0], 65)
            with patch("deep_hedging.collect.get_expiries") as api:
                with self.assertRaises(FileExistsError):
                    collect(date(2026,3,31), date(2026,4,1), output, calendar, resume=True)
                api.assert_not_called()

    def test_empty_response_on_known_trading_date_is_a_failure(self):
        with TemporaryDirectory() as tmp:
            output, calendar = Path(tmp)/"out", Path(tmp)/"spot.parquet"
            pd.DataFrame({"timestamp": [pd.Timestamp("2026-03-31 09:15", tz="Asia/Kolkata")]}).to_parquet(calendar)
            with patch("deep_hedging.collect.get_expiries", return_value=["2026-03-30", "2026-04-28"]), \
                 patch("deep_hedging.collect.make_request", return_value=self.metadata()), \
                 patch("deep_hedging.collect.get_expired_candles", return_value=[]):
                with self.assertRaises(ValueError):
                    collect(date(2026,3,31), date(2026,4,1), output, calendar)
            self.assertFalse((output/"front_month_futures.parquet").exists())


if __name__ == "__main__":
    unittest.main()
