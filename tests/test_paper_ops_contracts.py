from pathlib import Path
import sys
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from paper_ops.catalog import InstrumentCatalog
from paper_ops.contracts import load_master, needs_reroll, select_contracts, trading_days_left

MASTER = ROOT / "data/raw/upstox/instruments/NSE_20261031.json.gz"
IST = "Asia/Kolkata"


def ms(day):
    return int(pd.Timestamp(f"{day} 15:30", tz=IST).timestamp() * 1000)


def records():
    base = {"segment": "NSE_FO", "underlying_key": "NSE_INDEX|Nifty 50", "underlying_symbol": "NIFTY", "lot_size": 65, "tick_size": 5.}
    out = []
    n = 0
    for day in ("2026-11-03", "2026-11-09", "2026-11-17"):
        for strike in range(22000, 24100, 50):
            for kind in ("CE", "PE"):
                n += 1
                out.append({**base, "instrument_key": f"NSE_FO|{n}", "instrument_type": kind, "strike_price": float(strike), "expiry": ms(day)})
    for day in ("2026-11-24", "2026-12-29"):
        n += 1
        out.append({**base, "instrument_key": f"NSE_FO|{n}", "instrument_type": "FUT", "expiry": ms(day), "tick_size": 10.})
    return out


class SelectionTests(unittest.TestCase):
    def test_monday_selection_skips_the_expiry_with_too_few_days_and_reads_real_dates(self):
        chosen = select_contracts(records(), pd.Timestamp("2026-11-02"), 23053.)
        expiries = {v["option"]["expiry_at"][:10] for v in chosen.values()}
        self.assertEqual(expiries, {"2026-11-09"})                  # 3 Nov has only 2 trading days left
        self.assertEqual(set(chosen), {"ATM_CE", "ATM_PE", "OTM_CE", "OTM_PE"})
        self.assertEqual(chosen["ATM_CE"]["option"]["strike"], 23050.)
        self.assertGreater(chosen["OTM_CE"]["option"]["strike"], 23053.)
        self.assertLess(chosen["OTM_PE"]["option"]["strike"], 23053.)
        self.assertEqual(chosen["ATM_CE"]["future"]["expiry_at"][:10], "2026-11-24")
        self.assertEqual(chosen["ATM_CE"]["option"]["tick_size"], .05)

    def test_selection_is_deterministic_and_never_silently_duplicates(self):
        a = select_contracts(records(), pd.Timestamp("2026-11-02"), 23053.)
        b = select_contracts(list(reversed(records())), pd.Timestamp("2026-11-02"), 23053.)
        self.assertEqual(a, b)
        with self.assertRaises(ValueError):
            select_contracts(records(), pd.Timestamp("2026-11-02"), 23053., otm_fraction=0.)

    def test_no_eligible_expiry_or_future_is_an_error(self):
        with self.assertRaises(ValueError):
            select_contracts(records(), pd.Timestamp("2026-11-30"), 23053.)
        with self.assertRaises(ValueError):
            select_contracts([r for r in records() if r["instrument_type"] != "FUT"], pd.Timestamp("2026-11-02"), 23053.)

    def test_reroll_on_time_and_on_moneyness_drift(self):
        chosen = select_contracts(records(), pd.Timestamp("2026-11-02"), 23053.)
        atm, otm = chosen["ATM_CE"], chosen["OTM_PE"]
        self.assertEqual(needs_reroll(atm, pd.Timestamp("2026-11-03"), 23100.), [])
        self.assertIn("fewer_than_reroll_days_remaining", needs_reroll(atm, pd.Timestamp("2026-11-09"), 23053.))
        self.assertIn("moneyness_drift", needs_reroll(atm, pd.Timestamp("2026-11-03"), 22400.))
        self.assertEqual(needs_reroll(otm, pd.Timestamp("2026-11-03"), 23080.), [])

    def test_trading_day_count(self):
        self.assertEqual(trading_days_left(pd.Timestamp("2026-11-02"), pd.Timestamp("2026-11-03")), 2)
        self.assertEqual(trading_days_left(pd.Timestamp("2026-11-02"), pd.Timestamp("2026-11-09")), 6)
        self.assertEqual(trading_days_left(pd.Timestamp("2026-11-10"), pd.Timestamp("2026-11-09")), 0)


@unittest.skipUnless(MASTER.exists(), "instrument master snapshot not present")
class RealMasterTests(unittest.TestCase):
    def test_selection_on_the_real_snapshot_validates_against_the_catalog(self):
        master = load_master(MASTER)
        chosen = select_contracts(master, pd.Timestamp("2026-11-02"), 23053.)
        catalog = InstrumentCatalog.from_file(MASTER, pd.Timestamp("2026-11-02 08:00", tz=IST))
        for name, contracts in chosen.items():
            report = catalog.validate({"option": contracts["option"], "future": contracts["future"]}, now=pd.Timestamp("2026-11-02 09:00", tz=IST))
            self.assertEqual(set(report["contracts"]), {"option", "future"}, name)


if __name__ == "__main__":
    unittest.main()
