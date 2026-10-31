from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from strike_selector import filter_contracts_by_strikes, get_atm_strike, select_strikes


class StrikeSelectorTests(unittest.TestCase):
    STRIKES = [24000. + 50 * i for i in range(30)]

    def test_atm_is_the_strike_closest_to_spot_with_ties_going_to_the_lower_strike(self):
        self.assertEqual(get_atm_strike(24520., self.STRIKES), 24500.)
        self.assertEqual(get_atm_strike(24525., self.STRIKES), 24500.)      # exact tie: min() keeps the first
        with self.assertRaises(ValueError):
            get_atm_strike(24500., [])

    def test_window_is_atm_plus_or_minus_n_and_sorted(self):
        chosen = select_strikes(24500., self.STRIKES, strikes_each_side=2)
        self.assertEqual(chosen, [24400., 24450., 24500., 24550., 24600.])

    def test_window_truncates_at_the_edges_of_the_chain(self):
        low = select_strikes(24000., self.STRIKES, strikes_each_side=3)
        high = select_strikes(self.STRIKES[-1], self.STRIKES, strikes_each_side=3)
        self.assertEqual(low[0], 24000.)
        self.assertEqual(len(low), 4)
        self.assertEqual(high[-1], self.STRIKES[-1])
        self.assertEqual(len(high), 4)

    def test_off_grid_atm_is_snapped_and_duplicates_are_ignored(self):
        chosen = select_strikes(24512., self.STRIKES + self.STRIKES[:3], strikes_each_side=1)
        self.assertEqual(chosen, [24450., 24500., 24550.])

    def test_filter_keeps_only_calls_and_puts_at_selected_strikes(self):
        contracts = [{"strike_price": 24500, "instrument_type": "CE"}, {"strike_price": 24500, "instrument_type": "PE"},
                     {"strike_price": 24500, "instrument_type": "FUT"}, {"strike_price": 25000, "instrument_type": "CE"}]
        kept = filter_contracts_by_strikes(contracts, [24500.])
        self.assertEqual([c["instrument_type"] for c in kept], ["CE", "PE"])


if __name__ == "__main__":
    unittest.main()
