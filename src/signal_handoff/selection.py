"""Pure, outcome-free entry selection rules. Inputs are decision-bar fields only."""
import hashlib

import numpy as np
import pandas as pd

RULES = ('ml', 'random', 'max_deviation', 'atm_nearest')
REQUIRED = ('instrument_key', 'option_type', 'expiry', 'strike', 'nifty_spot', 'reference_price_deviation', 'probability')


def _check(candidates):
    missing = set(REQUIRED).difference(candidates.columns)
    if missing:
        raise ValueError(f'Candidate fields missing: {sorted(missing)}')
    if (candidates.reference_price_deviation >= 0).any():
        raise ValueError('Long-only protocol: only undervalued candidates (negative deviation) may be passed')


def ml_choice(candidates, tau):
    """Highest out-of-time convergence probability at or above tau; ties by larger |deviation|, then key."""
    _check(candidates)
    eligible = candidates[np.isfinite(candidates.probability) & candidates.probability.ge(tau)]
    if eligible.empty:
        return None
    order = eligible.assign(_dev=eligible.reference_price_deviation.abs()).sort_values(
        ['probability', '_dev', 'instrument_key'], ascending=[False, False, True])
    return order.iloc[0]


def max_deviation_choice(candidates):
    _check(candidates)
    if candidates.empty:
        return None
    order = candidates.assign(_dev=candidates.reference_price_deviation.abs()).sort_values(
        ['_dev', 'instrument_key'], ascending=[False, True])
    return order.iloc[0]


def random_choice(candidates, decision_time, seed=8):
    _check(candidates)
    if candidates.empty:
        return None
    ordered = candidates.sort_values('instrument_key')
    digest = hashlib.sha256(f'{decision_time}|{seed}'.encode()).hexdigest()
    index = int(np.random.default_rng(int(digest[:12], 16)).integers(len(ordered)))
    return ordered.iloc[index]


def atm_choice(admitted, option_type, max_log_distance=.02):
    """Part 1 style no-signal control: nearest expiry then nearest ATM, restricted to the ML trade's option type."""
    if admitted.empty:
        return None
    pool = admitted[admitted.option_type.eq(option_type)].copy()
    pool['atm_distance'] = np.abs(np.log(pool.strike / pool.nifty_spot))
    pool = pool[pool.atm_distance.le(max_log_distance)]
    if pool.empty:
        return None
    return pool.sort_values(['expiry', 'atm_distance', 'strike', 'instrument_key']).iloc[0]


def select_slot(rule, candidates, admitted, decision_time, tau, ml_pick=None, seed=8):
    """Return the chosen row for one decision slot. Controls are evaluated only where the ML rule fires."""
    if rule == 'ml':
        return ml_choice(candidates, tau)
    if ml_pick is None:
        return None
    if rule == 'random':
        return random_choice(candidates, decision_time, seed)
    if rule == 'max_deviation':
        return max_deviation_choice(candidates)
    if rule == 'atm_nearest':
        return atm_choice(admitted, ml_pick.option_type)
    raise ValueError(f'Unknown rule: {rule}')
