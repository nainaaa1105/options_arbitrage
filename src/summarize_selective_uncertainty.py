"""Report cluster uncertainty for every predeclared selective diagnostic."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def run(folder, repeats=10000):
    result = []
    family_size = 3 * 5 * 4
    adjusted_tail = .05 / (2 * family_size)
    for target in ("endpoint", "first_passage", "sustained_two_bars"):
        d = pd.read_parquet(folder / f"{target}_fresh_predictions.parquet")
        d["session"] = pd.to_datetime(d.timestamp).dt.date
        sessions = sorted(d.session.unique())
        weights = np.random.default_rng(42).multinomial(len(sessions), np.full(len(sessions), 1/len(sessions)), size=repeats)
        for margin in (0., .1, .2, .3, .4):
            high = d.probability.ge(.5 + margin)
            accepted = high | d.probability.le(.5 - margin)
            low = accepted & ~high
            counts = pd.DataFrame({"total": np.ones(len(d)), "accepted": accepted.astype(int),
                                   "correct": ((high == d.actual) & accepted).astype(int),
                                   "positive": high.astype(int), "tp": (high & d.actual.eq(1)).astype(int),
                                   "negative": low.astype(int), "tn": (low & d.actual.eq(0)).astype(int),
                                   "session": d.session}).groupby("session").sum().reindex(sessions)
            sampled = weights @ counts.to_numpy(float)
            fields = {"coverage": (1, 0), "accuracy": (2, 1), "positive_precision": (4, 3), "negative_precision": (6, 5)}
            row = {"target": target, "margin": margin, "sessions": len(sessions)}
            for name, (numerator, denominator) in fields.items():
                valid = sampled[:, denominator] > 0
                values = sampled[valid, numerator] / sampled[valid, denominator]
                row[name] = None if not len(values) else {
                    "lower_95": float(np.quantile(values, .025)), "upper_95": float(np.quantile(values, .975)),
                    "adjusted_lower": float(np.quantile(values, adjusted_tail)),
                    "adjusted_upper": float(np.quantile(values, 1-adjusted_tail))}
            result.append(row)
    output = folder / "selective_session_uncertainty.json"
    if output.exists():
        raise FileExistsError("Preserving existing uncertainty report")
    output.write_text(json.dumps({"repeats": repeats, "clusters": "trading sessions", "family_size": family_size,
        "method": "percentile cluster bootstrap; adjusted tails use Bonferroni across all 60 diagnostic fields; exploratory, no finite-sample coverage guarantee",
        "fresh_period_policy_selected": False, "rows": result}, indent=2), encoding="utf-8")
    print(json.dumps([r for r in result if r["margin"] in (.3, .4)], indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, required=True)
    run(parser.parse_args().folder)
