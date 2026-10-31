"""Build a leave-one-out implied-volatility reference surface.

Each surface is isolated by timestamp, expiry, and option type.  The estimate
uses neighboring log-moneyness observations and never uses the target row's IV.
"""

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "options_volatility_features.parquet"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "options_reference_iv.parquet"
MIN_VALID_POINTS = 2
MAX_EXTRAPOLATION_LOG_MONEYNESS = 0.03


def calculate_reference_iv(
    group: pd.DataFrame,
    min_valid_points: int = MIN_VALID_POINTS,
    max_extrapolation_log_moneyness: float = MAX_EXTRAPOLATION_LOG_MONEYNESS,
) -> pd.DataFrame:
    """Estimate reference IV row by row within one timestamp/expiry/type group.

    IVs at duplicate coordinates are median-aggregated after excluding the
    target row. Interior values are linearly interpolated; edge estimates use
    the nearest observation only within the configured log-moneyness distance.
    """
    required = {"implied_volatility", "strike"}
    missing = required.difference(group.columns)
    if missing:
        raise ValueError(f"Missing columns for reference IV: {sorted(missing)}")
    if min_valid_points < 2:
        raise ValueError("min_valid_points must be at least 2")

    out = group.copy()
    iv = pd.to_numeric(out["implied_volatility"], errors="coerce").to_numpy(float)
    strike = pd.to_numeric(out["strike"], errors="coerce").to_numpy(float)
    if "log_moneyness" in out:
        x = pd.to_numeric(out["log_moneyness"], errors="coerce").to_numpy(float)
    elif "nifty_spot" in out:
        spot = pd.to_numeric(out["nifty_spot"], errors="coerce").to_numpy(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            x = np.log(strike / spot)
    else:
        raise ValueError("Reference IV requires log_moneyness or nifty_spot")

    valid = np.isfinite(iv) & (iv > 0) & (iv <= 5.0) & np.isfinite(x)
    refs = np.full(len(out), np.nan, dtype=float)
    methods = np.full(len(out), "insufficient_points", dtype=object)
    valid_indices = np.flatnonzero(valid)
    order = np.argsort(x[valid_indices])
    sorted_indices = valid_indices[order]
    # The normal option chain has one valid contract at each coordinate. Compute
    # its leave-one-out neighbors in arrays; keep duplicate handling below.
    coordinates = x[sorted_indices]
    if len(coordinates) and np.all(np.diff(coordinates) > 1e-12):
        values = iv[sorted_indices]
        pos = np.searchsorted(coordinates, x, side="left")
        left, right = pos - 1, pos + valid.astype(int)
        enough = (len(coordinates) - valid.astype(int)) >= min_valid_points
        finite = np.isfinite(x)
        methods[~finite] = "invalid_target_coordinate"
        right_clipped = np.clip(right, 0, len(coordinates) - 1)
        exact = (finite & enough & ~valid & (right < len(coordinates))
                 & np.isclose(coordinates[right_clipped], x, rtol=0., atol=1e-12))
        refs[exact] = values[right_clipped[exact]]
        methods[exact] = "other_contract_same_coordinate"
        interior = finite & enough & ~exact & (left >= 0) & (right < len(coordinates))
        a, b = left[interior], right[interior]
        refs[interior] = values[a] + (x[interior] - coordinates[a]) * (values[b] - values[a]) / (coordinates[b] - coordinates[a])
        methods[interior] = "interpolated"
        edge = finite & enough & ~exact & ~interior
        nearest = np.clip(np.where(left < 0, right, left), 0, len(coordinates) - 1)
        close = np.abs(coordinates[nearest] - x) <= max_extrapolation_log_moneyness
        accepted = edge & close
        refs[accepted] = values[nearest[accepted]]
        methods[accepted] = "nearest_edge_extrapolation"
        methods[edge & ~close] = "extrapolation_too_far"
        out["reference_iv"] = refs
        out["reference_iv_method"] = methods
        return out
    sorted_x, inverse = np.unique(x[sorted_indices], return_inverse=True)
    sorted_iv = iv[sorted_indices]
    # Surface groups are small (normally one contract per strike); aggregate
    # duplicates once, then use array searches for each leave-one-out estimate.
    medians = np.array([
        np.median(sorted_iv[inverse == j]) for j in range(len(sorted_x))
    ])
    counts = np.bincount(inverse, minlength=len(sorted_x))
    unique_by_row = np.full(len(out), -1, dtype=int)
    unique_by_row[sorted_indices] = inverse

    for i, target_x in enumerate(x):
        if not np.isfinite(target_x):
            methods[i] = "invalid_target_coordinate"
            continue

        px = sorted_x
        py = medians
        own_coordinate = unique_by_row[i] if valid[i] else -1
        if own_coordinate >= 0:
            if counts[own_coordinate] == 1:
                keep = np.arange(len(sorted_x)) != own_coordinate
                px, py = sorted_x[keep], medians[keep]
            else:
                same_coordinate_rows = np.flatnonzero(
                    valid & np.isclose(x, target_x, rtol=0.0, atol=1e-12)
                )
                peers = iv[same_coordinate_rows[same_coordinate_rows != i]]
                if len(peers):
                    py = medians.copy()
                    py[own_coordinate] = np.median(peers)

        if len(px) < min_valid_points:
            continue
        pos = int(np.searchsorted(px, target_x, side="left"))
        if pos < len(px) and np.isclose(px[pos], target_x, rtol=0.0, atol=1e-12):
            refs[i] = py[pos]
            methods[i] = "other_contract_same_coordinate"
        elif 0 < pos < len(px):
            x0, x1 = px[pos - 1], px[pos]
            y0, y1 = py[pos - 1], py[pos]
            refs[i] = y0 + (target_x - x0) * (y1 - y0) / (x1 - x0)
            methods[i] = "interpolated"
        else:
            edge = 0 if pos == 0 else len(px) - 1
            if abs(px[edge] - target_x) <= max_extrapolation_log_moneyness:
                refs[i] = py[edge]
                methods[i] = "nearest_edge_extrapolation"
            else:
                methods[i] = "extrapolation_too_far"

    out["reference_iv"] = refs
    out["reference_iv_method"] = methods
    return out


def add_reference_iv_surface(df: pd.DataFrame) -> pd.DataFrame:
    """Add expiry-isolated leave-one-out reference IV and residual features."""
    keys = ["timestamp", "expiry", "option_type"]
    required = set(keys) | {"strike", "implied_volatility"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns for reference surface: {sorted(missing)}")

    out = df.copy()
    out["_reference_row_id"] = np.arange(len(out))
    # Carry only coordinates/IV through thousands of small surface groups.
    # Retaining all OHLC/Greeks columns per group needlessly multiplies memory.
    surface_columns = keys + ["strike", "implied_volatility", "_reference_row_id"]
    surface_columns += [c for c in ("log_moneyness", "nifty_spot") if c in out]
    parts = []
    for _, group in out[surface_columns].groupby(keys, sort=False, dropna=False):
        reference = calculate_reference_iv(group)
        parts.append(reference[["_reference_row_id", "reference_iv", "reference_iv_method"]])
    reference = pd.concat(parts, ignore_index=True).sort_values("_reference_row_id")
    out["reference_iv"] = reference.reference_iv.to_numpy()
    out["reference_iv_method"] = reference.reference_iv_method.to_numpy()
    out = out.drop(columns="_reference_row_id").reset_index(drop=True)
    out["iv_surface_residual"] = out["implied_volatility"] - out["reference_iv"]
    out["iv_surface_residual_pct"] = np.where(
        out["reference_iv"] > 0,
        out["iv_surface_residual"] / out["reference_iv"],
        np.nan,
    )
    return out


def main() -> None:
    print("BUILDING EXPIRY-ISOLATED REFERENCE IV SURFACE")
    print(f"Input:  {INPUT_FILE}")
    print(f"Output: {OUTPUT_FILE}")
    df = pd.read_parquet(INPUT_FILE)
    result = add_reference_iv_surface(df)
    print(f"Rows: {len(result):,}")
    print(f"Valid reference IV: {result['reference_iv'].notna().sum():,}")
    print(result["reference_iv_method"].value_counts(dropna=False).to_string())
    result.to_parquet(OUTPUT_FILE, index=False)
    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
