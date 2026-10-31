"""Explicit, serializable assumptions; these are not historical SPAN margins."""

from dataclasses import dataclass, asdict
import math


@dataclass(frozen=True)
class HedgeConfig:
    initial_capital: float = 5_000_000.
    option_lots: int = 4
    option_side: int = 1
    horizon_minutes: int = 30
    bar_minutes: int = 5
    max_future_lots: int = 4
    future_margin_fraction: float = .25
    short_option_margin_per_lot: float = 0.
    # Fraction of option units: .25 of four equal-sized lots is one lot.
    # A band below the half-lot rounding dead zone is redundant.
    no_trade_band: float = .25
    option_half_spread: float = .15
    option_slippage: float = .15
    future_half_spread: float = .10
    future_slippage: float = .10
    option_tick_size: float = .05
    brokerage_per_order: float = 20.
    future_brokerage_cap_rate: float = .0005
    option_exchange_rate: float = .0003503
    future_exchange_rate: float = .0000173
    sebi_rate: float = .000001
    option_ipft_rate: float = .000005
    future_ipft_rate: float = .000001
    option_stamp_rate: float = .00003
    future_stamp_rate: float = .00002
    gst_rate: float = .18
    # None enables the date-dependent STT schedule; 0 is useful for tests.
    option_stt_override: float | None = None
    future_stt_override: float | None = None
    funding_rate: float = 0.
    risk_free_rate: float = 0.
    capital_fraction: float = .95
    cost_basis: str = "configured_proxy"

    def __post_init__(self):
        for name, value in asdict(self).items():
            if isinstance(value, (int, float)) and (not math.isfinite(value) or value < 0) and name != "option_side":
                raise ValueError(f"Invalid nonnegative finite setting: {name}")
        if self.initial_capital <= 0 or self.option_tick_size <= 0:
            raise ValueError("Capital and tick size must be positive")
        if self.option_side not in (-1, 1):
            raise ValueError("option_side must be -1 or +1")
        if self.option_side == -1 and self.short_option_margin_per_lot <= 0:
            raise ValueError("Short options require an explicit margin proxy")
        if self.bar_minutes != 5 or self.horizon_minutes not in (30, 60):
            raise ValueError("Part 1 supports five-minute data and 30/60-minute horizons")
        if self.option_lots < 1 or int(self.option_lots) != self.option_lots:
            raise ValueError("option_lots must be a positive integer")
        if self.max_future_lots < 0 or int(self.max_future_lots) != self.max_future_lots:
            raise ValueError("max_future_lots must be a nonnegative integer")
        if not 0 < self.future_margin_fraction <= 1 or not 0 < self.capital_fraction <= 1:
            raise ValueError("Margin and capital fractions must be in (0,1]")
        if self.cost_basis != "configured_proxy":
            raise ValueError("This OHLC engine must be labelled configured_proxy")

    def to_dict(self):
        return asdict(self)
