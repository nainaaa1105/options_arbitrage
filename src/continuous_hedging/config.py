from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class ResearchConfig:
    steps: int = 32
    maturity: float = 20/252
    strike: float = 1.
    sigma_low: float = .15
    sigma_high: float = .30
    cost_rate: float = 0.
    fixed_cost: float = 0.
    half_spread: float = 0.
    max_position: float = 1.2
    alpha: float = .95
    entropic_aversion: float = 100.
    risk_scale: float = .02

    def __post_init__(self):
        if any(not math.isfinite(v) or v < 0 for v in asdict(self).values()):
            raise ValueError("Settings must be finite and nonnegative")
        if int(self.steps) != self.steps or self.steps < 2 or min(self.maturity, self.strike, self.sigma_low, self.risk_scale, self.entropic_aversion) <= 0:
            raise ValueError("Invalid market specification")
        if self.sigma_high < self.sigma_low or self.max_position < 1 or not 0 < self.alpha < 1:
            raise ValueError("Invalid volatility/action/risk range")


@dataclass(frozen=True)
class FitConfig:
    seeds: tuple = (11, 22, 33)
    epochs: int = 12
    flat_epochs: int = 48
    updates: int = 24
    batch: int = 1024
    validation_paths: int = 4096
    test_paths: int = 16384
    learning_rate: float = .002
    delta_mae_tolerance: float = .06
    std_ratio_tolerance: float = 1.20

    def __post_init__(self):
        if min(self.epochs, self.flat_epochs, self.updates, self.batch, self.validation_paths, self.test_paths) < 1 or len(set(self.seeds)) != len(self.seeds) or not self.seeds:
            raise ValueError("Invalid fit budget/seeds")
        if min(self.learning_rate,self.delta_mae_tolerance,self.std_ratio_tolerance) <= 0:
            raise ValueError("Invalid optimizer/gates")
