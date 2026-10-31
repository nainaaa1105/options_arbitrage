"""Zero-drift GBM/Heston markets with internally consistent option marks.

Simulation time is a trading year of 252*375 minutes. Pricing and variance use
that same clock. Rates/carry are zero; these are stylized, not calibrated paths.
"""

from dataclasses import dataclass, asdict
import json

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.special import ndtr

TRADING_MINUTES = 252 * 375


@dataclass(frozen=True)
class MarketSpec:
    model: str = "gbm"
    horizon_minutes: int = 30
    steps: int = 6
    sigma_low: float = .12
    sigma_high: float = .35
    maturity_days_low: float = 1.
    maturity_days_high: float = 7.
    kappa: float = 2.
    theta: float = .04
    vol_of_vol: float = .5
    rho: float = -.7
    substeps: int = 5

    def __post_init__(self):
        if self.model not in ("gbm", "heston") or self.horizon_minutes not in (30, 60):
            raise ValueError("Unsupported simulation market/horizon")
        if self.steps != self.horizon_minutes // 5 or self.substeps < 1:
            raise ValueError("Need five-minute decisions and positive substeps")
        if not (0 < self.sigma_low <= self.sigma_high < .65 and 0 < self.kappa and 0 < self.theta
                and 0 < self.vol_of_vol and -1 < self.rho < 1):
            raise ValueError("Invalid diffusion parameters")
        if not (.25 + self.horizon_minutes/375 < self.maturity_days_low <= self.maturity_days_high <= 10):
            raise ValueError("Option maturity must stay within the pricing grid through liquidation")


def bs_marks(spot, strike, maturity, variance, call):
    s, k, t, v, call = np.broadcast_arrays(spot, strike, maturity, variance, call)
    root = np.sqrt(np.maximum(v*t, 1e-16))
    d1 = (np.log(s/k) + .5*v*t) / root
    d2 = d1-root
    price = np.where(call, s*ndtr(d1)-k*ndtr(d2), k*ndtr(-d2)-s*ndtr(-d1))
    delta = ndtr(d1)-(~call.astype(bool)).astype(float)
    intrinsic = np.maximum(np.where(call, s-k, k-s), 0.)
    return np.where(t > 0, price, intrinsic), delta


def heston_cf(u, maturity, variance, kappa, theta, xi, rho):
    """Little-trap characteristic function of log(S_T/S_0), zero rates."""
    u = np.asarray(u, dtype=complex)
    if xi < 1e-5:
        integrated = theta*maturity+(variance-theta)*(-np.expm1(-kappa*maturity))/kappa
        return np.exp(-.5*(u*u+1j*u)*integrated)
    b = kappa-rho*xi*1j*u
    d = np.sqrt(b*b+xi*xi*(u*u+1j*u))
    g = (b-d)/(b+d)
    exponential = np.exp(-d*maturity)
    c = kappa*theta/(xi*xi)*((b-d)*maturity - 2*np.log((1-g*exponential)/(1-g)))
    coefficient = (b-d)/(xi*xi)*(1-exponential)/(1-g*exponential)
    return np.exp(c+coefficient*variance)


def integration_nodes():
    z, w = np.polynomial.legendre.leggauss(64)
    bounds = (0., 1., 5., 20., 100., 500., 2000.)
    nodes, weights = [], []
    for a, b in zip(bounds[:-1], bounds[1:]):
        nodes.append((a+b)/2+(b-a)/2*z)
        weights.append((b-a)/2*w)
    return np.concatenate(nodes), np.concatenate(weights)


class HestonOracle:
    """Fourier call-price and spot-delta grid, shared by all compared policies.

    Uses normalized strike=1 and zero rates. Out-of-grid observations fail rather
    than extrapolating silently. Saved grids make model marks reproducible.
    """

    def __init__(self, spec):
        self.spec = spec
        self.x = np.linspace(-.10, .10, 401)
        self.v = np.linspace(.06, .65, 65)**2
        self.t = np.unique(np.r_[np.linspace(.25, 2, 15), np.linspace(2.5, 10, 16)]) / 252
        u, weights = integration_nodes()
        vv, tt = np.meshgrid(self.v, self.t, indexing="ij")
        phi = heston_cf(u[None, :]-.5j, tt.ravel()[:, None], vv.ravel()[:, None],
                        spec.kappa, spec.theta, spec.vol_of_vol, spec.rho)
        weighted = phi * (weights/(u*u+.25))[None, :]
        phase = np.exp(1j*u[:, None]*self.x[None, :])
        integral = np.real(weighted @ phase)
        derivative = np.real(weighted @ (1j*u[:, None]*phase))
        s = np.exp(self.x)[None, :]
        price = s-np.sqrt(s)*integral/np.pi
        delta = 1-(.5*integral+derivative)/(np.pi*np.sqrt(s))
        # Only tiny numerical lower-bound errors may be clipped.
        lower = np.maximum(s-1., 0.)
        if np.min(price-lower) < -2e-5 or np.max(price-s) > 2e-5:
            raise ArithmeticError(f"Heston Fourier grid violates call bounds: lower={np.min(price-lower):.8g}, upper={np.max(price-s):.8g}")
        price = np.maximum(price, lower)
        price = np.minimum(price, s)
        delta = np.clip(delta, 0., 1.)
        shape = (len(self.v), len(self.t), len(self.x))
        self.price_grid = price.reshape(shape).transpose(2, 0, 1)
        self.delta_grid = delta.reshape(shape).transpose(2, 0, 1)
        self._price = RegularGridInterpolator((self.x, self.v, self.t), self.price_grid, bounds_error=True)
        self._delta = RegularGridInterpolator((self.x, self.v, self.t), self.delta_grid, bounds_error=True)

    def marks(self, spot, strike, maturity, variance, call):
        s, k, t, v, call = np.broadcast_arrays(spot, strike, maturity, variance, call)
        points = np.column_stack([np.log(s/k).ravel(), v.ravel(), t.ravel()])
        cp = self._price(points).reshape(s.shape)*k
        cd = self._delta(points).reshape(s.shape)
        price = np.where(call, cp, cp-s+k)
        delta = np.where(call, cd, cd-1)
        intrinsic = np.maximum(np.where(call, s-k, k-s), 0.)
        return np.maximum(price, intrinsic), delta

    def direct_marks(self, spot, strike, maturity, variance, call):
        s, k, t, v, call = np.broadcast_arrays(spot, strike, maturity, variance, call)
        u, w = integration_nodes()
        phi = heston_cf(u[None, :]-.5j, t.ravel()[:, None], v.ravel()[:, None],
                        self.spec.kappa, self.spec.theta, self.spec.vol_of_vol, self.spec.rho)
        phase = np.exp(1j*np.log(s/k).ravel()[:, None]*u[None, :])
        integrand = phi*phase*(w/(u*u+.25))[None, :]
        integral = integrand.real.sum(axis=1).reshape(s.shape)
        derivative = (integrand*1j*u[None, :]).real.sum(axis=1).reshape(s.shape)
        cp = s-np.sqrt(s*k)*integral/np.pi
        cd = 1-(.5*integral+derivative)/(np.pi*np.sqrt(s/k))
        return np.where(call, cp, cp-s+k), np.where(call, cd, cd-1)

    def save(self, path):
        np.savez_compressed(path, x=self.x, variance=self.v, maturity=self.t,
                            call_price=self.price_grid, call_delta=self.delta_grid,
                            spec_json=json.dumps(asdict(self.spec)))


def simulate(spec, n, seed=None, rng=None, oracle=None):
    if n < 1:
        raise ValueError("Positive path count required")
    rng = rng if rng is not None else np.random.default_rng(seed)
    steps, internal = spec.steps, spec.substeps
    dt = spec.horizon_minutes / (TRADING_MINUTES*steps*internal)
    spot = np.empty((n, steps+1))
    variance = np.empty_like(spot)
    spot[:, 0] = rng.uniform(21000, 26000, n)
    variance[:, 0] = rng.uniform(spec.sigma_low, spec.sigma_high, n)**2
    strike = spot[:, 0]*np.exp(rng.uniform(-.02, .02, n))
    initial_time = rng.uniform(spec.maturity_days_low, spec.maturity_days_high, n)/252
    call = rng.integers(0, 2, n).astype(bool)
    s, v = spot[:, 0].copy(), variance[:, 0].copy()
    for step in range(1, steps+1):
        for _ in range(internal):
            z1, z2 = rng.standard_normal((2, n))
            vp = np.maximum(v, 0.)
            s *= np.exp(-.5*vp*dt + np.sqrt(vp*dt)*z1)
            if spec.model == "heston":
                zvar = spec.rho*z1+np.sqrt(1-spec.rho**2)*z2
                v += spec.kappa*(spec.theta-vp)*dt+spec.vol_of_vol*np.sqrt(vp*dt)*zvar
        spot[:, step], variance[:, step] = s, np.maximum(v, 0.)
    maturity = initial_time[:, None]-np.arange(steps+1)[None, :]*(spec.horizon_minutes/(TRADING_MINUTES*steps))
    bs_price, bs_delta = bs_marks(spot, strike[:, None], maturity, variance, call[:, None])
    if spec.model == "heston":
        if oracle is None:
            raise ValueError("Heston paths require the shared model-consistent pricing oracle")
        price, delta = oracle.marks(spot, strike[:, None], maturity, variance, call[:, None])
    else:
        price, delta = bs_price, bs_delta
    result = {"spot": spot, "future": spot.copy(), "variance": variance, "maturity": maturity,
              "strike": strike, "call": call, "option": price, "delta": delta, "bs_delta": bs_delta}
    if not all(np.isfinite(value).all() for value in result.values()):
        raise ArithmeticError("Non-finite simulation output")
    return result
