"""Risk-neutral zero-rate GBM; ACT-independent year fraction shared with pricing."""
import math
import torch


def bs(spot, strike, maturity, sigma, call):
    """Vector price/delta/gamma; terminal payoff has an explicit boundary."""
    t = torch.as_tensor(maturity, dtype=spot.dtype, device=spot.device)
    s = torch.as_tensor(sigma, dtype=spot.dtype, device=spot.device)
    call = torch.as_tensor(call, dtype=torch.bool, device=spot.device)
    if not torch.isfinite(spot).all() or (spot<=0).any() or not math.isfinite(strike) or strike<=0 or not torch.isfinite(t).all() or (t<0).any() or not torch.isfinite(s).all() or (s<=0).any():
        raise ValueError("Positive spot/strike/volatility and nonnegative finite maturity required")
    scale = s*torch.sqrt(t.clamp_min(1e-12))
    d1 = (torch.log(spot/strike)+.5*s*s*t)/scale
    d1-scale
    cdf = lambda x: .5*(1+torch.erf(x/math.sqrt(2)))
    # Premium cancellation matters for low-volatility, far-OTM options. Evaluate
    # each option side directly in float64 with the normal survival function;
    # keep hedge Greeks in their original dtype for checkpoint compatibility.
    sd, td, vd = spot.double(), t.double(), s.double()
    scale64 = vd*torch.sqrt(td.clamp_min(1e-12))
    d164 = (torch.log(sd/strike)+.5*vd*vd*td)/scale64
    d264 = d164-scale64
    normal = lambda x: .5*torch.erfc(-x/math.sqrt(2))
    cp = sd*normal(d164)-strike*normal(d264)
    pp = strike*normal(-d264)-sd*normal(-d164)
    intrinsic64 = torch.where(call, sd-strike, strike-sd).clamp_min(0)
    upper64 = torch.where(call, sd, torch.full_like(sd, strike))
    price = torch.minimum(torch.maximum(torch.where(call, cp, pp), intrinsic64), upper64).to(spot.dtype)
    delta = torch.where(call, cdf(d1), cdf(d1)-1)
    gamma = torch.exp(-.5*d1*d1)/(math.sqrt(2*math.pi)*spot*scale)
    payoff = torch.where(call, (spot-strike).clamp_min(0), (strike-spot).clamp_min(0))
    terminal_delta = torch.where(spot>strike, 1., torch.where(spot<strike, 0., .5))
    terminal_delta = torch.where(call, terminal_delta, terminal_delta-1)
    return (torch.where(t>0, price, payoff), torch.where(t>0, delta, terminal_delta), torch.where(t>0, gamma, 0.))


def simulate(config, count, seed=None, generator=None, dtype=torch.float32):
    if count < 1:
        raise ValueError("Need paths")
    if generator is not None and seed is not None:
        raise ValueError("Use either a seed or a continuing generator")
    g = generator if generator is not None else torch.Generator().manual_seed(seed if seed is not None else 0)
    initial = config.strike*(.9+.2*torch.rand(count, generator=g, dtype=dtype))
    sigma = config.sigma_low+(config.sigma_high-config.sigma_low)*torch.rand(count, generator=g, dtype=dtype)
    call = torch.rand(count, generator=g)>.5
    dt = config.maturity/config.steps
    z = torch.randn(count, config.steps, generator=g, dtype=dtype)
    increments = -.5*sigma[:,None]**2*dt+sigma[:,None]*math.sqrt(dt)*z
    spot = initial[:,None]*torch.exp(torch.cat([torch.zeros(count,1,dtype=dtype), increments.cumsum(1)], dim=1))
    maturity = torch.linspace(config.maturity, 0, config.steps+1,dtype=dtype)[None,:].expand(count,-1)
    price, delta, gamma = bs(spot, config.strike, maturity, sigma[:,None], call[:,None])
    return {"spot":spot, "sigma":sigma, "call":call, "maturity":maturity,
            "premium":price[:,0], "payoff":price[:,-1], "delta":delta[:,:-1], "gamma":gamma[:,:-1]}


def coarsen(batch, steps):
    finest = batch["spot"].shape[1]-1
    if finest%steps:
        raise ValueError("Coarse grid must divide fine grid")
    stride = finest//steps
    return {key:(value[:,::stride] if key in ("spot","maturity","volatility_path") else value[:,::stride] if key in ("delta","gamma") else value)
            for key,value in batch.items()}
