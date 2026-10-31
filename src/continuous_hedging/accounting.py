"""Self-financing zero-rate cash accounting for one short European option."""
import torch


def pnl(batch, positions, config):
    spot = batch["spot"]
    if positions.shape != spot[:,:-1].shape or not torch.isfinite(positions).all() or (positions.abs()>config.max_position+1e-6).any():
        raise ValueError("Need finite bounded pre-move positions [paths, steps]")
    if not torch.isfinite(spot).all() or (spot<=0).any():
        raise ValueError("Need positive finite spot")
    changes = torch.diff(torch.cat([torch.zeros_like(positions[:,:1]), positions, torch.zeros_like(positions[:,:1])],dim=1),dim=1)
    proportional = config.cost_rate*(changes.abs()*spot).sum(1)
    spread = config.half_spread*changes.abs().sum(1)
    fixed = config.fixed_cost*(changes.detach().abs()>1e-10).sum(1)
    gain = (positions*torch.diff(spot,dim=1)).sum(1)
    gross = batch["premium"]-batch["payoff"]+gain
    net = gross-proportional-spread-fixed
    # Independent stock/cash ledger includes purchase/sale notionals and the
    # terminal sale. These notionals telescope into trading gains at r=0.
    cash = batch["premium"]-(changes*spot).sum(1)-batch["payoff"]-proportional-spread-fixed
    if not torch.allclose(cash.detach(),net.detach(),atol=2e-6,rtol=1e-5):
        raise ArithmeticError("Continuous cash/P&L reconciliation failed")
    return {"net_pnl":net,"gross_pnl":gross,"cost":proportional+spread+fixed,
            "proportional_cost":proportional,"spread_cost":spread,"fixed_cost":fixed,
            "turnover_units":changes.abs().sum(1),"turnover_notional":(changes.abs()*spot).sum(1),
            "trades":(changes.detach().abs()>1e-10).sum(1),"reconciliation_error":cash-net}
