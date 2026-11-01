from dataclasses import replace
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from continuous_hedging.config import ResearchConfig,FitConfig
from continuous_hedging.market import bs,simulate,coarsen
from continuous_hedging.accounting import pnl
from continuous_hedging.objectives import cvar,entropic,risk
from continuous_hedging.policies import ContinuousPolicy,positions
from continuous_hedging.baselines import baseline,ww_width
from continuous_hedging.training import fit,load
from continuous_hedging.evaluation import paired_intervals


class ContinuousTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_hand_calculated_short_call_with_opening_and_terminal_costs(self):
        c=ResearchConfig(steps=3,strike=100.,cost_rate=.01,fixed_cost=2.,half_spread=.1)
        b={'spot':torch.tensor([[100.,110.,105.,120.]],dtype=torch.float64),
           'premium':torch.tensor([10.],dtype=torch.float64),'payoff':torch.tensor([20.],dtype=torch.float64)}
        h=torch.tensor([[.5,.6,.4]],dtype=torch.float64)
        result=pnl(b,h,c)
        # Gains=5-3+6=8; changes=.5,.1,-.2,-.4, including liquidation.
        expected_cost=.01*(50+11+21+48)+.1*1.2+2*4
        self.assertAlmostEqual(float(result['net_pnl']),10-20+8-expected_cost,places=10)
        self.assertAlmostEqual(float(result['cost']),expected_cost,places=10)
        self.assertLess(float(result['reconciliation_error'].abs().max()),1e-10)

    def test_terminal_payoff_put_and_price_parity(self):
        s=torch.tensor([.8,1.,1.2],dtype=torch.float64)
        t=torch.zeros(3,dtype=torch.float64)
        price,delta,gamma=bs(s,1.,t,.2,torch.zeros(3,dtype=torch.bool))
        torch.testing.assert_close(price,torch.tensor([.2,0,0],dtype=torch.float64))
        self.assertTrue(torch.isfinite(delta).all())
        self.assertTrue(gamma.eq(0).all())
        cp,_,_=bs(s,1.,.1,.2,torch.ones(3,dtype=torch.bool))
        pp,_,_=bs(s,1.,.1,.2,torch.zeros(3,dtype=torch.bool))
        torch.testing.assert_close(cp-pp,s-1)

    def test_bs_delta_gamma_finite_difference(self):
        s=torch.tensor([.9,1.,1.1],dtype=torch.float64)
        price,d,g=bs(s,1.,.08,.2,True)
        plus=bs(s+1e-5,1.,.08,.2,True)[0];minus=bs(s-1e-5,1.,.08,.2,True)[0]
        torch.testing.assert_close(d,(plus-minus)/2e-5,atol=1e-7,rtol=1e-6)
        torch.testing.assert_close(g,(plus-2*price+minus)/1e-10,atol=3e-6,rtol=1e-5)

    def test_seed_and_continuing_generator(self):
        c=ResearchConfig()
        a,b=simulate(c,32,seed=5),simulate(c,32,seed=5)
        for key in a:torch.testing.assert_close(a[key],b[key],atol=0,rtol=0)
        generator=torch.Generator().manual_seed(1)
        a,b=simulate(c,32,generator=generator),simulate(c,32,generator=generator)
        self.assertFalse(torch.equal(a['spot'],b['spot']))

    def test_zero_drift_gbm_martingale(self):
        b=simulate(ResearchConfig(),20000,seed=4)
        ratio=b['spot'][:,-1]/b['spot'][:,0]
        self.assertLess(abs(float(ratio.mean())-1),5*float(ratio.std())/len(ratio)**.5)

    def test_future_price_perturbation_cannot_change_past_actions(self):
        for cost in (0.,.001):
            c=ResearchConfig(steps=8,cost_rate=cost)
            a=simulate(c,16,seed=82);b={k:v.clone() for k,v in a.items()}
            for key in ('spot','delta','gamma'):
                b[key][:,4:]*=1.1
            model=ContinuousPolicy('flat')
            torch.nn.init.normal_(model.network[-1].weight,std=.1)
            torch.testing.assert_close(positions(model,a,c)[:,:4],positions(model,b,c)[:,:4],atol=0,rtol=0)

    def test_accounting_gradcheck(self):
        c=ResearchConfig(steps=3,cost_rate=.002)
        b=simulate(c,4,seed=2,dtype=torch.float64)
        h=torch.tensor([[.2,.3,.4]]*4,dtype=torch.float64,requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(lambda x:pnl(b,x,c)['net_pnl'],(h,)))

    def test_cvar_fractional_tail_and_translation(self):
        loss=torch.tensor([0.,1.,2.,3.],dtype=torch.float64)
        self.assertAlmostEqual(float(cvar(loss,.625)),(3+.5*2)/1.5)
        self.assertAlmostEqual(float(cvar(loss+5,.625)),float(cvar(loss,.625))+5)
        self.assertAlmostEqual(float(cvar(loss,.2)),6/3.2)

    def test_entropic_stable_and_zero_aversion_limit(self):
        loss=torch.tensor([1000.,1001.,1002.],dtype=torch.float64)
        self.assertTrue(torch.isfinite(entropic(loss,100)))
        self.assertAlmostEqual(float(entropic(loss,1e-6)),1001.,places=5)

    def test_gradients_reach_network_and_eta(self):
        c=ResearchConfig(steps=8)
        b=simulate(c,256,seed=2)
        model=ContinuousPolicy()
        eta=torch.nn.Parameter(torch.tensor(0.))
        loss=risk(pnl(b,positions(model,b,c),c)['net_pnl'],c,'cvar',eta)
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        self.assertGreater(float(model.network[-1].weight.grad.abs().sum()),0)
        self.assertNotEqual(float(eta.grad),0)

    def test_whalley_width_dimensions_and_cost_scaling(self):
        s,g=torch.tensor([1.]),torch.tensor([5.])
        w=ww_width(s,g,.001,100.)
        self.assertAlmostEqual(float(w),(.001*1.5*25/100)**(1/3),places=6)
        torch.testing.assert_close(ww_width(s,g,.008,100.),2*w)
        torch.testing.assert_close(ww_width(s,g,0.,100.),torch.zeros(1))

    def test_band_trades_to_edge_and_holds_inside(self):
        c=ResearchConfig(steps=3)
        b={'delta':torch.tensor([[.5,.55,.8]])}
        h=baseline(b,c,'constant_band',width=.1)
        torch.testing.assert_close(h,torch.tensor([[.4,.45,.7]]))
        b['delta']=torch.tensor([[.5,.49,.45]])
        torch.testing.assert_close(baseline(b,c,'constant_band',width=.1),torch.tensor([[.4,.4,.4]]))

    def test_no_cost_bands_equal_delta(self):
        c=ResearchConfig();b=simulate(c,32,seed=99)
        torch.testing.assert_close(baseline(b,c,'whalley_wilmott'),b['delta'])
        torch.testing.assert_close(baseline(b,c,'constant_band'),b['delta'])

    def test_delta_hedge_std_declines_with_frequency(self):
        c=ResearchConfig(steps=128);b=simulate(c,10000,seed=984)
        coarse=coarsen(b,8)
        a=pnl(coarse,baseline(coarse,replace(c,steps=8),'delta'),replace(c,steps=8))['net_pnl'].std()
        z=pnl(b,baseline(b,c,'delta'),c)['net_pnl'].std()
        self.assertLess(float(z),float(a)*.5)

    def test_module_needs_no_market_data_or_signals(self):
        for file in (Path(__file__).resolve().parents[1]/'src/continuous_hedging').glob('*.py'):
            self.assertNotIn('import train_models',file.read_text())
            self.assertNotIn('from deep_hedging',file.read_text())
        # Sampling, forward pass and a gradient work with all file reads denied.
        with patch('pandas.read_parquet',side_effect=AssertionError('External data access')):
            c=ResearchConfig(steps=4);b=simulate(c,32,seed=9)
            model=ContinuousPolicy();risk(pnl(b,positions(model,b,c),c)['net_pnl'],c,'entropic').backward()

    def test_tiny_actual_fit_and_checkpoint_replay(self):
        c=ResearchConfig(steps=8)
        settings=FitConfig(seeds=(9,),epochs=1,updates=2,batch=64,validation_paths=128,test_paths=128)
        b=simulate(c,128,seed=4)
        with TemporaryDirectory() as tmp:
            model,saved=fit(c,settings,b,'cvar','flat',9,Path(tmp))
            replay,_=load(Path(tmp)/saved['checkpoint'])
            torch.testing.assert_close(positions(model,b,c),positions(replay,b,c),atol=0,rtol=0)

    def test_bad_config_or_shape_rejected(self):
        with self.assertRaises(ValueError):ResearchConfig(alpha=1.)
        c=ResearchConfig();b=simulate(c,4,seed=1)
        with self.assertRaises(ValueError):pnl(b,torch.zeros(4,3),c)
        with self.assertRaises(ValueError):bs(torch.tensor([-1.]),1.,.1,.2,True)

    def test_fixed_trade_count_cost_not_claimed_differentiable(self):
        c=ResearchConfig(steps=4,fixed_cost=.01)
        b=simulate(c,32,seed=6)
        with TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError,'nondifferentiable'):
                fit(c,FitConfig(epochs=1),b,'cvar','flat',1,Path(tmp))

    def test_output_parameterizations_are_explicit_and_bounded(self):
        c=ResearchConfig();b=simulate(c,16,seed=11)
        for parameterization in ('sigmoid','projected'):
            model=ContinuousPolicy('delta',parameterization=parameterization)
            h=positions(model,b,c)
            self.assertLess(float(h.abs().max()),1.200001)
            torch.testing.assert_close(h,b['delta'],atol=2e-6,rtol=0)
        with self.assertRaises(ValueError):ContinuousPolicy(parameterization='unknown')

    def test_paired_risk_interval_preserves_constant_shift(self):
        a=np.linspace(-.2,.3,100)
        result=paired_intervals(a-.01,a,repeats=20)
        self.assertAlmostEqual(result['mean_pnl_difference'],-.01)
        self.assertAlmostEqual(result['cvar95_difference'],.01)
        self.assertAlmostEqual(result['cvar95_lower95'],.01)
        self.assertAlmostEqual(result['cvar95_upper95'],.01)


if __name__=='__main__':unittest.main()
