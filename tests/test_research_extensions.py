from dataclasses import replace
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
import numpy as np
import pandas as pd
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from continuous_hedging.config import ResearchConfig,FitConfig
from continuous_hedging.market import simulate,bs,coarsen
from continuous_hedging.policies import BandPolicy,ContinuousPolicy,features,positions
from continuous_hedging.training import fit,load
from continuous_hedging.rounding import round_units,rounded_policy,rounded_baseline
from continuous_hedging.lot_training import LotPolicy,rollout,fit_lot,load_lot
from continuous_hedging.accounting import pnl
from continuous_hedging.worlds import calibrate,HestonPremium,WorldSampler,heston_cf,split_returns
from continuous_hedging.sensitivity import observable_variance_proxy,execution_stresses


class ExtensionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.p={'sigma_low':.12,'sigma_high':.3,'kappa':2.,'theta':.04,'xi':.5,'rho':-.7,'bootstrap_log_center':0.}
        cls.c=ResearchConfig(steps=20,cost_rate=.001)
        cls.oracle=HestonPremium(cls.p,cls.c.maturity,.12,.3)

    def test_band_bounds_do_not_use_inventory_and_hold_exactly(self):
        c=ResearchConfig(steps=4,cost_rate=.001);b=simulate(c,16,seed=3);model=BandPolicy()
        x=features(b,c,torch.zeros(16),0);lo,hi=model.bounds(x)
        previous=(lo+hi)/2;x[:,5]=previous/c.max_position
        lo2,hi2=model.bounds(x)
        torch.testing.assert_close(lo,lo2);torch.testing.assert_close(hi,hi2)
        result=model(x);expected=previous*torch.where(b['call'],1.,-1.)
        torch.testing.assert_close(result,expected)

    def test_band_causal_rollout_and_cash_reconcile(self):
        c=ResearchConfig(steps=4,cost_rate=.001);b=simulate(c,32,seed=4);other={k:v.clone() for k,v in b.items()}
        other['spot'][:,3:]*=1.1;other['delta'][:,3:]=0.
        model=BandPolicy();a=positions(model,b,c);z=positions(model,other,c)
        torch.testing.assert_close(a[:,:3],z[:,:3],atol=0,rtol=0)
        self.assertLess(float(pnl(b,a,c)['reconciliation_error'].abs().max()),1e-6)

    def test_band_checkpoint_schema_and_real_fit(self):
        c=ResearchConfig(steps=4,cost_rate=.001);b=simulate(c,64,seed=5)
        with TemporaryDirectory() as tmp:
            folder=Path(tmp);m,s=fit(c,FitConfig(epochs=1,updates=2,batch=64),b,'entropic','delta',11,folder,family='band')
            reloaded,_=load(folder/s['checkpoint'])
            torch.testing.assert_close(positions(m,b,c),positions(reloaded,b,c),atol=0,rtol=0)

    def test_rounding_ties_bounds_and_units(self):
        h=torch.tensor([.125,.375,-.125,-.375,1.2])
        torch.testing.assert_close(round_units(h,4),torch.tensor([0.,.25,0.,-.25,1.]))
        for n in (1,4,10,40):
            result=round_units(torch.linspace(-1.2,1.2,501),n)
            torch.testing.assert_close(result*n,(result*n).round(),atol=1e-5,rtol=0)
            self.assertLessEqual(float(result.abs().max()),1.200001)

    def test_rounding_is_closed_loop_not_open_loop(self):
        class Toy(torch.nn.Module):
            def forward(self,x):return .4+.5*x[:,5]*1.2
        c=ResearchConfig(steps=4,cost_rate=.001);b=simulate(c,8,seed=3);b['call'][:]=True
        teacher=Toy();actual=rounded_policy(teacher,b,c,1)
        naive=round_units(positions(teacher,b,c),1)
        self.assertTrue(actual.eq(0).all())
        self.assertTrue(naive[:,1:].eq(1).all())

    def test_rounded_delta_cash_and_book_scaling(self):
        c=ResearchConfig(steps=4,cost_rate=.001);b=simulate(c,64,seed=4)
        h=rounded_baseline(b,c,4,'delta');r=pnl(b,h,c)
        self.assertLess(float(r['reconciliation_error'].abs().max()),1e-6)
        self.assertTrue(torch.isfinite(r['net_pnl']*4).all())

    def test_categorical_whole_lots_and_gradient(self):
        c=ResearchConfig(steps=4,cost_rate=.001);b=simulate(c,128,seed=8);m=LotPolicy(4)
        r,h,logp,entropy=rollout(m,b,c,True)
        torch.testing.assert_close(h*4,(h*4).round(),atol=0,rtol=0)
        loss=(-r['net_pnl'].detach()*logp).mean()-.001*entropy.mean();loss.backward()
        self.assertGreater(float(m.network[-1].weight.grad.abs().sum()),0)

    def test_tiny_categorical_fit_reload(self):
        c=ResearchConfig(steps=4,cost_rate=.001);b=simulate(c,64,seed=91)
        with TemporaryDirectory() as tmp:
            folder=Path(tmp);settings=FitConfig(epochs=1,updates=2,batch=64)
            saved=fit_lot(c,settings,b,ContinuousPolicy('delta'),4,11,folder)
            a,_=load_lot(folder/'policy_seed_11.pt');z,_=load_lot(folder/'policy_seed_11.pt')
            torch.testing.assert_close(rollout(a,b,c)[1],rollout(z,b,c)[1],atol=0,rtol=0)
            self.assertIn(saved['best_epoch'],(0,1))

    def test_daily_calibration_cannot_use_later_returns(self):
        rng=np.random.default_rng(6);dates=pd.bdate_range('2025-01-01','2026-10-29').date
        data=pd.DataFrame({'date':dates,'return':rng.normal(0,.01,len(dates))});changed=data.copy()
        changed.loc[changed.date>=pd.Timestamp('2026-07-01').date(),'return']=10.
        self.assertEqual(calibrate(split_returns(data)['train']),calibrate(split_returns(changed)['train']))

    def test_heston_cf_and_constant_variance_limit(self):
        self.assertAlmostEqual(abs(heston_cf(0,.1,.04,2,.04,.5,-.7)-1),0,places=12)
        self.assertAlmostEqual(abs(heston_cf(-1j,.1,.04,2,.04,.5,-.7)-1),0,places=12)
        parameters={**self.p,'xi':0.}
        oracle=HestonPremium(parameters,self.c.maturity,.12,.3)
        s=np.array([.95,1.,1.05]);v=np.full(3,.04);call=np.ones(3,bool)
        price=oracle.direct(s,v,call)
        expected=bs(torch.tensor(s),1.,self.c.maturity,.2,True)[0].numpy()
        np.testing.assert_allclose(price,expected,atol=1e-7)

    def test_heston_premium_interpolation_audit_and_bounds(self):
        rng=np.random.default_rng(3);s=rng.uniform(.9,1.1,100);v=rng.uniform(.12,.3,100)**2
        call=np.ones(100,bool);price=self.oracle.marks(s,v,call)
        self.assertLess(float(np.max(abs(price-self.oracle.direct(s,v,call)))),2e-4)
        self.assertTrue((price>=np.maximum(s-1,0)-1e-8).all())
        with self.assertRaises(ValueError):self.oracle.marks(np.array([2.]),np.array([.04]),True)

    def test_heston_seed_reproducible_and_positive(self):
        sampler=WorldSampler('heston',self.p,oracle=self.oracle)
        a=sampler(self.c,64,seed=4);b=sampler(self.c,64,seed=4)
        for name in a:torch.testing.assert_close(a[name],b[name],atol=0,rtol=0)
        self.assertTrue(a['spot'].gt(0).all())
        self.assertTrue(a['volatility_path'].gt(0).all())

    def test_bootstrap_rv_is_strictly_past_with_explicit_daily_clock(self):
        sampler=WorldSampler('bootstrap',self.p,np.linspace(-.02,.02,80))
        b=sampler(self.c,64,seed=3);again=sampler(self.c,64,seed=3)
        torch.testing.assert_close(b['spot'],again['spot'],atol=0,rtol=0)
        self.assertEqual(b['volatility_path'].shape,b['spot'].shape)
        with self.assertRaisesRegex(ValueError,'20 daily'):sampler(replace(self.c,steps=32),64,seed=3)

    def test_future_volatility_cannot_change_past_hedges(self):
        b=WorldSampler('heston',self.p,oracle=self.oracle)(self.c,64,seed=9)
        other={k:v.clone() for k,v in b.items()};other['volatility_path'][:,5:]*=2
        m=ContinuousPolicy('delta')
        torch.testing.assert_close(positions(m,b,self.c)[:,:5],positions(m,other,self.c)[:,:5],atol=0,rtol=0)

    def test_coarsening_samples_current_volatility_on_same_grid(self):
        b=WorldSampler('heston',self.p,oracle=self.oracle)(self.c,64,seed=9)
        coarse=coarsen(b,5)
        torch.testing.assert_close(coarse['volatility_path'],b['volatility_path'][:,::4])
        self.assertEqual(coarse['volatility_path'].shape,coarse['spot'].shape)

    def test_world_sampler_training_hook_and_reloaded_features(self):
        sampler=WorldSampler('bootstrap',self.p,np.random.default_rng(3).normal(0,.01,150))
        b=sampler(self.c,64,seed=4)
        with TemporaryDirectory() as tmp:
            folder=Path(tmp);m,s=fit(self.c,FitConfig(epochs=1,updates=2,batch=64),b,'cvar','delta',11,folder,sampler=sampler,extra_metadata={'world':'bootstrap'})
            restored,metadata=load(folder/s['checkpoint'])
            self.assertEqual(metadata['experiment_context']['world'],'bootstrap')
            torch.testing.assert_close(positions(m,b,self.c),positions(restored,b,self.c),atol=0,rtol=0)

    def test_observable_volatility_proxy_uses_only_past_returns(self):
        b=WorldSampler('heston',self.p,oracle=self.oracle)(self.c,128,seed=21)
        changed={k:v.clone() for k,v in b.items()};changed['spot'][:,12:]*=1.1
        a,z=observable_variance_proxy(b,self.c),observable_variance_proxy(changed,self.c)
        torch.testing.assert_close(a['volatility_path'][:,:12],z['volatility_path'][:,:12],atol=0,rtol=0)
        torch.testing.assert_close(a['premium'],b['premium'],atol=0,rtol=0)
        torch.testing.assert_close(a['payoff'],b['payoff'],atol=0,rtol=0)
        self.assertFalse(torch.equal(a['volatility_path'][:,12:],z['volatility_path'][:,12:]))

    def test_fixed_order_stress_normalizes_by_book(self):
        for book in (1,4,10,40):
            stress=execution_stresses(self.c,book)['spread_and_fixed_order']
            self.assertAlmostEqual(stress.fixed_cost*book,.00002)
            self.assertEqual(stress.half_spread,.0001)

    def test_observation_sensitivity_never_reads_latent_volatility(self):
        b=WorldSampler('heston',self.p,oracle=self.oracle)(self.c,64,seed=31)
        changed={k:v.clone() for k,v in b.items()}
        changed['sigma']*=3;changed['volatility_path']*=3
        a,z=observable_variance_proxy(b,self.c),observable_variance_proxy(changed,self.c)
        for key in ('sigma','volatility_path','delta','gamma'):
            torch.testing.assert_close(a[key],z[key],atol=0,rtol=0)

    def test_low_volatility_premiums_respect_bounds_and_put_call_parity(self):
        s=torch.linspace(.85,1.15,2001,dtype=torch.float32)
        for sigma in (.01,.074,.15):
            cp=bs(s,1.,20/252,sigma,True)[0]
            pp=bs(s,1.,20/252,sigma,False)[0]
            self.assertTrue(cp.ge((s-1).clamp_min(0)).all())
            self.assertTrue(pp.ge((1-s).clamp_min(0)).all())
            self.assertTrue(cp.le(s).all() and pp.le(1).all())
            torch.testing.assert_close(cp-pp,s-1,atol=2e-8,rtol=1e-6)
        self.assertGreater(float(bs(torch.tensor([.9]),1.,20/252,.074,True)[0]),0.)


if __name__=='__main__':unittest.main()
