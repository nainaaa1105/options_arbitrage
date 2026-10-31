"""Training-only daily-return calibration and causal synthetic GBM/Heston/bootstrap worlds."""
from dataclasses import dataclass,asdict
import math
import numpy as np
import pandas as pd
import torch
from scipy.interpolate import RegularGridInterpolator
from .market import bs,simulate


def daily_returns(file):
    d=pd.read_parquet(file).sort_values('timestamp').copy()
    d['date']=pd.to_datetime(d.timestamp).dt.date
    if d.date.duplicated().any() or not np.isfinite(d.close).all() or d.close.le(0).any():raise ValueError('Invalid daily spot source')
    d['return']=np.log(d.close/d.close.shift(1))
    return d.loc[d['return'].notna(),['date','return']].reset_index(drop=True)


def split_returns(d):
    v=pd.Timestamp('2026-07-01').date();t=pd.Timestamp('2026-09-16').date()
    return {'train':d.loc[d.date<v].copy(),'validation':d.loc[(d.date>=v)&(d.date<t)].copy(),'test':d.loc[d.date>=t].copy()}


def calibrate(training):
    if len(training)<100:raise ValueError('Need at least 100 training daily returns')
    r=training['return'].to_numpy(float)
    vol=pd.Series(r).rolling(20,min_periods=20).std(ddof=1).dropna().to_numpy()*math.sqrt(252)
    variance=vol**2;x,y=variance[:-1],variance[1:]
    slope=float(np.cov(x,y,ddof=0)[0,1]/max(np.var(x),1e-12))
    phi=float(np.clip(slope,.05,.999))
    kappa=float(np.clip(-math.log(phi)*252,.2,12.))
    theta=float(np.clip(variance.mean(),.0036,.36))
    intercept=float(y.mean()-slope*x.mean());innovation=y-intercept-slope*x
    xi=float(np.clip(np.std(innovation,ddof=1)*math.sqrt(252/theta),.05,1.5))
    rho=float(np.clip(np.corrcoef(r[-len(innovation):],innovation)[0,1],-.9,.9))
    if not math.isfinite(rho):rho=0.
    center=float(np.log(np.exp(r).mean()))
    return {'sigma_low':float(np.clip(np.quantile(vol,.1),.06,.6)),
            'sigma_high':float(np.clip(np.quantile(vol,.9),.06,.6)),
            'kappa':kappa,'theta':theta,'xi':xi,'rho':rho,'bootstrap_log_center':center,
            'training_rows':len(training),'training_start':str(training.date.min()),'training_end':str(training.date.max()),
            'method':'20-day realized-variance AR(1) moment proxy, parameter caps predeclared; not identified option-surface Heston calibration',
            'risk_premium_assumption':'Zero volatility risk premium when using proxy parameters in Fourier premiums'}


def heston_cf(u,t,v,kappa,theta,xi,rho):
    u=np.asarray(u,dtype=complex)
    if xi<1e-5:
        integrated=theta*t+(v-theta)*(-np.expm1(-kappa*t))/kappa
        return np.exp(-.5*(u*u+1j*u)*integrated)
    b=kappa-rho*xi*1j*u;d=np.sqrt(b*b+xi*xi*(u*u+1j*u));g=(b-d)/(b+d);e=np.exp(-d*t)
    c=kappa*theta/(xi*xi)*((b-d)*t-2*np.log((1-g*e)/(1-g)))
    return np.exp(c+(b-d)/(xi*xi)*(1-e)/(1-g*e)*v)


class HestonPremium:
    """Only initial premiums require Fourier pricing; terminal liability is exact payoff."""
    def __init__(self,parameters,maturity,sigma_low,sigma_high):
        self.p,self.t=parameters,maturity
        self.x=np.linspace(-.12,.12,401);self.v=np.linspace(sigma_low*.999,sigma_high*1.001,65)**2
        if self.v[-1]-self.v[0]<1e-10:self.v=np.linspace(max(self.v[0]*.9,1e-6),self.v[0]*1.1,65)
        z,w=np.polynomial.legendre.leggauss(64);bounds=(0,1,5,20,100,500,2000)
        self.u=np.concatenate([(a+b)/2+(b-a)/2*z for a,b in zip(bounds[:-1],bounds[1:])])
        self.w=np.concatenate([(b-a)/2*w for a,b in zip(bounds[:-1],bounds[1:])])
        phi=self._cf(self.u[None,:]-.5j,self.v[:,None])
        integral=np.real((phi*(self.w/(self.u*self.u+.25))[None,:])@np.exp(1j*self.u[:,None]*self.x[None,:]))
        s=np.exp(self.x)[None,:];price=s-np.sqrt(s)*integral/np.pi
        lower=np.maximum(s-1,0)
        if np.min(price-lower)<-2e-5 or np.max(price-s)>2e-5:raise ArithmeticError('Heston premium Fourier bounds failed')
        self.grid=np.minimum(np.maximum(price,lower),s).T
        self.interpolator=RegularGridInterpolator((self.x,self.v),self.grid,bounds_error=True)

    def _cf(self,u,v):return heston_cf(u,self.t,v,*(self.p[k] for k in ('kappa','theta','xi','rho')))

    def marks(self,spot,variance,call):
        s,v,call=np.broadcast_arrays(spot,variance,call)
        cp=self.interpolator(np.column_stack([np.log(s).ravel(),v.ravel()])).reshape(s.shape)
        return np.maximum(np.where(call,cp,cp-s+1),np.maximum(np.where(call,s-1,1-s),0.))

    def direct(self,spot,variance,call):
        s,v,call=np.broadcast_arrays(spot,variance,call)
        integrand=self._cf(self.u[None,:]-.5j,v.ravel()[:,None])*np.exp(1j*np.log(s).ravel()[:,None]*self.u[None,:])*(self.w/(self.u*self.u+.25))[None,:]
        cp=s-np.sqrt(s)*integrand.real.sum(1).reshape(s.shape)/np.pi
        return np.where(call,cp,cp-s+1)

    def save(self,file):np.savez_compressed(file,x=self.x,variance=self.v,call_premium=self.grid,maturity=self.t)


class WorldSampler:
    def __init__(self,name,parameters,returns=None,block=3,oracle=None,recenter=True):
        if name not in ('gbm','heston','bootstrap'):raise ValueError('Unknown world')
        self.name,self.p,self.block,self.oracle,self.recenter=name,parameters,block,oracle,recenter
        self.returns=None if returns is None else torch.tensor(np.asarray(returns,dtype=float),dtype=torch.float32)
        if name=='bootstrap' and (self.returns is None or len(self.returns)<block or block<1):raise ValueError('Insufficient bootstrap library')

    def __call__(self,c,count,seed=None,generator=None):
        if count<1:raise ValueError('Positive path count required')
        if generator is not None and seed is not None:raise ValueError('Seed or continuing generator, not both')
        g=generator or torch.Generator().manual_seed(seed if seed is not None else 0)
        if self.name=='gbm':return simulate(c,count,generator=g)
        spot0=c.strike*(.9+.2*torch.rand(count,generator=g));sigma=c.sigma_low+(c.sigma_high-c.sigma_low)*torch.rand(count,generator=g)
        call=torch.rand(count,generator=g)>.5
        if self.name=='heston':
            substeps=max(1,math.ceil(320/c.steps));dt=c.maturity/(c.steps*substeps)
            s=spot0.clone();v=sigma.square();spots=[s.clone()];variances=[v.clone()]
            for i in range(c.steps):
                for _ in range(substeps):
                    z=torch.randn(count,2,generator=g);vp=v.clamp_min(0)
                    s=s*torch.exp(-.5*vp*dt+torch.sqrt(vp*dt)*z[:,0])
                    zv=self.p['rho']*z[:,0]+math.sqrt(1-self.p['rho']**2)*z[:,1]
                    v=v+self.p['kappa']*(self.p['theta']-vp)*dt+self.p['xi']*torch.sqrt(vp*dt)*zv
                spots.append(s.clone());variances.append(v.clamp_min(1e-8))
            spot=torch.stack(spots,1);volatility=torch.stack(variances,1).sqrt()
        else:
            # Each step is one daily return, with a strictly past 20-return RV.
            if c.steps!=20 or abs(c.maturity-20/252)>1e-10:raise ValueError('Daily bootstrap requires 20 daily steps / 20 trading days')
            needed=c.steps+20;blocks=math.ceil(needed/self.block)
            starts=torch.randint(len(self.returns)-self.block+1,(count,blocks),generator=g)
            indices=(starts[:,:,None]+torch.arange(self.block)[None,None,:]).flatten(1)[:,:needed]
            history=self.returns[indices]
            if self.recenter:history=history-self.p['bootstrap_log_center']
            forward=history[:,20:]
            spot=spot0[:,None]*torch.exp(torch.cat([torch.zeros(count,1),forward.cumsum(1)],1))
            volatility=torch.stack([history[:,i:i+20].std(1)*math.sqrt(252) for i in range(c.steps+1)],1).clamp_min(.01)
            sigma=volatility[:,0]
        maturity=torch.linspace(c.maturity,0,c.steps+1)[None,:].expand(count,-1)
        price,delta,gamma=bs(spot,c.strike,maturity,volatility,call[:,None])
        premium=price[:,0]
        if self.name=='heston':
            if self.oracle is None:raise ValueError('Heston needs a verified initial-premium oracle')
            premium=torch.tensor(self.oracle.marks((spot0/c.strike).numpy(),sigma.square().numpy(),call.numpy())*c.strike,dtype=spot.dtype)
        return {'spot':spot,'sigma':sigma,'call':call,'maturity':maturity,'premium':premium,'payoff':price[:,-1],
                'delta':delta[:,:-1],'gamma':gamma[:,:-1],'volatility_path':volatility}
