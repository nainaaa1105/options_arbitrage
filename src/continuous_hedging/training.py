import copy
from dataclasses import asdict
import json
import pandas as pd
import torch
from .market import simulate
from .policies import ContinuousPolicy,BandPolicy,positions,FEATURES
from .accounting import pnl
from .objectives import risk
from .evaluation import metrics


def fit(config,fit_config,validation,objective,initialization,seed,output,sampler=None,family='mlp',extra_metadata=None):
    if config.fixed_cost:
        raise ValueError("Exact fixed trade-count costs are nondifferentiable; use proportional/spread costs for backprop training")
    torch.manual_seed(seed)
    generator=torch.Generator().manual_seed(seed+7000)
    if family not in ('mlp','band'):raise ValueError('Unknown continuous policy family')
    if family=='band' and initialization!='delta':raise ValueError('Band center uses a disclosed delta prior')
    model=ContinuousPolicy(initialization,config.max_position) if family=='mlp' else BandPolicy(config.max_position)
    eta=torch.nn.Parameter(torch.tensor(.1))
    groups=[{"params":model.parameters(),"lr":fit_config.learning_rate}]
    if objective=="cvar":groups.append({"params":[eta],"lr":.01})
    optimizer=torch.optim.Adam(groups)
    scheduler=torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=.95)
    best,best_state,best_epoch,best_eta=float('inf'),None,None,None
    records=[]
    first_state=copy.deepcopy(model.state_dict())
    for epoch in range(fit_config.epochs+1):
        grad_values=[]
        if epoch:
            for _ in range(fit_config.updates):
                batch=simulate(config,fit_config.batch,generator=generator) if sampler is None else sampler(config,fit_config.batch,generator=generator)
                result=pnl(batch,positions(model,batch,config),config)
                loss=risk(result["net_pnl"],config,objective,eta)
                if not torch.isfinite(loss):raise ArithmeticError("Non-finite continuous loss")
                optimizer.zero_grad()
                loss.backward()
                grad=torch.nn.utils.clip_grad_norm_(model.parameters(),10.)
                if not torch.isfinite(grad):raise ArithmeticError("Non-finite neural gradient")
                if objective=="cvar":eta.grad.clamp_(-10,10)
                optimizer.step()
                grad_values.append(float(grad))
            scheduler.step()
        with torch.no_grad():
            h=positions(model,validation,config)
            result=pnl(validation,h,config)
            # Evaluate the exact empirical risk, not the learned training eta.
            score=float(risk(result["net_pnl"],config,objective))
            row={"epoch":epoch,"validation_objective":score,**metrics(result,h,validation["delta"],config.entropic_aversion),
                 "gradient_norm":sum(grad_values)/max(len(grad_values),1),"eta_normalized":float(eta.detach()),
                 "learning_rate":optimizer.param_groups[0]["lr"]}
        records.append(row)
        if score<best:
            best,best_epoch,best_state,best_eta=score,epoch,copy.deepcopy(model.state_dict()),float(eta.detach())
        print(f"c={config.cost_rate:g} {objective}/{initialization} seed={seed} epoch={epoch}: risk={score:.4f}, delta MAE={row['delta_mae']:.4f}",flush=True)
    model.load_state_dict(best_state)
    update=sum(float((model.state_dict()[k]-v).abs().sum()) for k,v in first_state.items())
    metadata={"market":asdict(config),"fit":asdict(fit_config),"objective":objective,"initialization":initialization,'family':family,
              "seed":seed,"best_epoch":best_epoch,"validation_objective":best,"eta_normalized":best_eta,
              "selected_parameter_l1_change":update,"features":list(FEATURES),"hidden":model.hidden,"parameterization":model.parameterization,
              "selection":"Minimum validation objective, including epoch-zero diagnostic/fallback", "torch_version":str(torch.__version__)}
    if extra_metadata:metadata['experiment_context']=extra_metadata
    file=output/f"{objective}_{initialization}_{seed}.pt"
    torch.save({"state_dict":best_state,**metadata},file)
    pd.DataFrame(records).to_csv(output/f"{objective}_{initialization}_{seed}_curve.csv",index=False)
    return model,{**metadata,"checkpoint":file.name}


def load(file):
    saved=torch.load(file,map_location='cpu',weights_only=True)
    if saved['features']!=list(FEATURES):raise ValueError("Feature schema mismatch")
    parameterization=saved.get('parameterization')
    if parameterization is None:
        # Initial development checkpoints predate the explicit architecture
        # field. Resolve them ONLY from their hash-verified as-run source.
        from pathlib import Path
        import hashlib
        root=Path(file).parent.parent
        source=root/'source_snapshot/policies.py'
        plan=json.loads((root/'experiment_plan.json').read_text())
        if hashlib.sha256(source.read_bytes()).hexdigest()!=plan['source_sha256']['policies.py']:
            raise ValueError('Unversioned checkpoint source provenance mismatch')
        text=source.read_text()
        if 'magnitude=(prior+self.network' in text:parameterization='projected'
        elif 'torch.logit(prior.clamp' in text:parameterization='sigmoid'
        else:raise ValueError('Unknown unversioned continuous checkpoint architecture')
    model=BandPolicy(saved['market']['max_position'],saved['hidden']) if saved.get('family')=='band' else ContinuousPolicy(saved['initialization'],saved['market']['max_position'],saved['hidden'],parameterization)
    model.load_state_dict(saved['state_dict']);model.eval()
    return model,saved
