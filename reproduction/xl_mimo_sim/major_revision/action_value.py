"""Action-conditioned, physically anchored Double DQN; unchanged real simulator."""
import copy
from dataclasses import replace
import numpy as np
import torch
from torch import nn
from future_vr.controllers import ConstrainedDQN,Transition,QNetwork,select_top
from major_revision.full_domain import FullDomainRollout
from major_revision.gram_control import (subarray_grams,rates_from_gram,
    prefix_forecast_rates,gram_belief_features,representation_size)

PRICE=2.
BLOCK=4


def slot_energy(cfg,active,waking,observed,transition):
    """Vectorized original analytic accounting, not measured electrical power."""
    antennas=active*cfg.antennas_per_subarray
    flops=8.*antennas*cfg.users**2+(8./3.)*cfg.users**3
    processing=flops*(cfg.bandwidth_mhz*1e6/cfg.symbols_per_processing_block)/(cfg.compute_efficiency_gflop_w*1e9)
    power=(cfg.fixed_power_w+cfg.tx_power_w/cfg.pa_efficiency+active*cfg.active_subarray_w+
           waking*cfg.waking_subarray_w+(cfg.subarrays-active-waking)*cfg.sleeping_subarray_w+processing)
    return power*cfg.slot_s+transition+cfg.scan_energy_j_per_subarray*observed


def full_slot_energy(cfg):
    return float(slot_energy(cfg,cfg.subarrays,0,cfg.subarrays,0.))


def action_size(cfg):
    return cfg.subarrays+cfg.horizon*(cfg.users+3)+3


def observation_size(cfg):
    return representation_size(cfg)+cfg.subarrays*action_size(cfg)


def hold_predictions(cfg,tracker,hardware,score):
    """All prefix hold commands, physical slots t,...,t+H-1, causal CSI only."""
    masks=np.array([select_top(score,n) for n in range(1,cfg.subarrays+1)])
    active=np.broadcast_to(hardware.active,masks.shape).copy()
    countdown=np.broadcast_to(hardware.countdown,masks.shape).copy()
    grams=subarray_grams(cfg,tracker.channel_estimate)
    measured=np.diagonal(grams,axis1=-2,axis2=-1).real/cfg.antennas_per_subarray
    future,_,_=tracker.predict(cfg.horizon)
    gains=np.concatenate((tracker.gain_db[None],future[:-1]))
    scale=np.sqrt(10.**(gains/10.)/np.maximum(measured[None],1e-20))
    forecasts=grams[None]*scale[..., :,None]*scale[...,None,:]
    energies,available,working,recovering=[],[],[],[]
    first_on=first_off=None
    for h in range(cfg.horizon):
        observed=active.sum(axis=1)
        off=(~masks)&(active|(countdown>0))
        on=masks&(~active)&(countdown==0)
        if h==0:
            first_on,first_off=on.sum(axis=1),off.sum(axis=1)
        active[off]=False
        countdown[off]=0
        countdown[on]=cfg.wake_delay_slots
        if cfg.wake_delay_slots==0:
            active[on]=True
        na,nw=active.sum(axis=1),(countdown>0).sum(axis=1)
        transition=cfg.sleep_energy_j*off.sum(axis=1)+cfg.wake_energy_j*on.sum(axis=1)
        energies.append(slot_energy(cfg,na,nw,observed,transition))
        available.append(active.copy())
        working.append(na)
        recovering.append(nw)
        completing=countdown==1
        countdown[countdown>0]-=1
        active[completing]=True
    available=np.stack(available,axis=1)
    working=np.stack(working,axis=1)
    gram_sums=np.einsum('ahq,hqkl->ahkl',available,forecasts)
    rates=rates_from_gram(cfg,gram_sums,working*cfg.antennas_per_subarray)
    return dict(masks=masks,rates=rates,energy=np.stack(energies,axis=1),
                active=working,waking=np.stack(recovering,axis=1),on=first_on,off=first_off)


def observation(run):
    _,score=run.base_observation()
    cfg=run.cfg
    prediction=hold_predictions(cfg,run.tracker,run.hw,score)
    ratios=np.clip(prediction['rates']/run.req[None,None,:],0,4)
    candidates=np.concatenate((prediction['masks'].astype(float),
        prediction['energy']/full_slot_energy(cfg),ratios.reshape(cfg.subarrays,-1),
        prediction['active']/cfg.subarrays,prediction['waking']/cfg.subarrays,
        np.arange(1,cfg.subarrays+1)[:,None]/cfg.subarrays,
        prediction['on'][:,None]/cfg.subarrays,prediction['off'][:,None]/cfg.subarrays),axis=1)
    common=gram_belief_features(cfg,run.tracker,run.hw,run.req,run.previous_req,run.age,run.last_outage)
    state=np.concatenate((common,candidates.ravel())).astype(np.float32)
    assert candidates.shape==(cfg.subarrays,action_size(cfg))
    return state,score


def prior_values(cfg,state):
    rows=state[...,representation_size(cfg):].reshape(*state.shape[:-1],cfg.subarrays,action_size(cfg))
    energies=rows[...,cfg.subarrays:cfg.subarrays+cfg.horizon]
    start=cfg.subarrays+cfg.horizon
    ratios=rows[...,start:start+cfg.horizon*cfg.users].reshape(*rows.shape[:-1],cfg.horizon,cfg.users)
    return -energies[...,:BLOCK].mean(axis=-1)-PRICE*(ratios[...,:BLOCK,:]<1.).mean(axis=(-1,-2))


class ActionValueNetwork(nn.Module):
    def __init__(self,cfg,architecture='shared'):
        super().__init__()
        self.cfg,self.architecture=cfg,architecture
        common,action=representation_size(cfg),action_size(cfg)
        if architecture=='shared':
            self.common=nn.Sequential(nn.Linear(common,64),nn.LayerNorm(64),nn.SiLU())
            self.action=nn.Sequential(nn.Linear(action,32),nn.LayerNorm(32),nn.SiLU())
            self.head=nn.Sequential(nn.Linear(96,64),nn.SiLU(),nn.Linear(64,1))
            final=self.head[-1]
        elif architecture=='flat':
            self.flat=QNetwork(observation_size(cfg),cfg.subarrays,64)
            final=self.flat.layers[-1]
        else:
            raise ValueError('Unknown architecture')
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(self,x):
        cfg=self.cfg
        rows=x[:,representation_size(cfg):].reshape(-1,cfg.subarrays,action_size(cfg))
        energy=rows[:,:,cfg.subarrays:cfg.subarrays+BLOCK].mean(-1)
        start=cfg.subarrays+cfg.horizon
        rates=rows[:,:,start:start+cfg.horizon*cfg.users].reshape(-1,cfg.subarrays,cfg.horizon,cfg.users)
        prior=-energy-PRICE*(rates[:,:,:BLOCK,:]<1.).float().mean((-1,-2))
        if self.architecture=='flat':
            residual=self.flat(x)
        else:
            common=self.common(x[:,:representation_size(cfg)])[:,None].expand(-1,cfg.subarrays,-1)
            residual=self.head(torch.cat((common,self.action(rows)),dim=-1)).squeeze(-1)
        return prior+residual


class ActionValueDQN(ConstrainedDQN):
    def __init__(self,cfg,seed,architecture='shared'):
        super().__init__(cfg,observation_size(cfg),seed)
        self.min_count,self.actions=1,cfg.subarrays
        torch.manual_seed(seed)
        self.online=ActionValueNetwork(cfg,architecture)
        self.target=copy.deepcopy(self.online)
        self.optimizer=torch.optim.AdamW(self.online.parameters(),lr=cfg.rl_lr,weight_decay=1e-5)
        self.dual=PRICE


def baseline_command(run,mode,margin=1.,minimum=1):
    _,score=run.base_observation()
    cfg=run.cfg
    if mode=='full':
        return np.ones(cfg.subarrays,bool)
    if mode=='prior':
        state,score=observation(run)
        return select_top(score,int(np.argmax(prior_values(cfg,state)))+1)
    if mode=='prs':
        rates=prefix_forecast_rates(cfg,run.tracker,run.hw,score)
        energy=np.arange(1,cfg.subarrays+1,dtype=float)
    elif mode=='cost':
        prediction=hold_predictions(cfg,run.tracker,run.hw,score)
        rates,energy=prediction['rates'],prediction['energy'].sum(axis=1)
    else:
        raise ValueError('Unknown baseline')
    req=run.req[None,None,:]*margin
    outage=np.mean(rates<req,axis=(1,2))
    shortfall=np.mean(np.maximum(req-rates,0)/req,axis=(1,2))
    def key(i):
        feasible=outage[i]<=cfg.outage_target
        return (not feasible,energy[i] if feasible else shortfall[i],energy[i],i)
    action=min(range(minimum-1,cfg.subarrays),key=key)
    return select_top(score,action+1)


def action_episode(cfg,tr,model=None,training=False,epsilon=0.,mode='prs',margin=1.,minimum=1):
    if training and (model is None or len(tr.channel)<=cfg.episode_steps):
        raise ValueError('Training needs a real successor observation')
    run=FullDomainRollout(replace(cfg,episode_steps=cfg.episode_steps+1) if training else cfg,tr)
    pending=None
    losses=[]
    def remember(successor):
        old,action,start,energy=pending
        model.remember(Transition(old,action,(run.energy-energy)/((run.t-start)*full_slot_energy(cfg)),
            float(np.mean(run.outages[start:run.t])),successor,False))
        losses.append(model.learn())
    while run.t<cfg.episode_steps:
        if run.t%BLOCK:
            run.step()
            continue
        if model is None:
            command=baseline_command(run,mode,margin,minimum)
        else:
            state,score=observation(run)
            if training and pending is not None:
                remember(state)
            action,count=model.choose(state,epsilon if training else 0.)
            command=select_top(score,count)
            pending=(state,action,run.t,run.energy)
        run.step(command)
    if training:
        successor,_=observation(run)
        remember(successor)
    result=run.metrics()
    result.update(method='Action-value DQN' if model is not None else mode,
                  below_four_fraction=float(np.mean(np.asarray(run.counts)<4)),
                  td_loss=float(np.mean(losses)) if losses else 0.)
    return result
