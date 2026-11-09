"""Known four-slot service signatures and a local risk-only projection."""
import numpy as np
from major_revision.action_value import action_size,observation_size
from major_revision.gram_control import representation_size


def service_signatures(cfg,features,slots=4):
    """Causal available masks [root, action, slot, subarray], not a Q quotient."""
    x=np.atleast_2d(features)
    q,k=cfg.subarrays,cfg.users
    start=q*k*k+3*q*k
    raw_active=x[:,start:start+q]
    raw_countdown=x[:,start+2*q:start+3*q]*cfg.horizon
    actions=x[:,representation_size(cfg):observation_size(cfg)].reshape(-1,q,action_size(cfg))
    masks=actions[:,:,:q]
    if not (np.all(np.isin(raw_active,[0.,1.])) and np.all(np.isin(masks,[0.,1.]))):
        raise ValueError('Serialized active/command fields must be binary')
    if not np.allclose(raw_countdown,np.rint(raw_countdown),rtol=0,atol=1e-5):
        raise ValueError('Countdown serialization must represent integer slots')
    active=np.broadcast_to(raw_active[:,None,:],masks.shape).astype(bool).copy()
    countdown=np.broadcast_to(np.rint(raw_countdown)[:,None,:],masks.shape).astype(int).copy()
    masks=masks.astype(bool)
    if np.any(countdown<0) or np.any(active&(countdown>0)):
        raise ValueError('Invalid hardware state')
    snapshots=[]
    for _ in range(slots):
        off=(~masks)&(active|(countdown>0))
        on=masks&(~active)&(countdown==0)
        active[off]=False
        countdown[off]=0
        countdown[on]=cfg.wake_delay_slots
        if cfg.wake_delay_slots==0:
            active[on]=True
        snapshots.append(active.copy())
        completing=countdown==1
        countdown[countdown>0]-=1
        active[completing]=True
    return np.stack(snapshots,axis=2)


def service_projection(signatures):
    same=np.all(signatures[:,:,None]==signatures[:,None,:],axis=(-1,-2))
    return same.astype(np.float64)/same.sum(-1,keepdims=True)


def project_risk(costs,projection):
    """Tie C4 risk only; retain each action's original energy and C16 tail."""
    result=np.asarray(costs,dtype=np.float64).copy()
    current=result[:,:,0,1].copy()
    projected=np.einsum('nij,nj->ni',projection,current)
    result[:,:,0,1]=projected
    result[:,:,1,1]+=projected-current
    return result
