"""Interference-aware power allocation and causal joint-control simulation."""
from dataclasses import dataclass

import numpy as np

from future_vr.belief import VisibilityBelief
from future_vr.controllers import select_top
from future_vr.experiment import _prediction
from future_vr.system import HardwareState, XLSystem
from major_revision.budget_search import forecasts


def effective_gains(cfg, gram):
    """Batched exact RZF gains from user-domain Gram matrices."""
    noise = XLSystem(cfg).noise_w
    inv = np.linalg.inv(gram + cfg.users*noise/cfg.tx_power_w*np.eye(cfg.users))
    norm2 = np.diagonal(inv.conj().swapaxes(-2,-1) @ gram @ inv, axis1=-2, axis2=-1).real
    norm = np.maximum(np.sqrt(np.maximum(norm2, 0.)), 1e-15)
    return np.abs((gram @ inv)/norm[...,None,:])**2


def rates(cfg, gains, powers):
    diag = np.diagonal(gains, axis1=-2, axis2=-1)
    desired = diag*powers
    cross = gains*(1-np.eye(cfg.users))
    interference = (cross @ powers[...,None])[...,0]
    sinr = desired/(interference + XLSystem(cfg).noise_w)
    return cfg.bandwidth_mhz*(1-cfg.pilot_fraction)*np.log2(1+sinr)


def required_powers(cfg, gains, targets):
    """Return p and physical feasibility before the total-power-cap check."""
    shape = gains.shape[:-2]
    gains = gains.reshape(-1, cfg.users, cfg.users)
    targets = np.broadcast_to(targets, (*shape,cfg.users)).reshape(-1,cfg.users)
    diag = np.diagonal(gains, axis1=-2, axis2=-1)
    gamma = np.expm1(np.log(2)*targets/(cfg.bandwidth_mhz*(1-cfg.pilot_fraction)))
    scale = np.divide(gamma, diag, out=np.zeros_like(gamma), where=diag>0)
    f = scale[...,None]*gains*(1-np.eye(cfg.users))
    ok = np.all((diag>0)|(gamma==0), axis=-1)
    ok &= np.max(np.abs(np.linalg.eigvals(f)), axis=-1) < 1-1e-10
    result = np.zeros_like(gamma)
    if np.any(ok):
        result[ok] = np.linalg.solve(np.eye(cfg.users)-f[ok],
            (scale[ok]*XLSystem(cfg).noise_w)[...,None])[...,0]
    ok &= np.isfinite(result).all(-1) & (result>=-1e-12).all(-1)
    return np.maximum(result,0).reshape(*shape,cfg.users), ok.reshape(shape)


def allocate(cfg, gains, requirements):
    """Shared allocator: meet full targets when feasible, else preserve service."""
    equal = np.full(gains.shape[:-1], cfg.tx_power_w/cfg.users)
    before = rates(cfg, gains, equal)
    targets = np.broadcast_to(requirements, before.shape)
    power, ok = required_powers(cfg, gains, targets*(1+1e-8))
    ok &= power.sum(-1) <= cfg.tx_power_w
    fallback_target = np.minimum(before, targets*(1+1e-8))
    fallback, fallback_ok = required_powers(cfg, gains, fallback_target)
    fallback = (1-1e-8)*fallback + 1e-8*equal
    fallback_ok &= fallback.sum(-1) <= cfg.tx_power_w+1e-10
    fallback = np.where(fallback_ok[...,None], fallback, equal)
    power = np.where(ok[...,None], power, fallback)
    after = rates(cfg, gains, power)
    service_ok = np.all(after >= np.minimum(before, targets)-1e-7, axis=-1)
    bounded = (power>=0).all(-1) & (power.sum(-1)<=cfg.tx_power_w+1e-10)
    numerical = ~(service_ok & bounded & np.isfinite(after).all(-1))
    power = np.where(numerical[...,None], equal, power)
    power = np.where(np.all(gains==0,axis=(-2,-1))[...,None],0.,power)
    after = rates(cfg, gains, power)
    return power, after, numerical


def slot_energy(cfg, active, waking, tx, switching, observed):
    """Return total, circuit/processing, PA, switching and pilot energy."""
    processing = (8*active*cfg.antennas_per_subarray*cfg.users**2 + 8*cfg.users**3/3)
    processing *= cfg.bandwidth_mhz*1e6/cfg.symbols_per_processing_block/(cfg.compute_efficiency_gflop_w*1e9)
    circuit = cfg.slot_s*(cfg.fixed_power_w + active*cfg.active_subarray_w
        + waking*cfg.waking_subarray_w
        + (cfg.subarrays-active-waking)*cfg.sleeping_subarray_w + processing)
    pa = cfg.slot_s*tx/cfg.pa_efficiency
    pilot = cfg.scan_energy_j_per_subarray*observed
    return circuit+pa+switching+pilot, circuit, pa, switching, pilot


@dataclass
class Candidates:
    masks: np.ndarray
    energy: np.ndarray
    outage: np.ndarray
    shortfall: np.ndarray
    features: np.ndarray
    prior: np.ndarray


class Environment:
    def __init__(self, cfg, trajectory, variable_power=True):
        if cfg.wake_delay_slots < 1 or cfg.horizon != 8:
            raise ValueError('Phase one supports positive wake delay and horizon eight')
        self.cfg, self.tr, self.variable_power = cfg, trajectory, variable_power
        self.hw = HardwareState(cfg)
        self.tracker = VisibilityBelief(cfg, trajectory.gain_db[0])
        self.tracker.channel_estimate = trajectory.channel[0].copy()
        self.age = np.zeros(cfg.subarrays)
        self.t, self.prepared, self.last_outage = 0, False, 0.
        self.records = []
        self.full_slot = float(slot_energy(cfg,cfg.subarrays,0,cfg.tx_power_w,0,cfg.subarrays)[0])

    def prepare(self):
        if self.prepared:
            return
        self.observed = self.hw.active.copy()
        self.age = np.where(self.observed,0,self.age+1)
        self.tracker.update(self.tr.gain_db[self.t], self.observed)
        antenna = np.repeat(self.observed,self.cfg.antennas_per_subarray)
        self.tracker.channel_estimate[:,antenna] = self.tr.channel[self.t][:,antenna]
        self.req = self.tr.requirements_mbps[self.t]
        self.prepared = True

    def candidates(self, price=8.):
        self.prepare()
        cfg = self.cfg
        score,_ = _prediction(cfg,self.tracker,self.hw,self.req)
        masks = np.array([select_top(score,m) for m in range(1,cfg.subarrays+1)] + [self.hw.command.copy()])
        count = len(masks)
        active = np.repeat(self.hw.active[None],count,axis=0)
        countdown = np.repeat(self.hw.countdown[None],count,axis=0)
        off = ~masks & (active | (countdown>0))
        on = masks & ~active & (countdown==0)
        switching = cfg.sleep_energy_j*off.sum(-1)+cfg.wake_energy_j*on.sum(-1)
        observed = active.sum(-1).copy()
        active[off],countdown[off],countdown[on] = False,0,cfg.wake_delay_slots
        active_sequence,waking_sequence,observed_sequence = [],[],[]
        for _ in range(cfg.horizon):
            active_sequence.append(active.copy())
            waking_sequence.append((countdown>0).sum(-1))
            observed_sequence.append(observed.copy())
            ready = countdown==1
            countdown[countdown>0] -= 1
            active[ready] = True
            observed = active.sum(-1)
        available = np.stack(active_sequence,axis=1)
        na, nw, no = available.sum(-1), np.array(waking_sequence).T, np.array(observed_sequence).T
        predicted = np.concatenate((self.tracker.channel_estimate[None], forecasts(cfg,self.tracker)[:-1]))
        blocks = predicted.reshape(cfg.horizon,cfg.users,cfg.subarrays,cfg.antennas_per_subarray).transpose(0,2,1,3)
        module_gram = blocks @ blocks.conj().swapaxes(-2,-1)
        gram = np.einsum('ahq,hqij->ahij',available,module_gram)
        gain = effective_gains(cfg,gram)
        power, achieved, _ = allocate(cfg,gain,self.req)
        switching_by_slot = np.zeros_like(na,dtype=float)
        switching_by_slot[:,0] = switching
        energy = slot_energy(cfg,na,nw,power.sum(-1),switching_by_slot,no)[0]
        outage = (achieved<self.req).mean(-1)
        shortfall = (np.maximum(self.req-achieved,0)/self.req).mean(-1)
        # First decision interval and its successor; hardware evolution is exact,
        # but future channels/demand persistence remain forecasts.
        eb = energy.reshape(count,2,4).mean(-1)/self.full_slot
        cb = outage.reshape(count,2,4).mean(-1)
        pb = power.sum(-1).reshape(count,2,4).mean(-1)/cfg.tx_power_w
        nb = na.reshape(count,2,4).mean(-1)/cfg.subarrays
        per_action = np.column_stack((eb,cb,pb,nb,shortfall.mean(-1),
            on.sum(-1)/cfg.subarrays,off.sum(-1)/cfg.subarrays,
            masks.sum(-1)/cfg.subarrays,np.all(masks==self.hw.command,axis=-1)))
        # Retain absolute link strength and complex user correlation per module.
        current = module_gram[0]
        diagonal = np.maximum(np.diagonal(current,axis1=-2,axis2=-1).real,1e-30)
        covariance = current/np.sqrt(diagonal[:,:,None]*diagonal[:,None,:])
        upper = np.triu_indices(cfg.users,1)
        context = np.concatenate((np.log10(diagonal/XLSystem(cfg).noise_w).ravel()/6,
            covariance[:,upper[0],upper[1]].real.ravel(), covariance[:,upper[0],upper[1]].imag.ravel(),
            self.hw.active,self.hw.command,self.hw.readiness()/cfg.horizon,
            np.minimum(self.age,32)/32,self.req/560,self.tracker.velocity,
            [self.last_outage,(cfg.episode_steps-self.t)/cfg.episode_steps]))
        features = np.concatenate((np.tile(context,(count,1)),per_action),axis=1).astype(np.float32)
        remaining = (cfg.episode_steps-self.t+3)//4
        tail_weight = sum(cfg.rl_gamma**n for n in range(1,remaining))
        prior = eb[:,0]+price*cb[:,0]+tail_weight*(eb[:,1]+price*cb[:,1])
        return Candidates(masks,energy.mean(-1),outage.mean(-1),shortfall.mean(-1),features,prior.astype(np.float32))

    def step(self, command):
        self.prepare()
        cfg = self.cfg
        active,switch,_ = self.hw.apply(command,1.)
        h = self.tracker.channel_estimate[:,np.repeat(active,cfg.antennas_per_subarray)]
        gain = effective_gains(cfg,h@h.conj().T)
        if self.variable_power:
            power,achieved,numerical = allocate(cfg,gain,self.req)
        else:
            power = np.full(cfg.users,cfg.tx_power_w/cfg.users)
            achieved,numerical = rates(cfg,gain,power),False
        components = slot_energy(cfg,int(active.sum()),int((self.hw.countdown>0).sum()),
            float(power.sum()),switch,int(self.observed.sum()))
        outage = float(np.mean(achieved<self.req))
        row = dict(slot=self.t,command_bits=sum(int(b)<<q for q,b in enumerate(command)),
            active_bits=sum(int(b)<<q for q,b in enumerate(active)),
            active_count=int(active.sum()),waking_count=int((self.hw.countdown>0).sum()),
            observed_count=int(self.observed.sum()),outage=outage,numerical_fallback=int(numerical),
            **dict(zip(('energy_j','circuit_j','pa_j','switching_j','pilot_j'),map(float,components))),
            **{f'power_{k}':float(v) for k,v in enumerate(power)},
            **{f'rate_{k}':float(v) for k,v in enumerate(achieved)})
        self.records.append(row)
        self.last_outage=outage
        self.hw.tick()
        self.t += 1
        self.prepared=False
        return components[0]/self.full_slot,outage

    def block(self, command):
        values = [self.step(command) for _ in range(min(4,self.cfg.episode_steps-self.t))]
        return np.mean(values,axis=0)

    def metrics(self):
        return {key:float(sum(r[key] for r in self.records)) for key in
                ('energy_j','circuit_j','pa_j','switching_j','pilot_j')} | {
            'outage':float(np.mean([r['outage'] for r in self.records])),
            'mean_active':float(np.mean([r['active_count'] for r in self.records])),
            'numerical_fallbacks':sum(r['numerical_fallback'] for r in self.records)}


def search_action(candidates, stride=1, target=.05):
    ids = sorted(set(range(0,len(candidates.masks)-1,stride)) | {len(candidates.masks)-2,len(candidates.masks)-1})
    feasible = [i for i in ids if candidates.outage[i]<=target]
    return min(feasible,key=lambda i:(candidates.energy[i],i)) if feasible else min(ids,
        key=lambda i:(candidates.shortfall[i],candidates.energy[i],i))
