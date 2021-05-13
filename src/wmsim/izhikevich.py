import numpy as np
import time
from . import ModelError, BaseNN, CommonNN


class IzhikevichNN(CommonNN):
    def __init__(self, **kwargs):
        super().__init__("Izhikevich Network", **kwargs)
        for param in ["a", "b", "c", "d", "v0", "u0"]:
            self.neuron_params.append(param)

        self.v = None
        self.u = None
        self.a = None
        self.b = None
        self.c = None
        self.d = None

        self.spike_threshold = kwargs.get("spike_threshold", 30)
        self._autoevolve = {
            "preset_1": self._autoevolve_preset_1,
            "preset_2": self._autoevolve_preset_2,
            "preset_3": self._autoevolve_preset_3,
            "preset_4": self._autoevolve_preset_4
        }[kwargs.get("autoevolve_formula", "preset_1")]

    def init(self):
        super().init()
        self.v = np.concatenate([grp[1]["v0"] for grp in self.neuron_groups.values()], axis=0)
        self.u = np.concatenate([grp[1]["u0"] for grp in self.neuron_groups.values()], axis=0)
        self.a = np.concatenate([grp[1]["a"] for grp in self.neuron_groups.values()], axis=0)
        self.b = np.concatenate([grp[1]["b"] for grp in self.neuron_groups.values()], axis=0)
        self.c = np.concatenate([grp[1]["c"] for grp in self.neuron_groups.values()], axis=0)
        self.d = np.concatenate([grp[1]["d"] for grp in self.neuron_groups.values()], axis=0)

    @property
    def network_state(self):
        state = super().network_state
        state.update({
            "simulation": {
                "v": np.array(self.v),
                "u": np.array(self.u),
            }
        })

        return state

    def load_state(self, state):
        super().load_state(state)
        self.v = state["simulation"]["v"]
        self.u = state["simulation"]["u"]

        self.a = np.concatenate([grp[1]["a"] for grp in self.neuron_groups.values()], axis=0)
        self.b = np.concatenate([grp[1]["b"] for grp in self.neuron_groups.values()], axis=0)
        self.c = np.concatenate([grp[1]["c"] for grp in self.neuron_groups.values()], axis=0)
        self.d = np.concatenate([grp[1]["d"] for grp in self.neuron_groups.values()], axis=0)

    def _autoevolve_preset_1(self, I):
        self.v += self.timestep * (0.04 * np.square(self.v) + 5 * self.v + 140 - self.u + I)
        self.u += self.timestep * (self.a * (self.b * self.v - self.u))

    def _autoevolve_preset_2(self, I):
        self.v += self.timestep * (0.04 * np.square(self.v) + 4.1 * self.v + 108 - self.u + I)
        self.u += self.timestep * (self.a * (self.b * self.v - self.u))

    def _autoevolve_preset_3(self, I):
        self.v += self.timestep * (0.04 * np.square(self.v) + 5 * self.v + 140 - self.u + I)
        self.u += self.timestep * (self.a * (self.b * (self.v + 65)))

    def _autoevolve_preset_4(self, I):
        self.v += 0.5 * self.timestep * (0.04 * np.square(self.v) + 5 * self.v + 140 - self.u + I)
        self.v += 0.5 * self.timestep * (0.04 * np.square(self.v) + 5 * self.v + 140 - self.u + I)
        self.u += self.timestep * (self.a * (self.b * self.v - self.u))

    def evolve_for(self, T, *, I=None, save_v=False, save_u=False, save_I=False, save_STDP=False):
        Nsteps = int(T/self.timestep)
        if I is None:
            I = np.zeros((Nsteps, self.N), dtype="float64")
        else:
            if I.shape[0] != Nsteps or I.shape[1] != self.N:
                raise ModelError("Dimensions for input (I) must be {}x{}.".format(Nsteps, self.N))

        # Convert delay to discrete steps
        ACD = np.array(np.ceil(self.ACD/self.timestep), dtype="int")
        max_delay = np.max(ACD)

        # Extend current input to include spikes arriving after simulation ends
        I = np.concatenate((I, np.zeros((max_delay, self.N), dtype="float64")), axis=0)

        # at time t (step st): neuron i --> delay j --> neuron k
        #   raster[st] = [i1,i2,i3...] (negative indices refer to previous period)
        raster = list([list() for i in range(Nsteps+max_delay)])

        # Apply activity cached from previous period
        I[:self.next_I.shape[0],:] += self.next_I
        for i in range(len(self.raster_cache)):
            raster[-len(self.raster_cache)+i] = self.raster_cache[i]

        if self.use_STDP:
            STDPp = np.zeros((Nsteps+max_delay+1, self.N), dtype="float64")
            STDPn = np.zeros((Nsteps+max_delay+1, self.N), dtype="float64")

            length = self.next_STDPp.shape[0]
            if length > 0:
                STDPp[-(length-1):,:] = self.next_STDPp[:-1,:]; STDPp[0,:] = self.next_STDPp[-1,:]
                STDPn[-(length-1):,:] = self.next_STDPn[:-1,:]; STDPn[0,:] = self.next_STDPn[-1,:]

        # All other recorded data are accessible with kwarg flags
        other = dict()
        if save_v:
            other["save_v"] = np.empty((Nsteps, self.N), dtype="float64")
        if save_u:
            other["save_u"] = np.empty((Nsteps, self.N), dtype="float64")

        # Arithmetic / lookup functions
        def post(source):
            return np.where(ACD[source,:]>0)[0]

        def pre(target):
            return np.where(ACD[:,target]>0)[0]

        def array_slice(array, start, end):
            if start < 0 and end >= 0:
                return array[start:] + array[:end]
            return array[start:end]

        start_time = time.time()
        for st in range(0, Nsteps):
            t = self.sim_time + st * self.timestep

            # Record spike activity and apply synapse into the future
            spikes = np.where(self.v>=self.spike_threshold)[0]
            raster[st].extend(spikes)
            for spike in spikes:
                targets = post(spike); delays = ACD[spike,targets]
                I[st+delays,targets] += self.S[spike,targets]

            # Reset dynamics
            self.v[spikes] = self.c[spikes]
            self.u[spikes] += self.d[spikes]

            # STDP protocol for each spike (pre_i -->) i --> k:
            #   Set STDP of i to max potential
            #   Apply positive STDP to pre_i --> i
            #   Apply negative STDP to i --> k
            #   Decay all STDP values
            if self.use_STDP:
                STDPp[st,spikes] = self.Ap; STDPn[st,spikes] = self.An;

                for spike in spikes:
                    sources = pre(spike); delays = ACD[sources,spike]
                    self.dS[sources,spike] += STDPp[st-delays,sources]

                dst = max_delay
                for prev_spikes in array_slice(raster, st-max_delay, st):
                    for spike in prev_spikes:
                        on_time_targets = np.where(ACD[spike,:]==dst)[0]
                        self.dS[spike,on_time_targets] += STDPn[st,on_time_targets]
                    dst -= 1

                STDPp[st+1,:] = STDPp[st,:] * self.p_decay
                STDPn[st+1,:] = STDPn[st,:] * self.n_decay

            # Update dynamics
            self._autoevolve(I[st,:])

            if save_v:
                other["save_v"][st,:] = np.array(self.v)
            if save_u:
                other["save_u"][st,:] = np.array(self.u)

        # Post-evolution STDP update:
        #   Apply dS to S (S = S + dS + 0.01)
        #   Decay dS values
        if self.use_STDP:
            self.next_STDPp = np.array(STDPp[Nsteps-max_delay:Nsteps+1,:], dtype="float64")
            self.next_STDPn = np.array(STDPn[Nsteps-max_delay:Nsteps+1,:], dtype="float64")

            idx = 0; labels = iter(self.neuron_groups.keys())
            while idx < self.N:
                label = next(labels); size = self.neuron_groups[label][0]
                if self.apply_STDP[label] is True:
                    for i in range(idx, idx+size):
                        targets = post(i)
                        self.S[i,targets] += 0.01 + self.dS[i,targets]
                    np.clip(self.S[idx:idx+size], self.S_min, self.S_max, self.S[idx:idx+size])
                idx += size
            self.dS *= self.dS_decay

        # Save to cache for use in next period
        self.next_I = np.array(I[Nsteps:,:], dtype="float64")
        self.raster_cache = raster[Nsteps-max_delay:Nsteps]

        end_time = time.time()
        print("Simulated {} ms ({} steps) in {} seconds.".format(T, Nsteps,
            round(end_time-start_time, 2)))

        self.sim_time += Nsteps * self.timestep

        if save_I:
            other["save_I"] = np.array(I)
        if save_STDP:
            other["save_STDP"] = (np.array(STDPp), np.array(STDPn))

        return raster[:Nsteps], other
