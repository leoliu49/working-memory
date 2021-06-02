import heapq
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

        # Cached values to apply to next evolution period
        self.next_I = None
        self.next_STDPp = None; self.next_STDPn = None
        self.raster_cache = None

        self.spike_threshold = kwargs.get("spike_threshold", 30)
        self._autoevolve = {
            "preset_1": self._autoevolve_preset_1,
            "preset_2": self._autoevolve_preset_2,
            "preset_3": self._autoevolve_preset_3,
            "preset_4": self._autoevolve_preset_4
        }[kwargs.get("autoevolve_formula", "preset_1")]

    def init(self):
        super().init()
        self.v = np.concatenate([grp[2]["v0"] for grp in self.neuron_groups.values()], axis=0)
        self.u = np.concatenate([grp[2]["u0"] for grp in self.neuron_groups.values()], axis=0)
        self.a = np.concatenate([grp[2]["a"] for grp in self.neuron_groups.values()], axis=0)
        self.b = np.concatenate([grp[2]["b"] for grp in self.neuron_groups.values()], axis=0)
        self.c = np.concatenate([grp[2]["c"] for grp in self.neuron_groups.values()], axis=0)
        self.d = np.concatenate([grp[2]["d"] for grp in self.neuron_groups.values()], axis=0)

        self.next_I = np.zeros((0, self.N), dtype="float64")
        self.next_STDPp = np.zeros((0, self.N), dtype="float64")
        self.next_STDPn = np.zeros((0, self.N), dtype="float64")
        self.raster_cache = list()

    @property
    def network_state(self):
        state = super().network_state
        state.update({
            "simulation": {
                "v": np.array(self.v),
                "u": np.array(self.u),
            }
        })

        state["network"].update({
            "cache": {
                "next_I": np.array(self.next_I),
                "next_STDPp": np.array(self.next_STDPp),
                "next_STDPn": np.array(self.next_STDPn),
                "raster_cache": list(self.raster_cache)
            }
        })

        return state

    def load_state(self, state):
        super().load_state(state)
        self.v = state["simulation"]["v"]
        self.u = state["simulation"]["u"]

        self.a = np.concatenate([grp[2]["a"] for grp in self.neuron_groups.values()], axis=0)
        self.b = np.concatenate([grp[2]["b"] for grp in self.neuron_groups.values()], axis=0)
        self.c = np.concatenate([grp[2]["c"] for grp in self.neuron_groups.values()], axis=0)
        self.d = np.concatenate([grp[2]["d"] for grp in self.neuron_groups.values()], axis=0)

        self.next_I = state["network"]["cache"]["next_I"]
        self.next_STDPp = state["network"]["cache"]["next_STDPp"]
        self.next_STDPn = state["network"]["cache"]["next_STDPn"]
        self.raster_cache = state["network"]["cache"]["raster_cache"]

    def _autoevolve_preset_1(self, I):
        self.v += self.timestep * (0.04 * np.square(self.v) + 5 * self.v + 140 - self.u) + I
        self.u += self.timestep * (self.a * (self.b * self.v - self.u))

    def _autoevolve_preset_2(self, I):
        self.v += self.timestep * (0.04 * np.square(self.v) + 4.1 * self.v + 108 - self.u) + I
        self.u += self.timestep * (self.a * (self.b * self.v - self.u))

    def _autoevolve_preset_3(self, I):
        self.v += self.timestep * (0.04 * np.square(self.v) + 5 * self.v + 140 - self.u) + I
        self.u += self.timestep * (self.a * (self.b * (self.v + 65)))

    def _autoevolve_preset_4(self, I):
        self.v += 0.5 * (self.timestep * (0.04 * np.square(self.v) + 5 * self.v + 140 - self.u) + I)
        self.v += 0.5 * (self.timestep * (0.04 * np.square(self.v) + 5 * self.v + 140 - self.u) + I)
        self.u += self.timestep * (self.a * (self.b * self.v - self.u))

    def evolve_for(self, T, *, I=None):
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

        # At time t (step st): neuron i --> delay j --> neuron k
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

        # All other recorded data are accessible in dictionary
        other = dict()
        other["save_v"] = np.empty((Nsteps, self.N), dtype="float64")
        other["save_u"] = np.empty((Nsteps, self.N), dtype="float64")
        other["firings"] = list()

        # Arithmetic / lookup functions
        def post(source):
            return np.where(ACD[source,:]>0)[0]

        def pre(target):
            return np.where(ACD[:,target]>0)[0]

        def array_slice(array, start, end):
            if start < 0 and end >= 0:
                return array[start:] + array[:end]
            return array[start:end]

        def STDP_filter(array):
            return array[np.where(np.isin(array, self.apply_STDP, assume_unique=True))[0]]

        start_time = time.time()
        for st in range(0, Nsteps):
            t = self.sim_time + st * self.timestep

            # Record spike activity
            spikes = np.where(self.v>=self.spike_threshold)[0]
            if len(spikes) > 0:
                raster[st] = spikes.tolist()
                other["firings"].append(np.vstack((
                    np.full(spikes.shape, t, dtype="int"),
                    spikes
                )).T)

            # Apply synapses into the future
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
                    sources = STDP_filter(pre(spike)); delays = ACD[sources,spike]
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

            other["save_v"][st,:] = np.array(self.v)
            other["save_u"][st,:] = np.array(self.u)

        # Post-evolution STDP update:
        #   Apply dS to S (S = S + dS + 0.01)
        #   Decay dS values
        if self.use_STDP:
            self.next_STDPp = np.array(STDPp[Nsteps-max_delay:Nsteps+1,:], dtype="float64")
            self.next_STDPn = np.array(STDPn[Nsteps-max_delay:Nsteps+1,:], dtype="float64")

            idx = 0; labels = iter(self.neuron_groups.keys())
            while idx < self.N:
                label = next(labels)
                size = self.neuron_groups[label][1] - self.neuron_groups[label][0]
                for i in range(idx, idx+size):
                    if i in self.apply_STDP:
                        targets = post(i)
                        adj = 0.01 + self.dS[i,targets]
                        self.S[i,targets] = np.clip(self.S[i,targets]+adj, self.S_min, self.S_max)
                idx += size
            self.dS *= self.dS_decay

        # Save to cache for use in next period
        self.next_I = np.array(I[Nsteps:,:], dtype="float64")
        self.raster_cache = raster[Nsteps-max_delay:Nsteps]

        self.sim_time += Nsteps * self.timestep

        end_time = time.time()
        other["meta"] = {
            "sim_time": T,
            "runtime": round(end_time-start_time, 2)
        }

        if self.use_STDP:
            other["save_STDP"] = (np.array(STDPp), np.array(STDPn))
        other["save_I"] = np.array(I)

        other["firings"] = np.concatenate(other["firings"])

        return raster[:Nsteps], other

    def stimulate_for(self, T, *, I=None):
        Nsteps = int(T/self.timestep)

        # Convert delay to discrete steps
        ACD = np.array(np.ceil(self.ACD/self.timestep), dtype="int")
        max_delay = np.max(ACD)

        # Extend current input to include spikes arriving after simulation ends
        I = np.concatenate((I, np.zeros((max_delay, self.N), dtype="float64")), axis=0)

        # Empty initial conditions
        firings = list()
        real_v = self.v
        self.v = np.concatenate([grp[2]["v0"] for grp in self.neuron_groups.values()], axis=0)
        real_u = self.u
        self.u = np.concatenate([grp[2]["u0"] for grp in self.neuron_groups.values()], axis=0)

        # Arithmetic / lookup functions
        def post(source):
            return np.where(ACD[source,:]>0)[0]

        for st in range(0, Nsteps):
            t = self.sim_time + st * self.timestep

            # Record spike activity
            spikes = np.where(self.v>=self.spike_threshold)[0]
            if len(spikes) > 0:
                firings.append(np.vstack((
                    np.full(spikes.shape, t, dtype="int"),
                    spikes
                )).T)

            # Apply synapses into the future
            for spike in spikes:
                targets = post(spike); delays = ACD[spike,targets]
                I[st+delays,targets] += self.S[spike,targets]

            # Reset dynamics
            self.v[spikes] = self.c[spikes]
            self.u[spikes] += self.d[spikes]

            # Update dynamics
            self._autoevolve(I[st,:])

        # Write back simulation state
        self.v = real_v
        self.u = real_u

        return np.concatenate(firings)


class RollbackIzhikevichNN(CommonNN):

    class ImpulsePayload:
        def __init__(self, arrival_time, dest_nids, strengths):
            self.arrival_time = arrival_time
            self.dest_nids = dest_nids
            self.strengths = strengths

        def __lt__(self, other):
            return self.arrival_time < other.arrival_time

    def __init__(self, **kwargs):
        super().__init__("Rollback Izhikevich Network", **kwargs)
        for param in ["a", "b", "c", "d", "v0", "u0"]:
            self.neuron_params.append(param)

        self.v = None
        self.u = None
        self.a = None
        self.b = None
        self.c = None
        self.d = None

        # Cached values to apply to next evolution period
        self.next_I = None

        self.spike_threshold = kwargs.get("spike_threshold", 30)
        self._autoevolve = {
            "preset_1": self._autoevolve_preset_1,
            "preset_2": self._autoevolve_preset_2,
            "preset_3": self._autoevolve_preset_3,
            "preset_4": self._autoevolve_preset_4
        }[kwargs.get("autoevolve_formula", "preset_1")]

    def init(self):
        super().init()
        self.v = np.concatenate([grp[2]["v0"] for grp in self.neuron_groups.values()], axis=0)
        self.u = np.concatenate([grp[2]["u0"] for grp in self.neuron_groups.values()], axis=0)
        self.a = np.concatenate([grp[2]["a"] for grp in self.neuron_groups.values()], axis=0)
        self.b = np.concatenate([grp[2]["b"] for grp in self.neuron_groups.values()], axis=0)
        self.c = np.concatenate([grp[2]["c"] for grp in self.neuron_groups.values()], axis=0)
        self.d = np.concatenate([grp[2]["d"] for grp in self.neuron_groups.values()], axis=0)

        self.next_I = list()

    @property
    def network_state(self):
        state = super().network_state
        state.update({
            "simulation": {
                "v": np.array(self.v),
                "u": np.array(self.u),
            }
        })

        state["network"].update({
            "cache": {
                "next_I": list(self.next_I)
            }
        })

        return state

    def load_state(self, state):
        super().load_state(state)
        self.v = state["simulation"]["v"]
        self.u = state["simulation"]["u"]

        self.a = np.concatenate([grp[2]["a"] for grp in self.neuron_groups.values()], axis=0)
        self.b = np.concatenate([grp[2]["b"] for grp in self.neuron_groups.values()], axis=0)
        self.c = np.concatenate([grp[2]["c"] for grp in self.neuron_groups.values()], axis=0)
        self.d = np.concatenate([grp[2]["d"] for grp in self.neuron_groups.values()], axis=0)

        self.next_I = state["network"]["cache"]["next_I"]

    def _autoevolve_preset_1(self, timestep):
        return (
            self.v + (timestep * (0.04 * np.square(self.v) + 5 * self.v + 140 - self.u)),
            self.u + (timestep * (self.a * (self.b * self.v - self.u)))
        )

    def _autoevolve_preset_2(self, timestep):
        return (
            self.v + (timestep * (0.04 * np.square(self.v) + 4.1 * self.v + 108 - self.u)),
            self.u + (timestep * (self.a * (self.b * self.v - self.u)))
        )

    def _autoevolve_preset_3(self, timestep):
        return (
            self.v + (timestep * (0.04 * np.square(self.v) + 5 * self.v + 140 - self.u)),
            self.u + (timestep * (self.a * (self.b * (self.v + 65))))
        )

    def _autoevolve_preset_4(self, timestep):
        v = self.v + (0.5 * timestep * (0.04 * np.square(self.v) + 5 * self.v + 140 - self.u))
        v += 0.5 * timestep * (0.04 * np.square(self.v) + 5 * self.v + 140 - self.u)
        return (
            v,
            self.u + (timestep * (self.a * (self.b * self.v - self.u)))
        )

    def evolve_for(self, T, *, I=None, jitter=None, collect_every=None):
        if I is None:
            Iqueue = list()
        else:
            I = list(I)
            I.extend(self.next_I)
            heapq.heapify(I)

        if jitter is not None:
            raise ModelError("Jitter is currently unsupported.")

        if collect_every is None:
            collect_every = self.timestep
        elif collect_every < self.timestep:
            raise ModelError("Cannot guarantee data collection on intervals smaller than timestep.")

        firings = list()

        # All other recorded data are accessible in dictionary
        other = dict()
        other["t_axis"] = list()
        other["save_v"] = list()
        other["save_u"] = list()

        # Arithmetic / lookup functions
        def post(source):
            return np.where(self.ACD[source,:]>0)[0]

        def pre(target):
            return np.where(self.ACD[:,target]>0)[0]

        start_time = time.time()
        t = 0
        checkpoint = 0
        last_collected = -collect_every
        while t <= T:
            if t > checkpoint:
                print("Simulating ms {}: {}".format(checkpoint, len(firings)))
                checkpoint += 1

            # 1. Find next impulse payload and determine timestep dynamically
            try:
                next_I = I[0]; dt_to_next_I = next_I.arrival_time - t
                if dt_to_next_I < self.timestep:
                    apply_impulse = True
                    timestep = dt_to_next_I
                else:
                    apply_impulse = False
                    timestep = self.timestep
            except IndexError:
                apply_impulse = False
                timestep = self.timestep

            # 2. Simulate for timestep and detect spikes
            new_v, new_u = self._autoevolve(timestep)

            spikes = np.where(new_v>=self.spike_threshold)[0]
            if len(spikes) > 0:
                # 3a. There are spikes: find earliest spike, and rollback to that specific time
                dv = new_v[spikes] - self.v[spikes]
                dv_to_threshold = self.spike_threshold - self.v[spikes]
                ratio = dv_to_threshold/dv

                earliest_idxs = np.where(ratio == ratio.min())
                earliest_spikes = spikes[earliest_idxs]

                timestep = timestep * ratio.min()
                new_v, new_u = self._autoevolve(timestep)
                spikes = earliest_spikes

            elif apply_impulse:
                # 3b. No spikes: apply impulse at end of simulation and recheck for spikes
                heapq.heappop(I)
                new_v[next_I.dest_nids] += next_I.strengths
                spikes = np.where(new_v>=self.spike_threshold)[0]

            t += timestep
            self.v = new_v
            self.u = new_u

            if t - last_collected >= collect_every:
                last_collected += collect_every
                other["t_axis"].append(t)
                other["save_v"].append(np.array(self.v))
                other["save_u"].append(np.array(self.u))

            # Apply reset dynamics
            if len(spikes) > 0:
                # Collect data on presence of spikes, overwriting past data if needed
                if other["t_axis"][-1] == t:
                    other["save_v"][-1] = np.array(self.v)
                    other["save_u"][-1] == np.array(self.u)
                else:
                    other["t_axis"].append(t)
                    other["save_v"].append(np.array(self.v))
                    other["save_u"].append(np.array(self.u))

                self.v[spikes] = self.c[spikes]
                self.u[spikes] += self.d[spikes]
                for spike in spikes:
                    for target in post(spike):
                        arrival_time = t + self.ACD[spike,target]
                        heapq.heappush(
                            I,
                            self.ImpulsePayload(arrival_time, target, self.S[spike,target])
                        )
                firings.append(np.vstack((
                    np.full(spikes.shape, self.sim_time+t),
                    spikes
                )).T)

        # Save to cache for use in next period
        self.next_I = I
        for impulse in self.next_I:
            impulse.arrival_time -= t

        self.sim_time += t
        end_time = time.time()

        other["t_axis"] = np.array(other["t_axis"])
        other["save_v"] = np.vstack(other["save_v"])
        other["save_u"] = np.vstack(other["save_u"])

        other["meta"] = {
            "sim_time": t,
            "runtime": round(end_time-start_time, 2)
        }

        return np.concatenate(firings), other
