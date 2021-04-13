import numpy as np
import time


class ModelError(Exception):
    pass


class NeuralNetwork:

    DEFAULT_TIMESTEP = 1.0

    def __init__(self, model_name, **kwargs):
        self.model_name = model_name

        self.N = 0
        self.neuron_groups = dict()
        self.neuron_params = list()
        self.S = None
        self.ACD = None

        self.sim_time = 0

        self.Ap = None; self.tau_p = None; self.p_decay = None
        self.An = None; self.tau_n = None; self.n_decay = None
        self.STDP_LUT = None; self.STDP_precompute = None

        self.apply_STDP = dict()

        self.timestep = kwargs.get("timestep", NeuralNetwork.DEFAULT_TIMESTEP)
        self.use_STDP = kwargs.get("use_STDP", False)

    def __repr__(self):
        return "<{} ({} neurons)>".format(self.model_name, self.N)

    def __str__(self):
        return repr(self)

    @property
    def network_config(self):
        report = list()
        report.append("{:^20} | {:^5} | Parameters\n".format("Label", "N"))
        for label, grp in self.neuron_groups.items():
            report.append("{:^20} | {:^5} | ".format(label, grp[0]))
            report.append("".join("{:>4} ~= {:>6} ".format(param, round(np.mean(values), 2))
                for param, values in grp[1].items()))
            report.append("\n")

        if self.use_STDP is True and self.p_decay is not None:
            report.append("   {}\n".format("".join(["-" for i in range(50)])))
            report.append("{:^20} | ".format("STDP"))
            report.append("{}*exp(dt/{})".format(self.Ap, self.tau_p))
            report.append("   --->   ")
            report.append("{}*exp(dt/{})".format(self.An, self.tau_n))
            report.append("    (precomputed to {} ratio)".format(self.STDP_precompute))

        return "".join(report)

    def add_neuron_group(self, *, N=1, label=None, **kwargs):
        if label is None:
            label = "neuron_group_{}".format(len(self.neuron_groups))
        if label in self.neuron_groups:
            raise ModelError("Label <{}> already exists.".format(label))

        try:
            grp_values = dict({param: np.array(kwargs[param], dtype="float64")
                for param in self.neuron_params})

            for param in self.neuron_params:
                if len(kwargs[param]) != N:
                    raise ModelError("Parameter <{}> is mismatched with N = {}.".format(param, N))

            self.neuron_groups[label] = (N, grp_values)
        except KeyError as e:
            raise ModelError("Neuron parameter <{}> not specified.".format(e.args[0]))

        self.apply_STDP[label] = self.use_STDP
        self.N += N

    def set_synapse_matrices(self, *, S, ACD):
        if S.shape[0] != self.N or S.shape[1] != self.N:
            raise ModelError("Dimensions for synapse matrix (S) must be {0}x{0}".format(N))
        if ACD.shape[0] != self.N or ACD.shape[1] != self.N:
            raise ModelError("Dimensions for delay matrix (ACD) must be {0}x{0}".format(N))
        self.S = np.array(S, dtype="float64")
        self.ACD = np.array(ACD, dtype="float64")

        # Negative/zero conduction delays are assumed as disconnected
        self.ACD[self.ACD <= 0] = 0

    def set_STDP_curve(self, Ap, tau_p, An, tau_n, precompute=0.001):
        if not self.use_STDP:
            raise ModelError("STDP is disabled.")

        self.Ap = Ap; self.tau_p = tau_p; self.p_decay = 1 - (1/(tau_p/self.timestep))
        self.An = -1 * abs(An); self.tau_n = tau_n; self.n_decay = 1 - (1/(tau_n/self.timestep))

        # Precompute STDP values for faster lookup during simulation
        self.STDP_precompute = precompute
        self.STDP_LUT = dict()

        stdp = self.Ap
        i = 0
        while abs(stdp) > self.STDP_precompute * abs(Ap):
            self.STDP_LUT[i] = stdp
            stdp *= self.p_decay
            i += 1

        stdp = self.An * self.n_decay
        i = -1
        while abs(stdp) > self.STDP_precompute * abs(An):
            self.STDP_LUT[i] = stdp
            stdp *= self.n_decay
            i -= 1

    def disable_STDP_for(self, label):
        self.apply_STDP[label] = False

    def init(self):
        if self.S is None or self.ACD is None:
            raise ModelError("Synapse weighting matrix (S) or delay matrix (ACD) is not set.")
        if self.use_STDP is True and (self.Ap is None or self.An is None):
            raise ModelError("STDP curve parameters are not set.")

    def get_state(self):
        state = {
            "model_name": self.model_name,
            "sim_time": self.sim_time,
            "timestep": self.timestep,
            "network": {
                "N": self.N,
                "neuron_params": self.neuron_params,
                "neuron_groups": self.neuron_groups,
                "S": np.array(self.S),
                "ACD": np.array(self.ACD),
            }
        }

        if self.use_STDP is True:
            state["STDP"] = {
                "Ap": self.Ap,
                "tau_p": self.tau_p,
                "An": self.An,
                "tau_n": self.tau_n,
                "precompute": self.STDP_precompute,
                "apply_to": self.apply_STDP
            }

        return state

    def save_state_to_file(self, filename):
        state = self.get_state()
        np.savez(filename, **state)

    def load_state(self, state):
        self.model_name = state["model_name"]
        self.sim_time = state["sim_time"]
        self.timestep = state["timestep"]
        self.N = state["network"]["N"]
        self.neuron_groups = state["network"]["neuron_groups"]
        self.neuron_params = state["network"]["neuron_params"]

        self.set_synapse_matrices(S=state["network"]["S"], ACD=state["network"]["ACD"])

        if "STDP" in state:
            self.use_STDP = True
            self.set_STDP_curve(state["STDP"]["Ap"], state["STDP"]["tau_p"], state["STDP"]["An"],
                state["STDP"]["tau_n"], state["STDP"]["precompute"])
            self.apply_STDP = state["STDP"]["apply_to"]

        print("Loaded state at {} ms.", self.sim_time)
        print(self.network_config)

    def load_state_from_file(self, filename):
        state = np.load(filename, allow_pickle=True)
        state = {key: item.item() for key, item in state.items()}
        self.load_state(state)


class IzhikevichNetwork(NeuralNetwork):

    SPIKE_THRESHOLD = 30

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

        self.I_cache = None

    def init(self):
        super().init()
        self.v = np.concatenate([grp[1]["v0"] for grp in self.neuron_groups.values()], axis=0)
        self.u = np.concatenate([grp[1]["u0"] for grp in self.neuron_groups.values()], axis=0)
        self.a = np.concatenate([grp[1]["a"] for grp in self.neuron_groups.values()], axis=0)
        self.b = np.concatenate([grp[1]["b"] for grp in self.neuron_groups.values()], axis=0)
        self.c = np.concatenate([grp[1]["c"] for grp in self.neuron_groups.values()], axis=0)
        self.d = np.concatenate([grp[1]["d"] for grp in self.neuron_groups.values()], axis=0)

        self.I_cache = None

    def get_state(self):
        state = super().get_state()
        state["simulation"] = {
            "v": np.array(self.v),
            "u": np.array(self.u),
            "I_cache": np.array(self.I_cache)
        }

        return state

    def load_state(self, state):
        super().load_state(state)
        self.v = state["simulation"]["v"]
        self.u = state["simulation"]["u"]
        self.I_cache = state["simulation"]["I_cache"]

        self.a = np.concatenate([grp[1]["a"] for grp in self.neuron_groups.values()], axis=0)
        self.b = np.concatenate([grp[1]["b"] for grp in self.neuron_groups.values()], axis=0)
        self.c = np.concatenate([grp[1]["c"] for grp in self.neuron_groups.values()], axis=0)
        self.d = np.concatenate([grp[1]["d"] for grp in self.neuron_groups.values()], axis=0)

    def evolve_for(self, T, *, I=None, save_v=False, save_u=False, save_I=False):
        steps = int(T/self.timestep)
        if I is None:
            I = np.zeros((steps, self.N), dtype="float64")
        else:
            if I.shape[0] != steps and I.shape[1] != self.N:
                raise ModelError("Dimensions for input current (I) must be {}x{}.".format(steps,
                    self.N))

        # Convert delay to discrete steps
        ACD_steps = np.array(np.ceil(self.ACD/self.timestep), dtype="int")
        max_delay = np.max(ACD_steps)

        # Extend current input to include spikes arriving after simulation ends
        I = np.concatenate((I, np.zeros((max_delay, self.N),
            dtype="float64")), axis=0)

        # Apply spike activity from previous period
        if self.I_cache is not None:
            I[:self.I_cache.shape[0],:] += self.I_cache

        # Complete spike transfer history
        #   at time t: neuron i --> delay j --> neuron k
        #       spike_raster[t] = [i1,i2,i3...]
        #       spike_graph[t,i,k] = j
        spike_raster = list([list() for i in range(steps)])
        spike_graph = np.zeros((steps, self.N, self.N), dtype="int")

        # All other recorded data accessible through kwargs
        other = dict()

        # Complete voltage and recovery variable evolution history
        if save_v is True:
            other["save_v"] = np.empty((steps, self.N), dtype="float64")
        if save_u is True:
            other["save_u"] = np.empty((steps, self.N), dtype="float64")

        start_time = time.time()
        for st in range(0, steps):
            t = self.sim_time + st * self.timestep

            # Find and record spike activity
            spikes = np.where(self.v>=IzhikevichNetwork.SPIKE_THRESHOLD)[0]
            spike_raster[st].extend(spikes)
            for spike in spikes:
                targets = np.where(ACD_steps[spike,:] > 0)[0]
                delays = ACD_steps[spike,targets]

                I[st+delays,targets] += self.S[spike,targets]
                spike_graph[st,spike,targets] = delays

            # Reset dynamics
            self.v[spikes] = self.c[spikes]
            self.u[spikes] += self.d[spikes]

            if self.use_STDP is True:
                pass

            # Update dynamics (half-step twice for v for numerical stability)
            self.v += 0.5 * (0.04 * np.square(self.v) + 5 * self.v + 140 - self.u + I[st,:])
            self.v += 0.5 * (0.04 * np.square(self.v) + 5 * self.v + 140 - self.u + I[st,:])
            self.u += self.a * (self.b * self.v - self.u)

            if save_v is True:
                other["save_v"][st,:] = np.array(self.v)
            if save_u is True:
                other["save_u"][st,:] = np.array(self.u)

        if self.use_STDP is True:
            pass

        self.I_cache = np.array(I[steps:,:], dtype="float64")

        end_time = time.time()
        print("Simulated {} ms ({} steps) in {} seconds.".format(T, steps,
            round(end_time-start_time, 2)))

        self.sim_time += steps * self.timestep

        if save_I is True:
            other["save_I"] = np.array(I)

        return spike_raster, spike_graph, other
