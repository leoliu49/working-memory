import numpy as np
import time


class ModelError(Exception):
    pass


class NeuralNetwork:

    DEFAULT_TIMESTEP = 1

    def __init__(self, model_name, **kwargs):
        self.model_name = model_name

        self.N = 0
        self.neuron_groups = dict()
        self.neuron_params = list()
        self.S = None

        self.sim_time = None

        self.Ap = None; self.tau_p = None; self.p_decay = None
        self.An = None; self.tau_n = None; self.n_decay = None

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

    def set_STDP_curve(self, Ap, tau_p, An, tau_n):
        if not self.use_STDP:
            raise ModelError("STDP is disabled.")

        self.Ap = Ap; self.tau_p = tau_p; self.p_decay = 1 - (1/(tau_p/self.timestep))
        self.An = -1 * abs(An); self.tau_n = tau_n; self.n_decay = 1 - (1/(tau_n/self.timestep))

    def init(self):
        self.sim_time = 0

        if self.S is None or self.ACD is None:
            raise ModelError("Synapse weighting matrix (S) or delay matrix (ACD) is not set.")
        if self.use_STDP is True and (self.Ap is None or self.An is None):
            raise ModelError("STDP curve parameters are not set.")


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

    def evolve_for(self, T, *, I=None):
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
        #       inbound[t+j,k,i] = j
        #       outbound[t,i,k] = j
        inbound = np.zeros((steps+max_delay, self.N, self.N), dtype="int")
        outbound = np.zeros((steps, self.N, self.N), dtype="int")

        start_time = time.time()
        for st in range(0, steps):
            t = self.sim_time + st * self.timestep

            # Find and record spike activity
            spikes = np.where(self.v>=IzhikevichNetwork.SPIKE_THRESHOLD)[0]
            for spike in spikes:
                targets = np.where(ACD_steps[spike,:] > 0)[0]
                delays = ACD_steps[spike,targets]

                I[st+delays,targets] += self.S[spike,targets]

                inbound[st+delays,targets,spike] = delays
                outbound[st,spike,targets] = delays

            # Reset dynamics
            self.v[spikes] = self.c[spikes]
            self.u[spikes] += self.d[spikes]

            if self.use_STDP is True:
                pass

            # Update dynamics (half-step twice for v for numerical stability)
            self.v += 0.5 * (0.04 * np.square(self.v) + 5 * self.v + 140 - self.u + I[st,:])
            self.v += 0.5 * (0.04 * np.square(self.v) + 5 * self.v + 140 - self.u + I[st,:])
            self.u += self.a * (self.b * self.v - self.u)

        if self.use_STDP is True:
            pass

        self.I_cache = np.array(I[steps:,:], dtype="float64")

        end_time = time.time()
        print("Simulated {} ms ({} steps) in {} seconds.".format(T, steps,
            round(end_time-start_time, 2)))

        self.sim_time += steps * self.timestep

        return inbound, outbound
