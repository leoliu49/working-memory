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
        self.firings = None

        if "timestep" in kwargs:
            self.timestep = kwargs["timestep"]
        else:
            self.timestep = NeuralNetwork.DEFAULT_TIMESTEP

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
        self.ACD = np.array(ACD, dtype="int")

    def init(self):
        self.sim_time = 0


class IzhikevichNetwork(NeuralNetwork):

    SPIKE_THRESHOLD = 30

    def __init__(self, *, use_STDP=False, **kwargs):
        super().__init__("Izhikevich Network", **kwargs)
        for param in ["a", "b", "c", "d", "v0", "u0"]:
            self.neuron_params.append(param)

        self.use_STDP = use_STDP

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

        # Extend current input to include spikes arriving after simulation ends
        max_delay = np.max(self.ACD)
        I = np.concatenate((I, np.zeros((int(np.ceil(max_delay / self.timestep)), self.N),
            dtype="float64")), axis=0)

        # Apply spike activity from previous period
        if self.I_cache is not None:
            I[:self.I_cache.shape[0],:] += self.I_cache

        firings = list()
        start_time = time.time()
        for st in range(0, steps):
            t = self.sim_time + st * self.timestep

            # Find and record spike activity
            spikes = np.where(self.v>=IzhikevichNetwork.SPIKE_THRESHOLD)[0]
            new_firings = np.vstack((
                np.full(spikes.shape, t),
                spikes
            )).T
            if new_firings.shape[0] != 0:
                firings.append(new_firings)

            # Reset dynamics
            self.v[spikes] = self.c[spikes]
            self.u[spikes] += self.d[spikes]

            # Set spike conduction in the future
            for spike in spikes:
                delays = np.array(np.ceil(self.ACD[spike,:]), "int")
                I[st+delays,[i for i in range(self.N)]] += self.S[spike,:]

            if self.use_STDP:
                pass # TODO

            # Update dynamics (half-step twice for v for numerical stability)
            self.v += 0.5 * (0.04 * np.square(self.v) + 5 * self.v + 140 - self.u + I[st,:])
            self.v += 0.5 * (0.04 * np.square(self.v) + 5 * self.v + 140 - self.u + I[st,:])
            self.u += self.a * (self.b * self.v - self.u)

        if self.use_STDP:
            pass # TODO

        self.I_cache = np.array(I[steps:,:], dtype="float64")

        end_time = time.time()
        print("Simulated {} ms ({} steps) in {} seconds.".format(T, steps,
            round(end_time-start_time, 2)))

        firings = np.concatenate(firings)
        if self.firings is None:
            self.firings = firings
        else:
            self.firings = np.concatenate((self.firings, firings), axis=0)

        self.sim_time += steps * self.timestep
