import numpy as np


class ModelError(Exception):
    pass


class BaseNN:
    def __init__(self, model_name):
        self.model_name = model_name

        self.N = 0
        self.neuron_groups = dict()
        self.neuron_params = list()

        self.sim_time = 0

    def __repr__(self):
        return "<{} ({} neurons) @ {} ms>".format(self.model_name, self.N, self.sim_time)

    def __str__(self):
        return repr(self)

    @property
    def network_config(self):
        report = list()
        report.append("{:^20} | {:^5} | Parameters\n".format("Label", "N"))
        for label, grp in self.neuron_groups.items():
            report.append("{:^20} | {:^5} | ".format(label, grp[1]-grp[0]))
            report.append("".join("{:>4} ~= {:>6} ".format(param, round(np.mean(values), 2))
                for param, values in grp[2].items()))
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

            self.neuron_groups[label] = (self.N, self.N+N, grp_values)
        except KeyError as e:
            raise ModelError("Neuron parameter <{}> not specified.".format(e.args[0]))

        self.N += N

    def init(self):
        self.sim_time = 0

    @property
    def network_state(self):
        state = {
            "model_name": self.model_name,
            "network": {
                "N": self.N,
                "neuron_params": self.neuron_params,
                "neuron_groups": self.neuron_groups
            },
            "sim_time": self.sim_time
        }

        return state

    def load_state(self, state):
        self.N = state["network"]["N"]
        self.neuron_params = state["network"]["neuron_params"]
        self.neuron_groups = state["network"]["neuron_groups"]

        self.sim_time = state["sim_time"]


class CommonNN(BaseNN):

    DEFAULT_TIMESTEP = 1.0

    def __init__(self, model_name, **kwargs):
        super().__init__(model_name)

        # For neuron i --> delay j --> apply v to neuron k
        #   synaptic weight matrix:     S[i,k] = v
        #   axonal conduction delay:    ACD[i,k] = j
        self.S = None
        self.ACD = None

        # STDP curve parameters
        self.Ap = None; self.tau_p = None; self.p_decay = None
        self.An = None; self.tau_n = None; self.n_decay = None
        self.dS = None; self.dS_decay = None
        self.S_min = None; self.S_max = None
        self.apply_STDP = set()

        self.timestep = kwargs.get("timestep", CommonNN.DEFAULT_TIMESTEP)
        self.use_STDP = kwargs.get("use_STDP", False)

    @property
    def network_config(self):
        report = [super().network_config]

        if self.use_STDP and self.p_decay is not None:
            report.append("   {}\n".format("".join(["-" for i in range(50)])))
            report.append("{:^20} | ".format("STDP"))
            report.append("{}*exp(Δt/{})".format(self.Ap, self.tau_p))
            report.append("   --->   ")
            report.append("{}*exp(Δt/{})".format(self.An, self.tau_n))

        return "".join(report)

    def add_neuron_group(self, *, N=1, label=None, **kwargs):
        super().add_neuron_group(N=N, label=label, **kwargs)
        if self.use_STDP:
            self.apply_STDP.update({i for i in range(self.N-N, self.N)})

    def set_synapse_matrices(self, *, S, ACD):
        if S.shape[0] != self.N or S.shape[1] != self.N:
            raise ModelError("Dimensions for synapse matrix (S) must be {0}x{0}".format(self.N))
        if ACD.shape[0] != self.N or ACD.shape[1] != self.N:
            raise ModelError("Dimensions for delay matrix (ACD) must be {0}x{0}".format(self.N))
        self.S = np.array(S, dtype="float64")
        self.ACD = np.array(ACD, dtype="float64")

        # Synaptic strength inversely correlated with timestep
        self.S /= self.timestep

        # Negative/zero conduction delays are assumed as disconnected
        self.ACD[self.ACD < 0] = 0

    def set_STDP_curve(self, Ap, tau_p, An, tau_n, dS_decay=0.9, S_min=0, S_max=10):
        if not self.use_STDP:
            raise ModelError("STDP is disabled.")

        self.Ap = Ap; self.tau_p = tau_p; self.p_decay = 1 - (1/(tau_p/self.timestep))
        self.An = -1 * abs(An); self.tau_n = tau_n; self.n_decay = 1 - (1/(tau_n/self.timestep))

        self.dS_decay = dS_decay
        self.S_min = S_min; self.S_max = S_max

    def disable_STDP_for(self, label):
        grp = self.neuron_groups[label]
        self.apply_STDP.difference_update({i for i in range(grp[0], grp[1])})

    def init(self):
        if self.S is None or self.ACD is None:
            raise ModelError("Synapse weighting matrix (S) or delay matrix (ACD) is not set.")
        if self.use_STDP is True and (self.Ap is None or self.An is None):
            raise ModelError("STDP curve parameters are not set.")

        if self.use_STDP:
            self.dS = np.zeros_like(self.S)

        self.apply_STDP = np.array([i for i in self.apply_STDP])

    @property
    def network_state(self):
        state = super().network_state

        state["network"].update({
            "S": np.array(self.S),
            "ACD": np.array(self.ACD),
            "timestep": self.timestep,
            "use_STDP": self.use_STDP
        })

        if self.use_STDP:
            state.update({
                "STDP": {
                    "Ap": self.Ap,
                    "tau_p": self.tau_n,
                    "An": self.An,
                    "tau_n": self.tau_n,
                    "dS": self.dS,
                    "dS_decay": self.dS_decay,
                    "S_min": self.S_min,
                    "S_max": self.S_max,
                    "apply_to": self.apply_STDP
                }
            })

        return state

    def load_state(self, state):
        super().load_state(state)
        self.S = state["network"]["S"]
        self.ACD = state["network"]["ACD"]

        self.timestep = state["network"]["timestep"]
        self.use_STDP = state["network"]["use_STDP"]

        if self.use_STDP:
            self.set_STDP_curve(state["STDP"]["Ap"], state["STDP"]["tau_p"], state["STDP"]["An"],
                state["STDP"]["tau_n"], state["STDP"]["dS_decay"], state["STDP"]["S_min"],
                state["STDP"]["S_max"])
            self.dS = state["STDP"]["dS"]
            self.apply_STDP = state["STDP"]["apply_to"]
