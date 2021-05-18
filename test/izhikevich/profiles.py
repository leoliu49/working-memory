"""
Modified version of the Izhikevich neuro-computational profiles from "Which
Model to Use for Cortical Spiking Neurons?"

E. M. Izhikevich, "Which model to use for cortical spiking neurons?," in IEEE
Transactions on Neural Networks, vol. 15, no. 5, pp. 1063-1070, Sept. 2004
~~~~~~~~
Created by Leo Liu, April 2021
"""
import argparse
import matplotlib.pyplot as plt
import numpy as np
from wmsim.izhikevich import IzhikevichNN


def make_I(profile, T, timestep):
    steps = int(T/timestep)
    I = np.zeros((steps, 1), dtype="float64")
    if profile == "tonic spiking":
        I[int(steps/10):] = 14
    elif profile == "phasic spiking":
        I[int(steps/10):] = 0.5
    elif profile == "tonic bursting":
        I[int(steps/10):] = 15
    elif profile == "phasic bursting":
        I[int(steps/10):] = 0.6
    elif profile == "mixed mode":
        I[int(steps/10):] = 10
    elif profile == "spike freq adapt":
        I[int(steps/10):] = 30
    elif profile == "class 1 exc.":
        I[int(steps/10)-1:,0] = 0.075 * np.linspace(0, 270, steps-int(steps/10)+1)
    elif profile == "class 2 exc.":
        I -= 0.5
        I[int(steps/10)-1:,0] += 0.015 * np.linspace(0, 270, steps-int(steps/10)+1)
    elif profile == "spike latency":
        I[int(steps/10):int(steps/10+3/timestep)] = 7.04
    elif profile == "subthresh. osc.":
        I[int(steps/10):int(steps/10+5/timestep)] = 2
    elif profile == "resonator":
        T1 = int(steps/10)
        T2 = T1 + int(20/timestep)
        T3 = int(0.7*steps)
        T4 = T3 + int(40/timestep)
        I[T1:T1+int(4/timestep)] = 0.65
        I[T2:T2+int(4/timestep)] = 0.65
        I[T3:T3+int(4/timestep)] = 0.65
        I[T4:T4+int(4/timestep)] = 0.65
    elif profile == "integrator":
        T1 = int(steps/11)
        T2 = T1 + int(5/timestep)
        T3 = int(0.7*steps)
        T4 = T3 + int(10/timestep)
        I[T1:T1+int(2/timestep)] = 9
        I[T2:T2+int(2/timestep)] = 9
        I[T3:T3+int(2/timestep)] = 9
        I[T4:T4+int(2/timestep)] = 9
    elif profile == "rebound spike":
        I[int(steps/10):int(steps/10+5/timestep)] = -15
    elif profile == "rebound burst":
        I[int(steps/10):int(steps/10+5/timestep)] = -15
    elif profile == "thresh. var.":
        I[int(steps/10):int(steps/10+5/timestep)] = 1
        I[int(0.7*steps):int(0.7*steps+5/timestep)] = -6
        I[int(0.8*steps):int(0.8*steps+5/timestep)] = 1
    elif profile == "bistability":
        I += 0.24
        I[int(steps/8):int(steps/8+5/timestep)] = 1.24
        I[int(0.72*steps):int(0.72*steps+5/timestep)] = 1.24
    elif profile == "DAP":
        I[int(steps/5-1/timestep):int(steps/5+1/timestep)] = 20
    elif profile == "accomodation":
        I[:int(steps/2),0] = 0.04 * np.linspace(0, 200, int(steps/2))
        I[int(0.75*steps):int(0.75*steps+12.5/timestep),0] = np.linspace(0, 4, int(12.5/timestep))
    elif profile == "inh. ind. spike":
        I += 80
        I[int(50/timestep):int(250/timestep)] = 75
    elif profile == "inh. ind. burst":
        I += 80
        I[int(50/timestep):int(250/timestep)] = 75

    return I


profiles = {
    "tonic spiking":    [0.02,  0.2,    -65,    6,      -70,    100],
    "phasic spiking":   [0.02,  0.25,   -65,    6,      -64,    200],
    "tonic bursting":   [0.02,  0.2,    -50,    2,      -70,    220],
    "phasic bursting":  [0.02,  0.25,   -55,    0.05,   -64,    200],
    "mixed mode":       [0.02,  0.2,    -55,    4,      -70,    160],
    "spike freq adapt": [0.01,  0.2,    -65,    8,      -70,    85],
    "class 1 exc.":     [0.02,  -0.1,   -55,    6,      -60,    300],
    "class 2 exc.":     [0.2,   0.26,   -65,    0,      -64,    300],
    "spike latency":    [0.02,  0.2,    -65,    6,      -70,    100],
    "subthresh. osc.":  [0.05,  0.26,   -60,    0,      -62,    200],
    "resonator":        [0.1,   0.26,   -60,    -1,     -62,    400],
    "integrator":       [0.02,  -0.1,   -55,    6,      -60,    100],
    "rebound spike":    [0.03,  0.25,   -60,    4,      -64,    200],
    "rebound burst":    [0.03,  0.25,   -52,    0,      -64,    200],
    "thresh. var.":     [0.03,  0.25,   -60,    4,      -64,    100],
    "bistability":      [0.1,   0.26,   -60,    0,      -61,    300],
    "DAP":              [1,     0.2,    -60,    -21,    -70,    50],
    "accomodation":     [0.02,  1,      -55,    4,      -65,    400],
    "inh. ind. spike":  [-0.02, -1,     -60,    8,      -63.8,  350],
    "inh. ind. burst":  [-0.026,-1,     -45,    -2,     -63.8,  350]
}

parser = argparse.ArgumentParser()
parser.add_argument("--timesteps", metavar="dt", type=float, nargs="+", default=[0.25, 0.05, 0.01],
    help="Simulation timestep (ms)")
args = parser.parse_args()
timesteps = list(dict.fromkeys(args.timesteps))

fig, axs = plt.subplots(5, 4)
last_ax = None

for plot_num, item in enumerate(profiles.items()):
    profile = item[0]; variables = item[1]

    a = variables[0]; b = variables[1]; c = variables[2]; d = variables[3]
    a = np.full((1,), a)
    b = np.full((1,), b)
    c = np.full((1,), c)
    d = np.full((1,), d)

    v0 = np.full((1,), variables[4])
    u0 = np.full((1,), -16) if profile == "accomodation" else b * v0

    T = variables[5]

    if profile in {"class 1 exc.", "integrator"}:
        autoevolve_formula = "preset_2"
    elif profile == "accomodation":
        autoevolve_formula = "preset_3"
    else:
        autoevolve_formula = "preset_1"

    for timestep in timesteps:
        steps = int(T/timestep)
        time = np.linspace(0, T-timestep, steps)

        n = IzhikevichNN(timestep=timestep, autoevolve_formula=autoevolve_formula)
        n.add_neuron_group(N=1, a=a, b=b, c=c, d=d, v0=v0, u0=u0, label=profile)
        n.set_synapse_matrices(S=np.zeros((1, 1)), ACD=np.zeros((1, 1)))
        n.init()

        I = make_I(profile, T, timestep)

        raster, other = n.evolve_for(T, I=I)
        all_v = np.clip(other["save_v"], None, n.spike_threshold)
        all_u = other["save_u"]

        ax = axs[int(plot_num/4), plot_num%4]
        ax.plot(time, all_v, linewidth=0.5, label="dt = {} ms".format(timestep))
        ax.set_xticks([])
        ax.set_yticks([])

        twin = ax.twinx()
        twin.plot(time, I[:,0]-np.min(I), linewidth=0.2)
        I_range = np.max(I) - np.min(I)
        twin.set_ylim(-1*I_range, 4*I_range)
        twin.set_xticks([])
        twin.set_yticks([])

        last_ax = ax

plt.suptitle("Spiking Profiles Identified by Izhikevich")
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels)
plt.show()
