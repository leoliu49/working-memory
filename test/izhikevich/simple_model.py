import argparse
import matplotlib.pyplot as plt
import numpy as np
from wmsim.models import *


T = 1000
N_exc = 800
N_inh = 200

A_TYPICAL = 0.02
B_TYPICAL = 0.2
C_TYPICAL = -65
D_TYPICAL = 2

r = np.random.rand(N_exc+N_inh)

a_exc = np.full((N_exc,), A_TYPICAL)
a_inh = A_TYPICAL + 0.08 * r[N_exc:]

b_exc = np.full((N_exc,), B_TYPICAL)
b_inh = 0.25 - 0.05 * r[N_exc:]

c_exc = C_TYPICAL + 15 * np.square(r[:N_exc])
c_inh = np.full((N_inh,), C_TYPICAL)

d_exc = 8 - 6 * np.square(r[:N_exc])
d_inh = np.full((N_inh,), D_TYPICAL)

S = np.random.rand(N_exc+N_inh, N_exc+N_inh)
S[:N_exc,:] *= 0.5
S[N_exc:,:] *= -1

ACD = np.full((N_exc+N_inh, N_exc+N_inh), 1)

v0_exc = np.full((N_exc,), C_TYPICAL, dtype="float64")
v0_inh = np.full((N_inh,), C_TYPICAL, dtype="float64")

u0_exc = b_exc * v0_exc
u0_inh = b_inh * v0_inh

noisy_input = np.random.randn(T, N_exc+N_inh)

parser = argparse.ArgumentParser()
parser.add_argument("--timesteps", metavar="dt", type=float, nargs="+", default=[1.0, 0.5, 0.25],
    help="Simulation timestep (ms)")
parser.add_argument("--track", metavar="id", type=int, nargs="+", default=list(),
    help="Track neuron behavior over time")
args = parser.parse_args()
timesteps = list(dict.fromkeys(args.timesteps))
track_nids = list(dict.fromkeys(args.track))

vs = dict(); us = dict()
for nid in track_nids:
    vs[nid] = dict.fromkeys(timesteps)
    us[nid] = dict.fromkeys(timesteps)

# Test sequence
print("Simulating simple model for {} ms with dt = {}".format(T, timesteps))
for timestep in timesteps:
    n = IzhikevichNetwork(timestep=timestep)
    n.add_neuron_group(N=N_exc, a=a_exc, b=b_exc, c=c_exc, d=d_exc, v0=v0_exc, u0=u0_exc, label="Excitatory")
    n.add_neuron_group(N=N_inh, a=a_inh, b=b_inh, c=c_inh, d=d_inh, v0=v0_inh, u0=u0_inh, label="Inhibitory")
    n.set_synapse_matrices(S=S, ACD=ACD)
    n.init()

    # Set I (spike at every 1 ms)
    Iin = np.empty(noisy_input.shape, dtype="float64")
    Iin[:,:N_exc] = noisy_input[:,:N_exc] * (5 / timestep)
    Iin[:,N_exc:] = noisy_input[:,N_exc:] * (2 / timestep)
    I = np.zeros((int(np.ceil(T/timestep)), N_exc+N_inh))
    I[::int(np.ceil(1/timestep)),:] = Iin

    raster, graph, other = n.evolve_for(T, I=I, save_v=True, save_u=True, save_I=True)
    firings = list()
    for st in range(len(raster)):
        t = st * n.timestep
        new_firings = np.vstack((
            np.full((len(raster[st]),), t),
            raster[st]
        )).T
        if new_firings.shape[0] != 0:
            firings.append(new_firings)
    firings = np.concatenate(firings)

    plt.scatter(firings[:,0], firings[:,1], s=0.5, c="steelblue", marker=".")
    plt.plot([0, T], [N_exc, N_exc], c="orange", ls="--")
    plt.title("Firing patterns of {} neurons over {} ms (dt = {} ms)".format(N_exc+N_inh, T,
        timestep))
    plt.xlabel("Time (ms)")
    plt.ylabel("Neuron ({}/{} excitatory/inhibitory)".format(N_exc, N_inh))
    plt.show()

    for nid in track_nids:
        vs[nid][timestep] = other["save_v"][:,nid]
        us[nid][timestep] = other["save_u"][:,nid]

# Plot voltage/recovery traces of tracked neurons
x_axes = {timestep: np.linspace(0, T-timestep, int(np.ceil(T/timestep))) for timestep in timesteps}
for nid in track_nids:
    fig, axs = plt.subplots(2, 1)
    for timestep in timesteps:
        axs[0].plot(x_axes[timestep], vs[nid][timestep], linewidth=0.6,
            label="{} ms".format(timestep))
        axs[0].set_ylabel("Membrane potential (mV)")

        axs[1].plot(x_axes[timestep], us[nid][timestep], linewidth=0.6,
            label="{} ms".format(timestep))
        axs[1].set_ylabel("Recovery variable")

        plt.xlabel("Time (ms)")
    axs[0].legend(); axs[1].legend()
    plt.suptitle("v and u trace of neuron #{}".format(nid))
    plt.show()
