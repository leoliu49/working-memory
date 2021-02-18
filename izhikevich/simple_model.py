"""
One-for-one copy of the Izhikevich simple neuron model from "Simple Model of
Spiking Neurons"

E. M. Izhikevich, "Simple model of spiking neurons," in IEEE Transactions on
Neural Networks, vol. 14, no. 6, pp. 1569-1572, Nov. 2003.
~~~~~~~~
Created by Leo Liu, December 2020
"""
import argparse
import matplotlib.pyplot as plt
import numpy as np
import time


T = 1000
N_exc = 800
N_inh = 200

parser = argparse.ArgumentParser(description=("Run excitatory and inhibitory "
                                 "neurons."))
parser.add_argument("T", metavar="T", type=int, nargs="?", default=T,
                    help="Simulation time (ms)")
args = parser.parse_args()
T = args.T

# Hodgkin-Huxley-type models reduces to 2D ODE after curve fitting:
#   v' = 0.04v^2 + 5v + 140 - u + I
#   u' = a(bv - u)
#   after-spike reset: if v >= 30mV, then v = c, u = u + d
#
# Model variables and parameters:
#   u: recovery variable
#   v: membrane potential of neuron in mV
#   a: time-scale of u (typical a = 0.02)
#   b: sensitivity of u to fluctuations in v (typical b = 0.2)
#   c: reset value of v (typical c = -65mV)
#   d: reset adjustment of u (typical d = 2)
#   I: noisy thalamic input
SPIKE_THRESHOLD = 30
A_TYPICAL = 0.02
B_TYPICAL = 0.2
C_TYPICAL = -65
D_TYPICAL = 2

# Uniform random variables to allow heterogeneity
r = np.random.rand(N_exc+N_inh)

a_exc = np.full((N_exc,), A_TYPICAL)
a_inh = A_TYPICAL + 0.08 * r[N_exc:]
a = np.concatenate((a_exc, a_inh), axis=0)

b_exc = np.full((N_exc,), B_TYPICAL)
b_inh = 0.25 - 0.05 * r[N_exc:]
b = np.concatenate((b_exc, b_inh), axis=0)

c_exc = C_TYPICAL + 15 * np.square(r[:N_exc])
c_inh = np.full((N_inh,), C_TYPICAL)
c = np.concatenate((c_exc,c_inh), axis=0)

d_exc = 8 - 6 * np.square(r[:N_exc])
d_inh = np.full((N_inh,), D_TYPICAL)
d = np.concatenate((d_exc, d_inh), axis=0)

# Randomly assigned synaptic weights (inhibitory connections are stronger)
S = np.random.rand(N_exc+N_inh, N_exc+N_inh)
S[:,:N_exc] *= 0.5
S[:,N_exc:] *= -1

# Initial values of v and u
v = np.full((N_exc+N_inh,), C_TYPICAL, dtype="float64")
u = b * v

# Simulation: run on 1ms resolution for T ms
firings = list()
start_time = time.time()
for t in range(T):
    # Thalamic input
    I = np.concatenate((5*np.random.randn(N_exc), 2*np.random.randn(N_inh)),
                       axis=0)

    spike_indices = np.where(v>=SPIKE_THRESHOLD)[0]
    new_firings = np.vstack((
        np.full(spike_indices.shape, t),
        spike_indices
    )).T
    firings.append(new_firings)
    # firings = np.append(firings, new_firings, axis=0)

    # Reset dynamics
    v[spike_indices] = c[spike_indices]
    u[spike_indices] += d[spike_indices]
    I += np.sum(S[:,spike_indices], axis=1)

    # Update dynamics (step 0.5ms twice for v for numerical stability)
    v += 0.5 * (0.04 * np.square(v) + 5 * v + 140 - u + I)
    v += 0.5 * (0.04 * np.square(v) + 5 * v + 140 - u + I)
    u += a * (b * v - u)

firings = np.concatenate(firings)
end_time = time.time()
print("Simulated {} ms in {} seconds.".format(T, round(end_time-start_time, 2)))

plt.scatter(firings[:,0], firings[:,1], s=0.5, c="steelblue", marker=".")
plt.plot([0, T], [N_exc, N_exc], c="orange", ls="--")
plt.title("Firing patterns of {} neurons over {} ms".format(N_exc+N_inh, T))
plt.xlabel("Time (ms)")
plt.ylabel("Neuron ({}/{} excitatory/inhibitory)".format(N_exc, N_inh))
plt.show()
