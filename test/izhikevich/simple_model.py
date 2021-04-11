import matplotlib.pyplot as plt
from wmsim.models import *

n = IzhikevichNetwork()

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

n.add_neuron_group(N=N_exc, a=a_exc, b=b_exc, c=c_exc, d=d_exc, v0=v0_exc, u0=u0_exc, label="Excitatory")
n.add_neuron_group(N=N_inh, a=a_inh, b=b_inh, c=c_inh, d=d_inh, v0=v0_inh, u0=u0_inh, label="Inhibitory")
n.set_synapse_matrices(S=S, ACD=ACD)

n.init()

I = np.random.randn(T, N_exc+N_inh)
I[:,:N_exc] *= 5
I[:,N_exc:] *= 2

raster, graph = n.evolve_for(T, I=I)
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
plt.title("Firing patterns of {} neurons over {} ms".format(N_exc+N_inh, T))
plt.xlabel("Time (ms)")
plt.ylabel("Neuron ({}/{} excitatory/inhibitory)".format(N_exc, N_inh))
plt.show()
