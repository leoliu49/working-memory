"""
One-for-one copy of the Izhikevich polychronous spiking model from
"Polychronization: Computation With Spikes"

E. M. Izhikevich, "Polychronization: Computation with Spikes," in Neural
Computation, vol. 18, no. 2, pp. 245-282, 1 Feb. 2006.
~~~~~~~~
Created by Leo Liu, January 2021
"""
import argparse
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import time


SIM_TIME = 60               # simulation time in seconds
SIM_START = 0               # start at second 0

parser = argparse.ArgumentParser(description=("Run a polychronous neuron "
                                 "simulation."))
parser.add_argument("T", metavar="T", type=int, nargs="?", default=SIM_TIME,
                    help="Simulation time (seconds)")
parser.add_argument("--load", metavar="FILE", action="store", type=str,
                    default=None, help="Continue simulation from saved state")
parser.add_argument("--save", action="store_true", default=False,
                    help="Save end state of simulation as Numpy object file")
parser.add_argument("-o", metavar="FILE", action="store", type=str,
                    default=None, help="Output file name of end state")
args = parser.parse_args()
SIM_TIME = args.T

state = None
if args.load is None:
    for png_file in glob.glob("spike_pattern_*.png"):
        os.remove(png_file)
else:
    state = np.load(args.load)
    print("Loading saved state from {}".format(args.load))

# Maintain a 4-to-1 excitatory-to-inhibitory ratio as the mammalian brain
N = 1000
N_RS = 800                  # excitatory (regular spiking) neurons
N_FS = 200                  # inhibitory (fast spiking) neurons
M = 100                     # synapses per neuron (randomly selected)

MAX_DELAY = 20              # max 20ms conduction delay
MAX_STRENGTH = 10           # max synaptic weighting

# Spiking neuron model (see simple_neuron.py)
SPIKE_THRESHOLD = 30
A_RS = 0.02; B_RS = 0.2; C_RS = -65; D_RS = 8
A_FS = 0.10; B_FS = 0.2; C_FS = -65; D_FS = 2

a_RS = np.full((N_RS,), A_RS); a_FS = np.full((N_FS,), A_FS)
a = np.concatenate((a_RS, a_FS), axis=0)

b_RS = np.full((N_RS,), B_RS); b_FS = np.full((N_FS,), B_FS)
b = np.concatenate((b_RS, b_FS), axis=0)

c_RS = np.full((N_RS,), C_RS); c_FS = np.full((N_FS,), C_FS)
c = np.concatenate((c_RS, c_FS), axis=0)

d_RS = np.full((N_RS,), D_RS); d_FS = np.full((N_FS,), D_FS)
d = np.concatenate((d_RS, d_FS), axis=0)

# Assign fixed synaptic weights
S = np.empty((N, M))
S[:N_RS,:] = 6; S[N_RS:,:] = -5
dS = np.zeros((N, M))

# Set conduction delays (1ms precision)
#   for excitatory neuron i --> random delay (j+1) --> indices m:
#       delays_LUT[i,j] = [m1,m2,m3,...]
#   for inhibitory neuron:
#       delays_LUT[i,1] = [1,2,3,...,M]
#   delay_sizes: size of each delays_LUT[i,j] array
delays_LUT = np.empty((N, MAX_DELAY, M), dtype="int64")
delay_sizes = np.zeros((N, MAX_DELAY), dtype="int64")
for i in range(N_RS):
    for m in range(M):
        j = np.random.randint(0, MAX_DELAY)
        delays_LUT[i,j,delay_sizes[i,j]] = m
        delay_sizes[i,j] += 1
for i in range(N_RS, N):
    delays_LUT[i,1,:] = np.linspace(0, M-1, M)
    delay_sizes[i,1] = M

# Set postsynaptic target lookup tables
#   for neuron i --> random delay (j+1) --> index m (neuron k):
#       post_LUT[i,m] = k
post_LUT = np.empty((N, M), dtype="int64")
for i in range(N_RS):
    post_LUT[i,:] = np.random.choice(N, M, replace=False)
for i in range(N_RS, N):
    post_LUT[i,:] = np.random.choice(N_RS, M, replace=False)

# Given a spike, find when and where to apply synaptic weight
#   for excitatory neuron i --> random delay (j+1) --> index m (neuron k):
#       syn_route_LUT[i,m] = [k,j]
syn_route_LUT = np.empty((N, M, 2), dtype="int64")
syn_route_LUT[:,:,0] = post_LUT
for i in range(N):
    for j in range(MAX_DELAY):
        m = delays_LUT[i,j,:delay_sizes[i,j]]
        syn_route_LUT[i,m,1] = j

# Set presynaptic target and STDP auxiliary lookup tables
#   for excitatory neuron i --> random delay (j+1) --> index m (neuron k):
#       pre_LUT[k] = [[i1,m1],[i2,m2],[i3,m3],...]
#       aux_LUT[k] = [MAX_DELAY-j1-1,MAX_DELAY-j2-1,MAX_DELAY-j3-1,...]
#   pre_sizes: size of each pre_LUT[k] and aux_LUT[k] array
pre_LUT = np.empty((N, N, 2), dtype="int64")
aux_LUT = np.empty((N, N), dtype="int64")
pre_sizes = np.zeros((N,), dtype="int64")
for i in range(N_RS):
    for j in range(MAX_DELAY):
        for di in range(delay_sizes[i,j]):
            m = delays_LUT[i,j,di]
            k = post_LUT[i,m]

            pre_LUT[k,pre_sizes[k],0] = i
            pre_LUT[k,pre_sizes[k],1] = m
            aux_LUT[k,pre_sizes[k]] = MAX_DELAY - j - 1
            pre_sizes[k] += 1

# Initial values
H = 1000 + MAX_DELAY + 1        # 1000ms in 1 second, used for STDP history
STDP = np.zeros((N, H))
v = np.full((N_RS+N_FS,), C_RS, dtype="float64")
u = b * v
firings = list()
firings.append(np.array([[-MAX_DELAY, 0]]))     # for STDP calculations

# For "--load" options: overwrite all variables
if state is not None:
    S = state["S"]
    dS = state["dS"]
    delays_LUT = state["delays_LUT"]
    delay_sizes = state["delay_sizes"]
    post_LUT = state["post_LUT"]
    syn_route_LUT = state["syn_route_LUT"]
    pre_LUT = state["pre_LUT"]
    aux_LUT = state["aux_LUT"]
    pre_sizes = state["pre_sizes"]
    STDP = state["STDP"]
    v = state["v"]
    u = state["u"]
    firings = [firing for firing in state["firings"]]
    SIM_START = state["SIM_TIME"]

# Simulation: run on 1ms resolution for SIM_TIME seconds
start_time = time.time()
for sec in range(SIM_START, int(SIM_TIME)):
    sec_start_time = time.time()
    print("Simulating second {} ... ".format(sec), end="", flush=True)
    for t in range(1000):
        # Thalamic input: one random neuron is stimulated
        I = np.zeros((N,)); I[np.random.randint(N)] = 20

        # Find and record spike activity
        spikes = np.where(v >= SPIKE_THRESHOLD)[0]
        exc_spikes = spikes[np.argwhere(spikes < N_RS)]
        new_firings = np.vstack((
            np.full(spikes.shape, t),
            spikes
        )).T
        if new_firings.shape[0] != 0:
            firings.append(new_firings)

        # Reset dynamics
        v[spikes] = c[spikes]
        u[spikes] = u[spikes] + d[spikes]

        # After pre-syn neuron fires, reset STDP
        STDP[spikes,t+MAX_DELAY] = 0.1

        # Apply STDP to dS
        for spike in spikes:
            pre_i = pre_LUT[spike,:pre_sizes[spike],0]
            pre_m = pre_LUT[spike,:pre_sizes[spike],1]
            aux = aux_LUT[spike,:pre_sizes[spike]]

            dS[pre_i,pre_m] += STDP[pre_i,t+aux]

        # Update rules
        #   for each fired neuron i --> neuron k in the past MAX_DELAY period:
        #       I[k] = I[k] + S[i,m]
        #       dS[i,m] = dS[i,m] - 1.2*STDP[k,t+MAX_DELAY]
        df = 0
        while firings[-1+df][0,0] > t - MAX_DELAY:  # all have same recorded t
            firing = firings[-1+df]
            dt = t - firing[0,0] + 1; j = dt-1

            for i in firing[:,1]:
                m = delays_LUT[i,j,:delay_sizes[i,j]]
                k = post_LUT[i,m]

                I[k] += S[i,m]
                dS[i,m] -= 1.2 * STDP[k,t+MAX_DELAY]

            df -= 1

        # Update dynamics (step 0.5ms twice for v for numerical stability)
        v += 0.5 * (0.04 * np.square(v) + 5 * v + 140 - u + I)
        v += 0.5 * (0.04 * np.square(v) + 5 * v + 140 - u + I)
        u += a * (b * v - u)

        # Each ms after firing: STDP = 0.95*STDP (20ms time constant)
        STDP[:,t+MAX_DELAY+1] = 0.95 * STDP[:,t+MAX_DELAY]

    sec_end_time = time.time()
    print("Done in {} seconds.".format(round(sec_end_time-sec_start_time, 2)),
        flush=True)

    # Reset variables after each second:
    #   1. STDP: transfer tail of previous second to new second
    #   2. excitatory S = S + 0.01 + dS (contained within 0 and MAX_STRENGTH)
    #   3. dS = 0.9*dS
    #   4. plot firings and reset, keeping only tail of previous second
    STDP[:,:MAX_DELAY+1] = STDP[:,1000:]
    S[:N_RS,:] = np.maximum(
        0,
        np.minimum(
            MAX_STRENGTH,
            S[:N_RS,:] + 0.01 + dS[:N_RS,:]
        )
    )
    dS *= 0.9

    firings = np.concatenate(firings)

    plt.scatter(firings[:,0], firings[:,1], s=0.5, c="steelblue",
        marker=".")
    plt.plot([0, 1000], [N_RS, N_RS], c="orange", ls="--")
    plt.title("Polychronization: {} neurons at second {}".format(N, sec+1))
    plt.xlabel("Time (ms)")
    plt.ylabel("Neuron ({}/{} excitatory/inhibitory)".format(N_RS, N_FS))
    plt.savefig("spike_pattern_{}s.png".format(sec+1), dpi=300)
    plt.close()

    tail_firings = firings[np.where(firings[:,0] > (1000 - MAX_DELAY))]
    tail_firings[:,0] -= 1000
    firings = [np.array([[-MAX_DELAY, 0]])]
    firings.extend([np.array([tf]) for tf in tail_firings])

end_time = time.time()
print("Simulated a total of {} seconds in {} seconds.".format(SIM_TIME,
    round(end_time-start_time, 2)))

if args.save is True:
    if args.o is None:
        timeinfo = time.gmtime()
        filename = "spnet_{}_{}_{}_{}_{}_{}.npz".format(*(timeinfo[:6]))
    else:
        filename = args.o

    np.savez(
        filename,
        S=S,
        dS=dS,
        delays_LUT=delays_LUT,
        delay_sizes=delay_sizes,
        post_LUT=post_LUT,
        syn_route_LUT=syn_route_LUT,
        pre_LUT=pre_LUT,
        aux_LUT=aux_LUT,
        pre_sizes=pre_sizes,
        STDP=STDP,
        v=v,
        u=u,
        firings=firings,
        SIM_TIME=SIM_TIME
    )

    print("Simulation state file saved to {}.".format(filename))
