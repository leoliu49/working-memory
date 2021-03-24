"""
One-for-one copy of the (old) polychronous neuron group search algorithm written
by Izhikevich

E. M. Izhikevich, "Polychronization: Computation with Spikes," in Neural
Computation, vol. 18, no. 2, pp. 245-282, 1 Feb. 2006.
~~~~~~~~
Created by Leo Liu, February 2021
"""
import argparse
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import time


class PNG:
    def __init__(self, polygroup, anchors, max_path_length):
        self.polygroup = polygroup
        self.anchors = anchors
        self.max_path_length = max_path_length

    def plot(self, savefig=None):
        fig = plt.figure()

        spikes = dict()
        traces = list()
        for trace in self.polygroup:
            spikes.setdefault(trace[1], set()).add(trace[0])
            spikes.setdefault(trace[3], set()).add(trace[4])
            traces.append(([trace[0], trace[4]], [trace[1], trace[3]]))

        scatter_x = list()
        scatter_y = list()
        for n, ts in spikes.items():
            for t in ts:
                scatter_x.append(t)
                scatter_y.append(n)
        plt.scatter(scatter_x, scatter_y, c="black", marker="^", zorder=2)

        for trace in traces:
            plt.plot(trace[0], trace[1], zorder=1)

        if savefig is None:
            fig.show()
        else:
            fig.savefig(savefig, dpi=300)


# Find polychronous neurons given a set of anchor neurons firing patterns
#   1. Run simulation with no STDP for T ms
#   2. Assign thalamus input to trigger anchor neurons to spike at precise time
#   3. For each firing, look at the last time window (ms):
#       - Find all pre-synaptic excitatory neurons
#       - Find when they fired / when spikes arrive
def find_polygroup(anchors, timing, network, window=10, T=150):
    global N, N_RS, N_FS, M, MAX_DELAY, MAX_STRENGTH
    global SPIKE_THRESHOLD, A_RS, A_FS, B_RS, B_FS, C_RS, C_FS, D_RS, D_FS
    global a, b, c, d

    S = network["S"]
    delays_LUT, delay_sizes = network["delays_LUT"], network["delay_sizes"]
    post_LUT = network["post_LUT"]
    syn_route_LUT = network["syn_route_LUT"]
    pre_LUT, pre_sizes = network["pre_LUT"], network["pre_sizes"]

    v = np.full((N,), C_RS, dtype="float64")
    u = b * v

    I_mat = np.zeros((N, T+MAX_DELAY))      # track thalamus inputs over time
    I_mat[anchors, timing] = 1000           # spike anchors at designated times

    last_fired = np.full((N,), -T)          # all neurons begin at resting

    firings = list()
    group = list()

    # Simulation: run on 1ms resolution for T ms
    for t in range(T):
        # Update dynamics (step 0.5ms twice for v for numerical stability)
        v += 0.5 * (0.04 * np.square(v) + 5 * v + 140 - u + I_mat[:,t])
        v += 0.5 * (0.04 * np.square(v) + 5 * v + 140 - u + I_mat[:,t])
        u += a * (b * v - u)

        spikes = np.where(v >= SPIKE_THRESHOLD)[0]
        new_firings = np.vstack((
            np.full(spikes.shape, t),
            spikes
        )).T
        if new_firings.shape[0] != 0:
            firings.append(new_firings)

        last_fired[spikes] = t

        # Reset dynamics
        v[spikes] = c[spikes]
        u[spikes] += d[spikes]

        for spike in spikes:
            k = syn_route_LUT[spike,:,0]; j = syn_route_LUT[spike,:,1]
            I_mat[k,t+j] += S[spike,:]

            # Find post-synaptic potential arrival times of this spike
            #   for neuron i --> delay (j+1) --> induced spiking neuron k:
            #       arrival_times = [t_last_spike_i + j + 1]
            pre_i = pre_LUT[spike,:pre_sizes[spike],0]
            pre_m = pre_LUT[spike,:pre_sizes[spike],1]
            pre_j = syn_route_LUT[pre_i,pre_m,1]
            pre_spike_times = last_fired[pre_i]
            arrival_times = pre_spike_times + pre_j + 1

            # For each pre-synaptic neuron:
            #   1. Discard neurons with PSP arrival times later than now
            #   2. Discard neurons with PSP arrival times more than 10ms ago
            #   3. Discard neurons with negative synaptic weights
            #   4. Store what's left as a PNG
            rel = np.where((arrival_times < t) & (arrival_times > t - window))[0]
            rel = rel[np.where(S[pre_i[rel],pre_m[rel]] > 0)]

            new_group = np.vstack((
                pre_spike_times[rel].T,     # at these times
                pre_i[rel].T,               # these neurons spiked,
                arrival_times[rel].T,       # arriving at these times
                np.full(rel.shape, spike),  # to this neuron
                np.full(rel.shape, t)       # which spiked now
            )).T
            if new_group.shape[0] != 0:
                group.append(new_group)

    return np.concatenate(group) if len(group) > 0 else np.empty((0,5))


# Given set of polychronous neuron spike traces (ordered) that make up a larger
# group, find the longest (feed-forward) node path length in the group
def find_longest_path_length(polygroup):
    path_lengths = dict()       # path_lengths[node] = max length of path so far
    max_length = 0
    for i in range(polygroup.shape[0]):
        end_node = polygroup[i,3]
        prev_node = polygroup[i,1]

        path_lengths[end_node] = max(
            path_lengths.get(end_node, 0),
            path_lengths.get(prev_node, 0) + 1
        )

        max_length = max(max_length, path_lengths[end_node])

    return max_length



parser = argparse.ArgumentParser(description=("Find polychronous neuron groups "
                                 "given a saved simulation state."))
parser.add_argument("state_file", metavar="FILE", type=str, nargs="?",
                    help="Simulation state to analyze")
parser.add_argument("--save", action="store_true", default=False,
                    help="Save end state of simulation as Numpy object file")
parser.add_argument("-o", metavar="FILE", action="store", type=str,
                    default=None, help="Output file name of end state")
args = parser.parse_args()

print("Loading saved state from {}".format(args.state_file))
state = np.load(args.state_file)

# Network constants (see spnet.py)
N = 1000
N_RS = 800                      # excitatory (regular spiking) neurons
N_FS = 200                      # inhibitory (fast spiking) neurons
M = 100                         # synapses per neuron (randomly selected)

MAX_DELAY = 20                  # max 20ms conduction delay
MAX_STRENGTH = 10               # max synaptic weighting

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

# Analyzer constants
NUM_ANCHORS = 3                 # analyze PNGs with neuron triplets as anchors
MIN_PNG_PATH_LENGTH = 3         # do not analyze small PNGs
MIN_PNG_S = 0.95 * MAX_STRENGTH # weak RS->RS synapses are ignored

# Extract network state
network = dict()
for name in ["S", "delays_LUT", "delay_sizes", "post_LUT", "syn_route_LUT",
        "pre_LUT", "pre_sizes"]:
    network[name] = state[name]

S = network["S"]
delays_LUT, delay_sizes = network["delays_LUT"], network["delay_sizes"]
post_LUT = network["post_LUT"]
syn_route_LUT = network["syn_route_LUT"]
pre_LUT, pre_sizes = network["pre_LUT"], network["pre_sizes"]

# Remove all weak RS->RS synapses
S[(post_LUT < N_RS) & (S > 0) & (S < MIN_PNG_S)] = 0

# Iteratively call find_polygroup on all pre-synaptic, "anchor" neuron triplets
# for each neuron:
#   1. Start with excitatory neuron k
#   2. Find all pre-synaptic neurons i
#   3. Filter out weak synapses (<= MIN_PNG_S), keeping only strong i
#   4. Try combinations of 3 strong neurons, and run a short simulation:
#       - Fire them so that post-synaptic potential arrives at k together
#       - Continue simulation and track all propogated firings
#   5. Resultant firing activity composes a polychronous neuron group
pngs = list()
for i in range(N_RS):
    new_pngs = list()

    pre_i = pre_LUT[i,:pre_sizes[i],0]; pre_m = pre_LUT[i,:pre_sizes[i],1]
    strong_i = pre_i[np.where(S[pre_i,pre_m] >= MIN_PNG_S)[0]]
    strong_m = pre_m[np.where(S[pre_i,pre_m] >= MIN_PNG_S)[0]]

    if len(strong_i) < NUM_ANCHORS:
        continue

    print("Checking neuron {} ({} pre-synaptic neurons) ... ".format(i,
        len(strong_i)), end="", flush=True)

    # Generator of anchor neuron combinations:
    #   [0 1 2], [0 1 3] ... [0 2 3], [0 2 4] ...
    def anchor_sequence(num_anchors, num_choices):
        seq = [_ for _ in range(num_anchors)]
        seq[-1] = seq[-2]

        inc = num_anchors - 1
        while True:
            inc = num_anchors - 1
            while inc >= 0 and seq[inc] == (num_choices - (num_anchors - inc)):
                inc -= 1

            if inc < 0:
                yield None
            else:
                seq[inc] += 1
                for i in range(inc + 1, num_anchors):
                    seq[i] = seq[i-1] + 1
                yield seq

    for seq in anchor_sequence(NUM_ANCHORS, len(strong_i)):
        if seq == None:
            break

        anchors = strong_i[seq]
        timing = MAX_DELAY - syn_route_LUT[anchors,strong_m[seq],1]
        polygroup = find_polygroup(anchors, timing, network)
        max_path_length = find_longest_path_length(polygroup)

        # Ignore small/frivolous PNGs
        if max_path_length < MIN_PNG_PATH_LENGTH:
            continue

        # Ignore false positives, where some anchors do not affect PNG
        #   for anchor neuron i, i must drive at least 2 spikes in PNG
        valid = True
        for anchor in anchors:
            if len(np.where(polygroup[:,1] == anchor)[0]) < 2:
                valid = False
                break
        if not valid:
            continue

        new_pngs.append(PNG(polygroup, anchors, max_path_length))

    print("{} PNGs found.".format(len(new_pngs)), flush=True)
    pngs.extend(new_pngs)

if args.save is True:
    if args.o is None:
        timeinfo = time.gmtime()
        filename = "pngs_{}_{}_{}_{}_{}_{}.pickle".format(*(timeinfo[:6]))
    else:
        filename = args.o

    pickle.dump(pngs, open(filename, "wb"), protocol=pickle.HIGHEST_PROTOCOL)

    print("Simulation state file saved to {}.".format(filename))
