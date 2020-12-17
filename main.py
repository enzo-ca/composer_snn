import pdb
import math
import torch
import random
import numpy as np
from tqdm import tqdm
from pprint import pprint
from dataprep import process_song
import matplotlib.pyplot as plt
from bindsnet.learning import PostPre
from bindsnet.network import Network
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.topology import Connection
from bindsnet.network.monitors import Monitor
from bindsnet.analysis.plotting import plot_spikes, plot_voltages

import os
import glob
from tqdm import tqdm
import math

CONFIG = dict()
CONFIG["inp_dim"] = 88
CONFIG["radius"] = 12
CONFIG["layer_dims"] = [CONFIG["inp_dim"],
                        sum([CONFIG["inp_dim"] - i - 1 for i in range(CONFIG["radius"])]),
                        CONFIG["radius"],
                        1]

CONFIG["time"] = 300
CONFIG["dt"] = 5
CONFIG["spike_time_window"] = 100
CONFIG["layers_w"] = (100, 1.5, 100, 20)
CONFIG["self_excitatory_w"] = 1
CONFIG["inhibition_w"] = 10
CONFIG["tc_decay"] = 100
CONFIG["refrac"] = 5

# CONFIG["time"] = 300
# CONFIG["dt"] = 5
# CONFIG["spike_time_window"] = 100
# CONFIG["layers_w"] = (100, 1.5, 100, 20)
# CONFIG["self_excitatory_w"] = 10
# CONFIG["inhibition_w"] = 1


seed = 0
gpu = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if gpu and torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
else:
    torch.manual_seed(seed)
    device = "cpu"
    if gpu:
        gpu = False

torch.set_num_threads(os.cpu_count() - 1)
print("Running on Device = ", device)


class Composer_SNN:
    def __init__(self, inp_dim, layer_dims, update_rule, out_inhb=True, dt=4):
        # Create a network
        self.network = Network(dt=dt, learning=True)

        ## Creating layers
        self.layers = list()
        # Input layer
        self.layers.append(Input(n=inp_dim, traces=True))
        # Hidden layers (The last hidden layer is the output layer as well)
        for layer in range(len(layer_dims)):
            # Modify the time constant decay value for this layers
            if layer in [0]:
                self.layers.append(LIFNodes(n=layer_dims[layer], traces=True, tc_decay=CONFIG["tc_decay"], refrac=CONFIG["refrac"]))
            # sum the inputs of the following layers
            elif layer in [1]:
                self.layers.append(LIFNodes(n=layer_dims[layer], traces=True, sum_input=True, refrac=CONFIG["refrac"]))
            else:
                self.layers.append(LIFNodes(n=layer_dims[layer], traces=True))

        # Adding layers
        for layer in range(len(self.layers)):
            self.network.add_layer(
                layer=self.layers[layer], name=str(layer)
            )

        ## Creating the connections between the layers
        self.connections = list()
        # Connection between layers
        for layer in range(len(self.layers) - 1):
            if layer == 0:
                # Using fixed weight between layers 0 and 1
                self.connections.append(Connection(
                    source=self.layers[layer],
                    target=self.layers[layer + 1],
                    w=CONFIG["layers_w"][0] * torch.ones(self.layers[layer].n, self.layers[layer + 1].n),
                ))
            elif layer == 1:
                # Using fixed weight between layers 1 and 2
                self.connections.append(Connection(
                    source=self.layers[layer],
                    target=self.layers[layer + 1],
                    w=CONFIG["layers_w"][1] * torch.ones(self.layers[layer].n, self.layers[layer + 1].n),
                ))
            elif layer == len(self.layers) - 2:
                self.connections.append(Connection(
                    source=self.layers[layer],
                    target=self.layers[layer + 1],
                    w=CONFIG["layers_w"][3] * torch.ones(self.layers[layer].n, self.layers[layer + 1].n),
                    update_rule=update_rule, nu=(1e-4, 1e-2)
                ))
            else:
                self.connections.append(Connection(
                    source=self.layers[layer],
                    target=self.layers[layer + 1],
                    w=CONFIG["layers_w"][2] * torch.ones(self.layers[layer].n, self.layers[layer + 1].n),
                ))
        # inhibition weights and Self-exitatory weights in layer 1
        self.connections.append(Connection(
            source=self.layers[1],
            target=self.layers[1],
            w=(torch.eye(self.layers[1].n) - CONFIG["inhibition_w"]) +
              CONFIG["self_excitatory_w"] * torch.eye(self.layers[1].n)
        ))

        # Adding connections to the network instance
        for layer in range(len(self.connections) - 1):
            self.network.add_connection(
                connection=self.connections[layer], source=str(layer), target=str(layer + 1)
            )
        # Adding inhibition connections to the network
        self.network.add_connection(
            connection=self.connections[len(self.connections) - 1], source='1', target='1'
        )

        # [Layers 0-1] - 1-by-1 connections between neurons in layers 0 and 1
        w0_1 = torch.zeros((CONFIG["inp_dim"], CONFIG["layer_dims"][0]))
        for i in range(CONFIG["layer_dims"][0]):
            w0_1[i][i] = 1
        self.network.connections['0', '1'].w *= w0_1
        del w0_1
        # [Layers 1-2] - Connect layers 1 and 2 using a combination (N1 2)
        w1_2 = torch.zeros((CONFIG["layer_dims"][0], CONFIG["layer_dims"][1]))
        lay2_neu = 0
        step = 1
        while step <= CONFIG["radius"]:
            for i in range(CONFIG["layer_dims"][0] - step):
                w1_2.T[lay2_neu][i] = 1
                w1_2.T[lay2_neu][i + step] = 1
                lay2_neu += 1
            step += 1
        self.network.connections['1', '2'].w *= w1_2
        del w1_2
        # [Layers 2-3] - Connect layers 2 and 3 using
        w2_3 = torch.zeros((CONFIG["layer_dims"][1], CONFIG["layer_dims"][2]))
        last_qty_w = 0
        curr_layer = 0
        for i in range(CONFIG["radius"]):
            curr_qty_w = CONFIG["layer_dims"][0] - (i + 1)
            w2_3.T[curr_layer][last_qty_w:last_qty_w + curr_qty_w] = 1
            curr_layer += 1
            last_qty_w += curr_qty_w
        self.network.connections['2', '3'].w *= w2_3
        del w2_3

    def set_monitors(self, time):
        self.time = time
        ## Creating monitors
        self.monitors = list()
        for layer in range(len(self.layers)):
            if layer == 0:
                continue
                # self.monitors.append(Monitor(
                #     obj=self.layers[layer],
                #     state_vars=("s",),  # Record spikes.
                #     time=self.time,  # Length of simulation (if known ahead of time).
                # ))
            else:
                self.monitors.append(Monitor(
                    obj=self.layers[layer],
                    state_vars=("s", "v"),  # Record spikes and voltages.
                    time=self.time,  # Length of simulation (if known ahead of time).
                ))
            # Add the monitor
            self.network.add_monitor(monitor=self.monitors[layer-1], name=str(layer-1))

    def run(self, inputs, time=CONFIG["time"]):
        self.network.run(inputs=inputs, time=time)


def main():
    ## Input sample test
    # spw = math.floor(CONFIG["spike_time_window"] / CONFIG["dt"])
    # dim = CONFIG["inp_dim"]
    # N = dim * spw
    # mel_int = 5
    # all_songs = list()
    # all_songs.append(torch.zeros(N, dim))
    # curr_input = 0
    # for i in range(N):
    #     if i != 0 and i % spw == 0:
    #         curr_input = mel_int + curr_input if mel_int + curr_input <= (dim - 1) else 0
    #     all_songs[0][i][curr_input] = 1

    ############################################################################################
    ## MIDI Data
    # Bach Data
    path = r"/Users/amadopena/PycharmProjects/BIC/hw3/finalproject/midis/Classical Piano Midis/Bach"
    max_files_to_process = 100
    all_songs = []
    for file in tqdm(glob.glob(os.path.join(path, "*.mid"))[:max_files_to_process]):
        str_name = os.path.basename(os.path.normpath(file))
        print("\nProcessing {}".format(str_name))
        song_samples = process_song(file, spike_frequency=math.floor(CONFIG["spike_time_window"]/CONFIG["dt"]))
        all_songs.append(song_samples)
        print("="*30)
    labels = torch.cat((torch.zeros(len(all_songs)), torch.ones(len(all_songs))))

    # Beethoven Data
    path = r"/Users/amadopena/PycharmProjects/BIC/hw3/finalproject/midis/Classical Piano Midis/Beethoven"
    max_files_to_process = len(all_songs)
    for file in tqdm(glob.glob(os.path.join(path, "*.mid"))[:max_files_to_process]):
        str_name = os.path.basename(os.path.normpath(file))
        print("\nProcessing {}".format(str_name))
        song_samples = process_song(file, spike_frequency=math.floor(CONFIG["spike_time_window"]/CONFIG["dt"]))
        all_songs.append(song_samples)
        print("="*30)
    ############################################################################################

    # Create model
    model = Composer_SNN(inp_dim=CONFIG["inp_dim"], layer_dims=CONFIG["layer_dims"], update_rule=PostPre,
                         dt=CONFIG["dt"])

    # Using GPU
    if gpu:
        model.network.to("cuda")


    for song_id in tqdm(range(len(all_songs))):
        # Obtain inputs
        if gpu:
            inputs = {"0": torch.tensor(all_songs[song_id]).cuda()}
        else:
            inputs = {"0": torch.tensor(all_songs[song_id])}

        # Calculate the simulation time for the current song
        CONFIG["time"] = inputs["0"].shape[0]*CONFIG["dt"]
        # CONFIG["time"] = 2000

        # Define the monitors
        model.set_monitors(CONFIG["time"])

        # Run the model for training and testing
        for phase in range(2):
            # Run training
            if phase == 0:
                # Activate learning
                model.network.learning = True

                # run model
                model.run(inputs, time=CONFIG["time"])

                model.network.reset_state_variables()

            # Run testing
            else:
                # Deactivate learning
                model.network.learning = False

                # run model
                model.run(inputs, time=CONFIG["time"])

        # Obtaining plots for spike plots.
        spikes = dict()
        for monitor in range(len(model.monitors)):
            spikes[str(monitor)] = model.monitors[monitor].get("s")

        if int(labels[song_id]) == 0:
            print(f"Bach: {sum(spikes['3'])}")
        else:
            print(f"Beethoven: {sum(spikes['3'])}")

        # Obtaining voltage plot.
        voltages = {str(len(CONFIG["layer_dims"]) - 1): model.monitors[len(CONFIG["layer_dims"]) - 1].get("v")}

        # plotting the data
        plt.ioff()
        plot_spikes(spikes)
        plt.savefig(f"./images/spikes_{song_id}.png")
        plt.clf()
        #
        # plot_voltages(voltages, plot_type="line")
        # plt.savefig(f"voltages.png")

        model.network.reset_state_variables()

    # pdb.set_trace()


if __name__ == "__main__":
    main()


