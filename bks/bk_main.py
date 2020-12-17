import pdb
from tqdm import tqdm
import math
import torch
import random
from pprint import pprint
import matplotlib.pyplot as plt
from bindsnet.learning import PostPre
from bindsnet.network import Network
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.topology import Connection
from bindsnet.network.monitors import Monitor
from bindsnet.analysis.plotting import plot_spikes, plot_voltages

CONFIG = dict()
CONFIG["inp_dim"] = 100
CONFIG["layer_dims"] = [70, 35, 12, 1]


class Composer_SNN:
    def __init__(self, inp_dim, layer_dims, update_rule, time=500, out_inhb=True):
        ## Instantiating variables
        self.time = time
        # Create a network
        self.network = Network()

        ## Creating layers
        self.layers = list()
        # Input layer
        self.layers.append(Input(n=inp_dim, traces=True))
        # Hidden layers (The last hidden layer is the output layer as well)
        for layer in range(len(layer_dims)):
            self.layers.append(LIFNodes(n=layer_dims[layer], traces=True))

        # Adding layers
        for layer in range(len(self.layers)):
            self.network.add_layer(
                layer=self.layers[layer], name=str(layer)
            )

        ## Creating the connections between the layers
        self.connections = list()
        for layer in range(len(self.layers) - 1):
            # pdb.set_trace()
            self.connections.append(Connection(
                source=self.layers[layer],
                target=self.layers[layer + 1],
                w=0.05 + 0.5 * torch.randn(self.layers[layer].n, self.layers[layer + 1].n),
                # update_rule=update_rule, nu=(1e-6, 1e-5),
            ))

        # Adding output inhibition
        # if out_inhb:
        #     for layer in range(len(self.layers)):
        #         self.connections.append(Connection(
        #             source=self.layers[layer],
        #             target=self.layers[layer],
        #             w=0.025 * (torch.eye(self.layers[layer].n) - 1),
        #         ))

        # Adding connections to the network instance
        for layer in range(len(self.connections)):
            # if layer == len(self.connections) - 1:
            #     self.network.add_connection(
            #         connection=self.connections[layer], source=str(layer), target=str(layer)
            #     )
            # else:
            self.network.add_connection(
                connection=self.connections[layer], source=str(layer), target=str(layer + 1)
            )

        ## Creating monitors
        self.monitors = list()
        for layer in range(len(self.layers)):
            if layer == 0:
                self.monitors.append(Monitor(
                    obj=self.layers[layer],
                    state_vars=("s",),  # Record spikes.
                    time=self.time,  # Length of simulation (if known ahead of time).
                ))
            else:
                self.monitors.append(Monitor(
                    obj=self.layers[layer],
                    state_vars=("s", "v"),  # Record spikes and voltages.
                    time=self.time,  # Length of simulation (if known ahead of time).
                ))
            # Add the monitor
            self.network.add_monitor(monitor=self.monitors[layer], name=str(layer))

    def run(self, inputs):
        self.network.run(inputs=inputs, time=self.time)


def main():
    # Creating input data
    input_data = torch.bernoulli(0.1 * torch.ones(500, CONFIG["inp_dim"])).byte()
    label_data = torch.cat((torch.zeros(250, 1), torch.ones(250, 1)), 1)
    data_indexes = [i for i in range(500)]

    random.shuffle(data_indexes)

    #  mask
    mask1 = torch.cat(
        (torch.zeros(250, math.floor(CONFIG["inp_dim"] / 2)), torch.ones(250, math.floor(CONFIG["inp_dim"] / 2))), 1)
    mask2 = torch.cat(
        (torch.ones(250, math.floor(CONFIG["inp_dim"] / 2)), torch.zeros(250, math.floor(CONFIG["inp_dim"] / 2))), 1)
    mask = torch.cat((mask1, mask2), 0)

    # Inputs
    input_data = torch.mul(input_data, mask)
    input_data = input_data[torch.randperm(input_data.size()[0])]

    inputs = {"0": input_data}

    # Create model
    model = Composer_SNN(inp_dim=CONFIG["inp_dim"], layer_dims=CONFIG["layer_dims"], update_rule=PostPre)

    pdb.set_trace()

    for epoch in tqdm(range(5)):
        # run model
        model.run(inputs)

        # Obtaining plots for spike plots.
        spikes = dict()
        for monitor in range(len(model.monitors)):
            spikes[str(monitor)] = model.monitors[monitor].get("s")

        # Obtaining voltage plot.
        voltages = {str(len(CONFIG["layer_dims"]) - 1): model.monitors[len(CONFIG["layer_dims"]) - 1].get("v")}

        # plotting the data
        plt.ioff()
        plot_spikes(spikes)
        plt.savefig(f"./images/spikes_{epoch}.png")
        plt.clf()

        # model.network.reset_state_variables()

        # plot_voltages(voltages, plot_type="line")
        # plt.savefig(f"voltages.png")


if __name__ == "__main__":
    main()


