"""
Running outside of gui is more optimized, and allows for faster trial and error.
Recommended workflow is build network outside of GUI first, and then implement it in the GUI for visual intuition.
"""


from architecture import neuron, layer, connection, NEXUS
import random
random.seed(2)
import numpy as np

class network_runner:

    def __init__(self):

        ################  Network vars  ################

        self.layers = []
        self.input_layers = []
        self.hidden_layers = []
        self.output_layers = []
        self.projections = []
        self.network = None

    ################  Layer control  ################

    def add_layer(self, size, name, neuron_type, lay_inhib, inhib_gain, ffi, fbi):
        newLayer = layer.Layer(size=size**2, neuron_type=neuron_type, name=name, lay_inhib=lay_inhib,
                               inhib_gain=inhib_gain, ff=ffi, fb=fbi)

        print("New layer added!")
        self.layers.append(newLayer)
        # Add the layer to correct subset
        if newLayer.neuron_type == "INPUT":
            self.input_layers.append(newLayer)
        elif newLayer.neuron_type == "HIDDEN":
            self.hidden_layers.append(newLayer)
        else:
            assert (newLayer.neuron_type == "OUTPUT")
            self.output_layers.append(newLayer)

    ################  Projection control  ################

    def add_projection(self, from_layer, to_layer):

        for i in self.layers:
            if from_layer == i.name:
                sending_layer = i
            if to_layer == i.name:
                receiving_layer = i

        if sending_layer is not None and receiving_layer is not None:
            newProjection = connection.Projection(sending_layer, receiving_layer)
            self.projections.append(newProjection)
        else:
            print("projection layer arguments error")

    ################  Builder  ################

    def build_network(self):

        print("Layers: ")
        for i in self.layers:
            print(i.name)
        print("Projections: ")
        for i in self.projections:
            print("From_layer: ", i.pre.name, " To_layer: ", i.post.name)
        self.network = NEXUS.Network(layers=self.layers, projections=self.projections)

        return self.network

    ################  Train/test  ################

    # Run one cycle for the network
    def train_network(self, input, output_pattern=None):

        for i in input:
            self.network.set_inputs(i)

        sse = self.network.cycle()
        print('{} sse={}'.format(self.network.cycle_count, sse))

    # Test the network with one additional cycle, and return the resulting activities
    def test_network(self, input_pattern):
        assert len(self.network.layers[0].neurons) == len(input_pattern)

        self.network.cycle()
        return [neuron.act_m for neuron in self.network.layers[-1].neurons]