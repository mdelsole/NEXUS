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
        newLayer = layer.Layer(size=size, neuron_type=neuron_type, name=name, lay_inhib=lay_inhib,
                               inhib_gain=inhib_gain, ffi=ffi, fbi=fbi)

        new = True
        # Check to see that this is an unique layer
        for i in self.layers:
            if i.name == newLayer.name:
                print("Layer already exists, updating")
                # TODO: Update
                new = False

        # If it's a unique layer, add it to the list of this network's layers
        if new:
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

        shape = int(size ** (1 / 2))

        # Square array of the layer's neurons' activities
        activities = np.reshape([neuron.act_m for neuron in newLayer.neurons], (-1, shape))

        return activities, name, new

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

    # Test the network with one additional cycle
    def test_network(self, input_pattern):
        assert len(self.network.layers[0].neurons) == len(input_pattern)
        self.network.set_inputs({'input_layer': input_pattern})

        self.network.cycle()
        return [neuron.act_m for neuron in self.network.layers[-1].neurons]

    # Run one cycle for the network
    def train_network(self, input_pattern, output_pattern):

        # Force the activity of the inputs and outputs
        self.network.set_inputs({self.input_layers[0].name: input_pattern})
        self.network.set_outputs({self.output_layers[0].name: output_pattern})

        sse = self.network.cycle()
        print('{} sse={}'.format(self.network.cycle_count, sse))

        # Th activities of neurons of the layers
        activities = [neuron.act_m for neuron in self.network.layers[-1].neurons]

        # Reshape into square 2d matrix for displaying
        size = np.size(activities)
        shape = int(size ** (1 / 2))
        display = np.reshape(activities, (-1, shape))

        return display

