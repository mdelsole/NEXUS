from architecture import neuron, layer, connection, NEXUS
import random
random.seed(0)
import numpy as np

class network_runner:

    def __init__(self):

        ################  Network vars  ################

        self.layers = []
        self.input_layers = []
        self.hidden_layers = []
        self.output_layers = []
        self.projections = []


    ################  Layer control  ################


    def addLayer(self, size, name, neuron_type, lay_inhib, inhib_gain, ffi, fbi):
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
                assert(newLayer.neuron_type == "OUTPUT")
                self.output_layers.append(newLayer)

        shape = int(size**(1/2))

        # Square array of the layer's neurons' activities
        activities = np.reshape([neuron.act_m for neuron in newLayer.neurons], (-1, shape))

        return activities, name, new

    ################  Projection control  ################



    ################  Builder  ################
    """
    
    def build_network(self, network_layers):
    
        projections = []
        for i in range(len(hidden_layers)):
            hidden_conn = connection.Projection(layers[-1],  hidden_layer)
            layers.append(hidden_layer)
            projections.append(hidden_conn)
    
        last_conn  = connection.Projection(layers[-1],  output_layer)
        projections.append(last_conn)
        layers.append(output_layer)
    
        network = NEXUS.Network(layers=layers, projections=projections)
    
        return network
    
    
    ################  Train/test  ################
    
    
    # Test the network with one additional cycle
    def test_network(network, input_pattern):
        assert len(network.layers[0].neurons) == len(input_pattern)
        network.set_inputs({'input_layer': input_pattern})
    
        network.cycle()
        return [neuron.act_m for neuron in network.layers[-1].neurons]
    
    
    # Run one cycle for the network
    def train_network(network, input_pattern, output_pattern):
        network.set_inputs({'input_layer': input_pattern})
        network.set_outputs({'output_layer': output_pattern})
    
        sse = network.cycle()
        print('{} sse={}'.format(network.cycle_count, sse))
        return [neuron.act_m for neuron in network.layers[-1].neurons]
    
    
    def run_test(size):
    
    
        network = build_network(size, size, 2)
        horizontal = 10*[0.0] + 5*[1.0] + 10*[0.0]
        vertical   = 5*[0.0, 0.0, 1.0, 0.0, 0.0]
        leftdiag   = [1.0, 0.0, 0.0, 0.0, 0.0,
                       0.0, 1.0, 0.0, 0.0, 0.0,
                       0.0, 0.0, 1.0, 0.0, 0.0,
                       0.0, 0.0, 0.0, 1.0, 0.0,
                       0.0, 0.0, 0.0, 0.0, 1.0]
        rightdiag  = [0.0, 0.0, 0.0, 0.0, 1.0,
                       0.0, 0.0, 0.0, 1.0, 0.0,
                       0.0, 0.0, 1.0, 0.0, 0.0,
                       0.0, 1.0, 0.0, 0.0, 0.0,
                       1.0, 0.0, 0.0, 0.0, 0.0]
    
        for i in range(5):
            train_network(network, horizontal, horizontal)
            train_network(network, vertical, vertical)
            train_network(network, leftdiag, leftdiag)
            train_network(network, rightdiag, rightdiag)
    """