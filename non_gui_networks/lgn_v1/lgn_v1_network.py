from non_gui_networks import non_gui_network_runner
from non_gui_networks.lgn_v1 import lgn_v1_input_loader

# Create a new network
new_network = non_gui_network_runner.network_runner()


################  Layers  ################

# LGN_ON: Size: 12x12, Projections: none
LGN_ON = new_network.add_layer(size=12, neuron_type="INPUT", name="LGN_ON", lay_inhib=True,
                               inhib_gain=2.0, ffi=1.0, fbi=1.0)

# LGN_OFF: Size: 12x12, Projections: none
LGN_OFF = new_network.add_layer(size=12, neuron_type="INPUT", name="LGN_OFF", lay_inhib=True,
                                inhib_gain=2.0, ffi=1.0, fbi=1.0)

# V1: Size: 14x14, Projections: Fm_LGN_ON, Fm_LGN_OFF, Fm_V1 (recurrent), inhib_fm_v1
V1 = new_network.add_layer(size=14, neuron_type="HIDDEN", name="V1", lay_inhib=True,
                           inhib_gain=2.0, ffi=1.0, fbi=1.0)

################  Projections  ################

new_network.add_projection("LGN_ON", "V1")
new_network.add_projection("LGN_OFF", "V1")
# Recurrent Connection

# Lateral inhibition
# Tessel circle (lateral inhibition, in a circle that fades as a function of distance from the center)

################  Build  ################

new_network.build_network()

################  Load Input and Train  ################

epochs = 5
for i in range(epochs):
    # Load degree of gaussians image fragment
    image, inverse = lgn_v1_input_loader.degree_of_gaussians()

    # Force activity of LGN with train_network
    input_LGN_ON = {'LGN_ON': image}
    input_LGN_OFF = {'LGN_OFF': inverse}
    input_patterns = [input_LGN_ON, input_LGN_OFF]

    # Train the network
    new_network.train_network(input=input_patterns)


################  Test  ################

# Grab receiving weights of V1 from LGN_ON