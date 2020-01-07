"""

Generate one layer of neurons

"""

# What type of neurons this layer contains
INPUT = 0
HIDDEN = 1
OUTPUT = 2


class Neuron:
    def __init__(self, params=None, neuron_type=HIDDEN):
        # What type of neuron this is
        self.neuron_type = neuron_type

        # Set the specifications for this neuron, or if none set it to the default
        self.params = params
        if self.params is None:
            self.params = NeuronParams

        # Call reset to initialize the neuron
        self.reset()

    # Reset the neuron's state. Called at creation and at the end of every cycle
    def reset(self):
        # Reset the excitatory inputs for the next step
        self.excitatory_inputs = []

        # Excitatory conductance
        self.g_e = 0

        # Net current
        self.I_net = 0

        # Membrane potential
        self.V_m = 0

        # Activity of the neuron, a.k.a. the firing rate
        self.act = 0

        # Adaptation current
        self.adapt_curr = 0


# The specifications of the neuron
class NeuronParams:
    def __init__(self, **kwargs):

        ##### Time step constants #####

        # Time step constant for net input update
        self.net_in_dt = 1.0/1.4

        # Time step constant for membrane potential update
        self.vm_dt = 1.0/3.3

        ##### Input channel constants #####

        # Excitatory max conductance
        self.g_bar_e = 1.0

        # Inhibitory max conductance
        self.g_bar_i = 1.0

        # Leak max conductance
        self.g_bar_l = 0.1

        # Leak constant current
        self.g_l = 1.0

        ##### Driving potential #####

        # Excitatory driving potential
        self.e_e = 1.0

        # Inhibitory driving potential
        self.e_i = 0.25

        # TODO: Leak driving potential
        self.e_l = 0.3

        ##### Activation function parameters #####

        # Threshold for activation
        self.act_thr = 0.5

        # Gain factor of the activation function (normalization)
        self.act_gain = 100

        # Clamp ranges
        self.act_min = 0.0
        self.act_max = 0.95

        # Make sure any entered keyword arguments correspond to existing parameters
        for key, value in kwargs.items():
            assert hasattr(self, key), 'the {} parameter does not exist'.format(key)
            setattr(self, key, value)