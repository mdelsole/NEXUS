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

        # TODO: Net current, equilibrium version
        self.I_net_r = self.I_net

        # Membrane potential
        self.v_m = self.params.v_m_init

        # Membrane potential
        self.V_m = 0

        # Activity of the neuron, a.k.a. the firing rate
        self.act = 0

        # Adaptation current
        self.adapt_curr = 0


# The specifications of the neuron
class NeuronParams:
    def __init__(self, **kwargs):

        ########### Time step constants ###########

        # Time step constant for net input update
        self.net_in_dt = 1.0/1.4

        # Time step constant for membrane potential update
        self.vm_dt = 1.0/3.3

        ########### Input channel constants ###########

        # Excitatory max conductance
        self.g_bar_e = 1.0

        # Inhibitory max conductance
        self.g_bar_i = 1.0

        # Leak max conductance
        self.g_bar_l = 0.1

        # Leak constant current
        self.g_l = 1.0

        ########### Driving potential constants ###########

        # Excitatory driving potential
        self.e_e = 1.0

        # Inhibitory driving potential
        self.e_i = 0.25

        # Leak driving potential
        self.e_l = 0.3

        ########### Activation function parameters ###########

        # Threshold for activation
        self.act_thr = 0.5

        # Gain factor of the activation function (for normalization)
        self.act_gain = 100

        # Clamp ranges; NXX1 can't reach 1, so clamp to 0.95
        self.act_min = 0.0
        self.act_max = 0.95

        ########### Spike behavior ###########

        # Normalized spike threshold for resetting v_m
        self.spk_thr = 1.2

        # Initial value for v_m
        self.v_m_init = 0.4

        # Value to reset v_m to after firing
        self.v_m_r = 0.3

        # TODO: Clamp?

        ########### Adaptation current ###########

        # Time step constant for adaptation current
        self.adapt_dt = 1.0/144.0

        # TODO: Gain that voltage produces on adaptation
        self.vm_gain = 0.04


        # If the table has been precomputed
        self.nnx1_table = None

        # Make sure any entered keyword arguments correspond to existing parameters
        for key, value in kwargs.items():
            assert hasattr(self, key), 'the {} parameter does not exist'.format(key)
            setattr(self, key, value)

    # Calculate the xx1 function
    def xx1(self, v_m):
        # TODO: g_e_theta instead of v_m?
        x = self.act_gain * max(v_m, 0.0)
        return x / (x + 1)

    # Pre-compute the convolution for the noisy xx1 function as a look-up table
    def nxx1(self):

        # If we have not precomputed the table yet
        if self.nnx1_table is None:
            # Resolution of the precomputed array
            resolution = 0.001

    def calculate_net_input(self):
        pass