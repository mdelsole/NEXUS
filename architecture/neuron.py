"""

Generate one layer of neurons

"""

import numpy as np
import scipy.interpolate

# What type of neurons this layer contains
INPUT = 0
HIDDEN = 1
OUTPUT = 2


class Neuron:
    def __init__(self, params=None, neuron_type=HIDDEN, **kwargs):

        # What type of neuron this is
        self.neuron_type = neuron_type

        # Call reset to initialize the neuron
        self.reset()

        ########### Time step constants ###########

        # Time step constant for net input update
        self.net_in_dt = 1.0 / 1.4

        # Time step constant for membrane potential update
        self.vm_dt = 1.0 / 3.3

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

        # Standard deviation for computing the noisy gaussian
        self.act_sd = 0.01

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
        self.adapt_dt = 1.0 / 144.0

        # TODO: Gain that voltage produces on adaptation
        self.vm_gain = 0.04

        # If the table has been precomputed
        self.nnx1_table = None

        # Make sure any entered keyword arguments correspond to existing parameters
        for key, value in kwargs.items():
            assert hasattr(self, key), 'the {} parameter does not exist'.format(key)
            setattr(self, key, value)

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
        self.v_m = self.v_m_init

        # Membrane potential (equilibrium version)
        self.v_m_eq = self.v_m

        # Activity of the neuron, a.k.a. the firing rate
        self.act = 0

        # Adaptation current
        self.adapt_curr = 0

    # Add an input for the next step
    def add_excitatory_inputs(self, input_act):
        self.excitatory_inputs.append(input_act)

    # Pre-compute the convolution for the noisy xx1 function as a look-up table
    def nxx1(self, v_m):

        # If we have not precomputed the table yet
        if self.nnx1_table is None:

            # Resolution of the precomputed array
            resolution = 0.001

            # Compute the gaussian
            ns_rng = max(3.0 * self.act_sd, resolution)
            # x represents v_m
            xs = np.arange(-ns_rng, ns_rng + resolution, resolution)
            var = max(self.act_sd, 1.0e-6) ** 2
            # Compute unscaled gaussian
            gaussian = np.exp(-xs ** 2 / var)
            # Normalize
            gaussian = gaussian / sum(gaussian)

            # Compute the xx1 function
            xs = np.arange(-2 * ns_rng, 1.0 + ns_rng + resolution, resolution)
            x2 = self.act_gain * np.maximum(xs, 0)
            xx1 = x2 / (x2 + 1)

            # Convolution
            conv = np.convolve(xx1, gaussian, mode='same')

            # Clamp to valid range
            xs_valid = np.arange(-ns_rng, 1.0 + resolution, resolution)
            conv = conv[np.searchsorted(xs, xs_valid[0], side='left'):
                        np.searchsorted(xs, xs_valid[-1], side='right')]
            assert len(xs_valid) == len(conv), '{} != {}'.format(len(xs_valid), len(conv))

            self.nxx1_table = xs_valid, conv

        xs, conv = self.nxx1_table
        if v_m < xs[0]:
            return 0.0
        elif xs[-1] < v_m:
            # Calculate the xx1 function
            # TODO: g_e_theta instead of v_m?
            x = self.act_gain * max(v_m, 0.0)
            return x / (x + 1)
        else:
            return float(scipy.interpolate.interp1d(xs, conv, kind='linear', fill_value='extrapolate')(v_m))

    # Calculate net input. Execute before every step
    def calculate_net_input(self):
        pass