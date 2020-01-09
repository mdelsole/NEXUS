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
    def __init__(self, neuron_type=HIDDEN,
                 log_names=('net_in', 'I_net', 'v_m', 'act', 'v_m_eq', 'adapt_curr'), **kwargs):

        # What type of neuron this is
        self.neuron_type = neuron_type

        ########### Time step constants ###########

        # Time step constant for net input update
        self.net_in_dt = 1.0 / 1.4
        # Time step constant for membrane potential update
        self.v_m_dt = 1.0 / 3.3
        # Time step constant for integration. 1 = 1 msec
        self.integ_dt = 1

        ########### Input channel constants ###########

        # Excitatory max conductance
        self.g_bar_e = 1.0

        # Inhibitory max conductance
        self.g_bar_i = 1.0

        # Leak max conductance
        self.g_bar_l = 0.1

        # Leak constant current
        self.g_l = 1.0

        self.g_e = 0
        self.net_in = self.g_bar_e * self.g_e

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

        # TODO: Gain that voltage produces, driving the adaptation current
        self.v_m_gain = 0.04

        # Value the adaptation current gains after spiking
        self.spike_gain = 0.00805

        # If the table has been precomputed
        self.nnx1_table = None

        # Make sure any entered keyword arguments correspond to existing parameters
        for key, value in kwargs.items():
            assert hasattr(self, key), 'the {} parameter does not exist'.format(key)
            setattr(self, key, value)

        # Logs for testing/visualization
        self.log_names = log_names
        self.logs = {name: [] for name in self.log_names}

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

        # TODO: Net current, equilibrium version. Rate-coded version

        self.I_net_r = self.I_net

        # Membrane potential
        self.v_m = self.v_m_init

        # Equilibrium membrane potential (the value that is settled at)
        self.v_m_eq = self.v_m

        # Activity of the neuron, a.k.a. the firing rate
        self.act = 0

        # Adaptation current
        self.adapt_curr = 0


    ################  Utility functions  ################


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


    ################  Neuron "integrate and fire" functions  ################


    # Calculate net excitatory input. Execute before every step
    def calculate_net_input(self):

        # Total instantaneous excitatory input to the neuron
        net_raw_input = 0.0

        if len(self.excitatory_inputs) > 0:
            # Total input = sum of excitatory inputs in this current step
            net_raw_input = sum(self.excitatory_inputs)

            # Clear the excitatory inputs for the next step
            self.excitatory_inputs = []

        # Update g_e
        # TODO: net_input?
        self.g_e = self.integ_dt * self.net_in_dt * (net_raw_input - self.g_e)

    # One step of the neuron. Update activity of the neuron
    def step(self, phase, g_i=0.0, dt_integ=1):

        # Calculate net current
        self.I_net = self.calculate_net_current(g_i, steps=2)
        # TODO: Rate-code version of I_net, to provide adequate coupling with v_m_eq
        self.I_net_r = self.calculate_net_current(g_i, steps=1)

        # Update v_m (membrane potential) and v_m_eq (equilibrium membrane potential)
        self.v_m += self.integ_dt * self.v_m_dt * self.I_net
        self.v_m_eq += self.integ_dt * self.v_m_dt * self.I_net_r

        # Reset v_m if it crosses the threshold
        if self.v_m > self.act_thr:
            self.did_spike = 1

            # Reset v_m to the reset value
            self.v_m = self.v_m_r
            self.I_net = 0.0
        else:
            self.did_spike = 0

        # Compute new_act from v_m_eq, b/c rate-coded
        if self.v_m_eq <= self.act_thr:

            # Activity = nnx1(how close v_m_eq is to threshold for activation)
            activity = self.nxx1(self.v_m_eq-self.act_thr)
        else:

            # Find g_e_theta, the level of excitatory input conductance that would make v_m_eq = act_thr
            gc_e = self.g_bar_e * self.g_e
            gc_i = self.g_bar_i * g_i
            gc_l = self.g_bar_l * self.g_l
            # Theta for g_e; the level g_e has to reach for the neuron to fire
            g_e_threshold = (gc_i * (self.e_i - self.act_thr) + gc_l * (self.e_l - self.act_thr)
                       - self.adapt_curr) / (self.act_thr - self.e_e)

            # Neuron's activation is the result of nnx1(g_e - g_e_threshold)
            activity = self.nxx1(gc_e - g_e_threshold)

        # Update activity
        self.act += self.integ_dt * self.v_m_dt * (activity - self.act)

        # Update adaptation
        self.adapt_curr += self.integ_dt * (self.adapt_dt * (self.v_m_gain * (self.v_m - self.e_l) - self.adapt_curr)
                + self.did_spike * self.spike_gain)

        # Update logs
        self.update_logs()


    # Calculate net current, factoring in inhibition + leak. Will be called from within step
    def calculate_net_current(self, g_i, steps=1):

        # Exctitatory conductance
        gc_e = self.g_bar_e * self.g_e
        # Inhibitory conductance. g_i will be input gotten from the layer handler class
        gc_i = self.g_bar_i * g_i
        # Leak conductance
        gc_l = self.g_bar_l * self.g_l

        v_m_eff = self.v_m_eq

        net_curr = 0.0
        # Iterative approach
        for _ in range(steps):
            net_curr = (gc_e * (self.e_e - v_m_eff)
                     + gc_i * (self.e_i - v_m_eff)
                     + gc_l * (self.e_l - v_m_eff)
                     - self.adapt_curr)

            v_m_eff += self.integ_dt/steps * self.v_m_dt * net_curr

        return net_curr


    ################  Logs/config  ################

    # Update our logs with the current state after each step
    def update_logs(self):
        for name in self.logs.keys():
            self.logs[name].append(getattr(self, name))

    # Display variable/constant values
    def show_config(self):
        print('Parameters:')
        for name in ['v_m_dt', 'net_in_dt', 'g_l', 'g_bar_e', 'g_bar_l', 'g_bar_i',
                     'e_e', 'e_l', 'e_i', 'act_thr', 'act_gain']:
            print('   {}: {:.2f}'.format(name, getattr(self, name)))
        print('State:')
        for name in ['g_e', 'I_net', 'v_m', 'act', 'v_m_eq']:
            print('   {}: {:.2f}'.format(name, getattr(self, name)))
