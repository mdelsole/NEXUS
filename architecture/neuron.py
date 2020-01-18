"""

Generate one area of neurons

"""

import numpy as np
import scipy.interpolate

# What type of neuron this is
INPUT = 0
HIDDEN = 1
OUTPUT = 2


class Neuron:

    def __init__(self, neuron_type=HIDDEN, log_names=('net_input', 'I_net', 'v_m', 'act', 'v_m_eq', 'adapt_curr'),
                 **kwargs):

        # What type of neuron this is
        self.neuron_type = neuron_type

        ########### Time step constants ###########

        # Time step constant for net input integration
        self.net_input_dt = 1/1.4
        # Time step constant for membrane potential update
        self.v_m_dt = 1/3.3
        # Time step constant for integration. 1 = 1 msec
        self.integ_dt = 1

        ########### Input channel parameters ###########

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

        # TODO: Clamp ranges; NXX1 can't reach 1, so clamp to 0.95
        self.act_min = 0.0
        self.act_max = 0.95

        ########### Spike behavior ###########

        # Normalized spike threshold for resetting v_m
        self.spk_thr = 1.2
        # Initial value for v_m
        self.v_m_init = 0.4
        # Value to reset v_m to after firing
        self.v_m_r = 0.3

        ########### Adaptation current ###########

        # Should be true always for most accurate results, but a bit more expensive
        self.adapt_on = True

        # Time step constant for adaptation current
        self.adapt_dt = 1.0 / 144.0
        # TODO: Gain that voltage produces
        self.v_m_gain = 0.04
        # Value that the adaptation current gains after the neuron spikes
        self.spike_gain = 0.00805

        # TODO: Bias?

        ########### Averages for learning ###########

        self.avg_init = 0.15
        self.avg_l_init = 0.4

        # To compute XCAL from the neuron, update these (cascaded) average variables
        # Base average to compute avg_s over
        self.avg_ss = self.avg_init
        # Short-term average synaptic activity
        self.avg_s = self.avg_init
        # Medium-term average synaptic activity
        self.avg_m = self.avg_init
        # Long-term average synaptic activity
        self.avg_l = self.avg_l_init

        # Rate constants for updating the running averages
        self.avg_ss_dt = 0.5
        self.avg_s_dt = 0.5
        self.avg_m_dt = 0.1

        # Computed once every cycle. Thus the full integration (reaching 1) will be complete after 10 cycles
        self.avg_l_dt = 0.1
        self.avg_l_min = 0.2
        self.avg_l_gain = 2.5

        self.avg_m_in_s = 0.1

        # TODO: Min/max for avg_l_lrn
        self.avg_lrn_min = 0.0001
        self.avg_lrn_max = 0.5

        # Linear mixing of avg_s and avg_m
        self.avg_s_eff = 0.0

        ########### Utility ###########

        for key, value in kwargs.items():
            assert hasattr(self, key), 'the {} parameter does not exist'.format(key)
            setattr(self, key, value)

        # If the convolution table for nxx1 has been precomputed
        self.nxx1_table = None

        self.log_names = log_names
        self.logs = {name: [] for name in self.log_names}

        # Reset to initialize the neuron
        self.reset()

    # Reset the neuron's state. Called at creation and at the beginning of every cycle
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

        # TODO: Forced activity
        self.act_ext = None
        # TODO: Non-depressed activity. Implemented?
        self.act_nd = self.act
        # TODO: Activity at the end of the minus phase
        self.act_m = self.act


    # Excitatory net input, for visualization
    @property
    def net_input(self):
        return self.g_bar_e * self.g_e

    # TODO: Force the activity of the neuron
    def force_activity(self, act_ext):
        assert len(self.excitatory_inputs) == 0

        # Forced activity
        self.act_ext = act_ext

        self.g_e = self.act_ext / self.g_bar_e  # neuron.net == neuron.act
        # cycle
        self.I_net = 0.0
        self.act = self.act_ext
        self.act_nd = self.act_ext
        if self.act == 0:
            self.v_m = self.e_l
        else:
            self.v_m = self.act_thr + self.act_ext / self.act_gain
        self.v_m_eq = self.v_m

    # Add an input for the next step
    def add_excitatory(self, inp_act):
        self.excitatory_inputs.append(inp_act)


    ################  Utility functions  ################


    # Pre-compute the convolution for the noisy xx1 function as a look-up table
    def nxx1(self, v_m):

        # If we have not precomputed the table yet
        if self.nxx1_table is None:
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
    # TODO: Use?
    def calculate_net_input(self):
        # If the activity of the neuron is forced, then normal external inputs are ignored, just set to the forced act
        if self.act_ext is not None:
            assert len(self.excitatory_inputs) == 0
            return

        net_raw_input = 0.0
        if len(self.excitatory_inputs) > 0:
            # Total input = sum of excitatory inputs in this current step
            net_raw_input = sum(self.excitatory_inputs)

            # Clear the excitatory inputs for the next step
            self.excitatory_inputs = []

        # Update g_e
        # TODO: net_input?
        self.g_e += self.integ_dt * self.net_input_dt * (net_raw_input - self.g_e)


    # One "time step" of the neuron. Update activity of the neuron
    def step(self, phase, g_i=0.0):

        # Forced activity
        if self.act_ext is not None:
            self.update_avgs()
            self.update_logs()
            # TODO: see self.force_activity
            return

        # Calculate net current. Half-step integration
        self.I_net = self.calculate_net_current(g_i, ratecoded=False, steps=2)
        # TODO: Rate-code version of I_net, to provide adequate coupling with v_m_eq. One-step integration
        self.I_net_r = self.calculate_net_current(g_i, ratecoded=True,  steps=1)

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

        # computing new_act, from v_m_eq (because rate-coded neuron)
        if self.v_m_eq <= self.act_thr:
            new_act = self.nxx1(self.v_m_eq - self.act_thr)
        else:
            gc_e = self.g_bar_e * self.g_e
            gc_i = self.g_bar_i * g_i
            gc_l = self.g_bar_l * self.g_l
            g_e_thr = (gc_i * (self.e_i - self.act_thr)
                       + gc_l * (self.e_l - self.act_thr)
                       - self.adapt_curr) / (self.act_thr - self.e_e)

            new_act = self.nxx1(gc_e - g_e_thr)  # gc_e == neuron.net

        # Update activity
        self.act_nd += self.integ_dt * self.v_m_dt * (new_act - self.act_nd)
        # TODO: Implement stp
        self.act = self.act_nd

        # Update adaptation
        if self.adapt_on:
            self.adapt_curr += self.integ_dt * (self.adapt_dt * (self.v_m_gain * (self.v_m - self.e_l)
                             - self.adapt_curr) + self.did_spike * self.spike_gain)

        # Update logs
        self.update_avgs()
        self.update_logs()


    def calculate_net_current(self, g_i, ratecoded=True, steps=1):

        # Input conductance = max (for normalization) * current
        gc_e = self.g_bar_e * self.g_e
        gc_i = self.g_bar_i * g_i
        gc_l = self.g_bar_l * self.g_l
        v_m_eff = self.v_m_eq if ratecoded else self.v_m

        # Iterative approach
        new_I_net = 0.0
        for _ in range(steps):
            new_I_net = (gc_e * (self.e_e - v_m_eff) + gc_i * (self.e_i - v_m_eff)
                     + gc_l * (self.e_l - v_m_eff) - self.adapt_curr)
            v_m_eff += self.integ_dt/steps * self.v_m_dt * new_I_net

        return new_I_net


    ################  Updating averages of activation (for learning)  ################


    # Update all the averages except long-term at the end of every step, for learning
    def update_avgs(self):
        self.avg_ss += self.integ_dt * self.avg_ss_dt * (self.act_nd - self.avg_ss)
        self.avg_s += self.integ_dt * self.avg_s_dt * (self.avg_ss - self.avg_s )
        self.avg_m += self.integ_dt * self.avg_m_dt * (self.avg_s  - self.avg_m )
        self.avg_s_eff = self.avg_m_in_s * self.avg_m + (1 - self.avg_m_in_s) * self.avg_s

    # Long-term average is separate, as it gets calculated at the end of every cycle instead
    def update_avg_l(self):
        self.avg_l += self.avg_l_dt * (self.avg_l_gain * self.avg_m - self.avg_l)
        self.avg_l = max(self.avg_l, self.avg_l_min)

    # TODO: For self-organizing learning
    @property  # Property lets us access a method like an attribute
    def avg_l_lrn(self):
        # No self-organization unless hidden area
        if self.neuron_type != HIDDEN:
            return 0.0
        avg_fact = (self.avg_lrn_max - self.avg_lrn_min)/(self.avg_l_gain - self.avg_l_min)
        return self.avg_lrn_min + avg_fact * (self.avg_l - self.avg_l_min)


    ################  Logs/config  ################

    # Record the current state of the neuron. Called after each step
    def update_logs(self):
        for name in self.logs.keys():
            self.logs[name].append(getattr(self, name))

    # Show the neuron's configurations
    def show_config(self):
        print('Parameters:')
        for name in ['v_m_dt', 'net_input_dt', 'g_l', 'g_bar_e', 'g_bar_l', 'g_bar_i',
                     'e_e', 'e_l', 'e_i', 'act_thr', 'act_gain']:
            print('   {}: {:.2f}'.format(name, getattr(self, name)))
        print('State:')
        for name in ['g_e', 'I_net', 'v_m', 'act', 'v_m_eq']:
            print('   {}: {:.2f}'.format(name, getattr(self, name)))