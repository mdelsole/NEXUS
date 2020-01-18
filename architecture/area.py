"""

Singular area dynamics

"""

import numpy as np

from .neuron import Neuron, INPUT, HIDDEN, OUTPUT


# TODO: Become area, be an input/hidden/output area
class Area:

    def __init__(self, size, neuron_type=HIDDEN, name=None, **kwargs):
        # What type of neurons are in this area
        self.neuron_type = neuron_type

        # Set the name of this area
        self.name = name

        # Create the neurons of the area, with number denoted by input parameter 'size'
        self.neurons = [Neuron(neuron_type=neuron_type) for _ in range(size)]

        # Projections from this area to other areas
        self.outgoing_projections = []
        # Projections to this area from other areas
        self.incoming_projections = []

        # Current step count for area
        self.step_count = 0


        ########### Inhibition ###########

        # Enable inhibition or not
        self.lay_inhib = True

        # Inhibitory conductance
        self.gc_i = 0.0
        # Feed-forward component of inhibition
        self.ffi = 0.0
        # Feed-back component of inhibition
        self.fbi = 0.0

        # Time-step constant for integration of feed-back inhibition
        self.fb_dt = 1 / 1.4

        # TODO: In-out areas: 1.0, 1.0, 1.8. Hidden Areas: 1.0, 0.5, 2.0?
        # Feed-forward inhibition gain factor (for scaling)
        self.ff = 1.0
        # Feed-back inhibition gain factor (for scaling)
        self.fb = 1.0
        # Overall inhibition gain factor
        self.inhib_gain = 1.8

        # Reset calculation value for fbi and ffi. If 1.0, fbi and ffi will reset to 0 at start of every cycle
        self.inhib_reset = 1.0

        # Threshold for activating feed-forward inhibition
        self.ff0 = 0.1

        ########### TODO: Average activity parameters ###########

        # Target for adapting inhibition, initial estimated average value level
        self.avg_act_targ_init = 0.2
        # Used to calculate avg_p_act_eff
        self.avg_act_adjust = 1.0
        # If true, avg_act_p_eff is constant, set to avg_act_targ_init
        self.avg_act_fixed = False
        # If true, override targ_init value with first estimation
        self.avg_act_use_first = False
        # Time constant for integrating act_p_avg
        self.avg_act_tau = False  # time constant for integrating act_p_avg

        # Average activity of the area. Computed after every step
        self.avg_act = 0.0
        self.avg_act_p_eff = self.avg_act_targ_init

        ########### Utility ###########

        for key, value in kwargs.items():
            assert hasattr(self, key)  # making sure the parameter exists.
            setattr(self, key, value)

        # Log inhibitory conductance
        self.logs = {'gc_i': []}

    # Return the matrix of activities for neurons in this area
    @property
    def activities(self):
        return [n.act for n in self.neurons]

    # Return the matrix of net excitatory input for neurons in this area
    @property
    def net_inputs(self):
        return [n.g_e for n in self.neurons]

    # Set the neuron's activities equal to the inputs
    def force_activity(self, activities):
        assert len(activities) == len(self.neurons)
        for n, act in zip(self.neurons, activities):
            n.force_activity(act)


    ################  Inhibition  ################


    # Compute the inhibition for the area
    def inhibition(self):

        if self.lay_inhib:
            # Retrieve net inputs of neurons in this area for feed-forward inhibition
            _net_inputs = self.net_inputs
            # Calculate feed-forward inhibition
            self.ffi = self.ff * max(0, np.mean(_net_inputs) - self.ff0)

            # Calculate feed-back inhibition
            self.fbi += self.fb_dt * (self.fb * self.avg_act - self.fbi)

            # Return overall inhibitory conductance
            return self.inhib_gain * (self.ffi + self.fbi)
        else:
            return 0.0


    ################  Temporal process control  ################


    # Advance the area one time-step, and all the neurons in it
    def step(self, phase):

        # Calculate the net inputs for the neurons this area
        for n in self.neurons:
            n.calculate_net_input()

        # Inhibition happens during minus phase; minus phase: network runs free, plus phase: network is clamped to value
        if phase == 'minus':
            self.gc_i = self.inhibition()

        # Advance the neurons in the area one time-step, set their inhibition
        for n in self.neurons:
            n.step(phase, g_i=self.gc_i)

        # Compute the average activity of the area
        self.avg_act = np.mean(self.activities)

        # Update logs
        self.update_logs()

        # Indicate that the area has been advanced one time-step
        self.step_count += 1

    # Initialize the area for a new cycle
    def cycle_init(self):

        # Reset all neurons
        for u in self.neurons:
            u.reset()

        # Reset inhibition; change inhib_reset to value below 1.0 to make it not reset to zero
        self.ffi -= self.inhib_reset * self.ffi
        self.fbi -= self.inhib_reset * self.fbi


    ################  Logs/config  ################


    # Show the area's configurations
    def show_config(self):
        print('Parameters:')
        for name in ['fb_dt', 'ff0', 'ff', 'fb', 'inhib_gain']:
            print('   {}: {:.2f}'.format(name, getattr(self, name)))
        print('State:')
        for name in ['gc_i', 'fbi', 'ffi']:
            print('   {}: {:.2f}'.format(name, getattr(self, name)))

    # Record the area's current state. Called after each step
    def update_logs(self):
        self.logs['gc_i'].append(self.gc_i)