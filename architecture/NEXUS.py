"""

Handler for the network as a whole

"""


class Network:

    def __init__(self, layers=(), projections=()):

        self.quarter_size = 25

        # Number of steps that have finished in the current cycle
        self.step_count = 0
        # Total number of steps executed (not reset at end of cycle)
        self.total_steps = 0
        # Current quarter number (1, 2, 3, or 4)
        self.quarter_num = 1
        # Number of cycles finished overall
        self.cycle_count = 0

        # Current phase
        self.phase = 'minus'

        # Layers of the network
        self.layers = list(layers)
        # Projections between layers of the network
        self.projections = list(projections)
        # Inputs and outputs of the network
        self._inputs, self._outputs = {}, {}

        # After initialization, build the network
        self.build()


    ################  Network Building  ################


    # Add a connection to the network
    def add_connection(self, connection):

        # Add the connection to the network's connection master list
        self.projections.append(connection)

        # Need to run "build" to reinitialize the network.
        # TODO: Enable real-time projection adding/deleting?
        self.build()

    # Add a layer to the network
    def add_layer(self, layer):

        # Add the layer to the network's layer master list
        self.layers.append(layer)

    # Pre-compute necessary data structures
    # Network building must be done anytime network structure (synapses, projections, etc.) is changed
    def build(self):

        # Net input scaling for different layers and their different levels of activation
        for layer in self.layers:

            # Calculate projection input scaling (normalization)
            rel_sum = sum(connection.wt_scale_rel for connection in layer.incoming_projections)

            # Apply projection input scaling
            for connection in layer.incoming_projections:
                connection.wt_scale_rel_eff = connection.wt_scale_rel / rel_sum


    ################  Utility functions  ################


    # Get a layer by its name. Name conflict returns oldest layer
    def _get_layer(self, name):
        for layer in self.layers:
            if layer.name == name:
                return layer
        raise ValueError("layer '{}' not found.".format(name))

    # Set input activities, done at the beginning of all quarters
    def set_inputs(self, act_map):

        # TODO: Set inputs according to the act_map
        # act_map = dictionary with layer names as keys, and activities arrays as values
        self._inputs = act_map

    # Set output activities, done at the beginning of all quarters
    def set_outputs(self, act_map):

        # TODO: Set outputs according to the act_map
        # act_map = dictionary with layer names as keys, and activities arrays as values
        self._outputs = act_map

    # TODO: Compute the sum of squared error (SSE) for prediction. Runs after minus phase finishes
    def compute_sse(self):

        sse = 0
        for name, activities in self._outputs.items():
            for act, neuron in zip(activities, self._get_layer(name).neurons):
                sse += (act - neuron.act_m)**2
        return sse


    ################  Step handlers  ################


    # Check if network is at a special moment requiring action before starting the step
    def _pre_step(self):

        # If a quarter just ended
        if self.step_count == self.quarter_size:
            self.quarter_num += 1

            # If a cycle just ended
            if self.quarter_num == 5:
                self.cycle_count += 1
                self.quarter_num = 1
            self.step_count = 0

        # If it's the start of a quarter
        if self.step_count == 0:
            # Compute net input scaling for projections
            for connection in self.projections:
               connection.compute_netin_scaling()

            # If it's the start of a cycle
            if self.quarter_num == 1:
                # Reset all layers
                for layer in self.layers:
                    layer.cycle_init()
                # TODO: Force activities for inputs
                for name, activities in self._inputs.items():
                    self._get_layer(name).force_activity(activities)

            # If it's the start of the plus phase
            elif self.quarter_num == 4:
                # TODO: Force activities for outputs
                for name, activities in self._outputs.items():
                    self._get_layer(name).force_activity(activities)

    # Check if network is at a special moment requiring action after executing the step
    def _post_step(self):

        # If it's the end of a quarter
        if self.step_count == self.quarter_size: # end of a quarter
            # If it's the end of the minus phase, call the phase end handler
            if self.quarter_num == 3:
                self.end_minus_phase()

            # If it's the end of the plus phase, call the phase end handler
            if self.quarter_num == 4:
                self.end_plus_phase()


    ################  Temporal process control  ################


    def cycle(self):
        """Execute a cycle. Will execute up until the end of the plus phase."""
        self.quarter()
        while self.quarter_num != 4:
            assert self.step_count == self.quarter_size
            self.quarter()
        return self.compute_sse()

    # Execute the functions that occur at a quarter-step
    def quarter(self):

        self.step()
        while self.step_count < self.quarter_size:
            self.step()

    # Advance the network one time-step. Called from quarter
    def step(self):
        self._pre_step()

        for conn in self.projections:
            conn.step()
        for layer in self.layers:
            layer.step(self.phase)
        self.step_count += 1
        self.total_steps   += 1

        self._post_step()


    ################  Minus/plus phase handlers  ################


    # Handler for the end of the minus phase (first 75 msec of a cycle)
    def end_minus_phase(self):

        # Store the neuron activity as the medium-term average synaptic activity
        for layer in self.layers:
            for neuron in layer.neurons:
                neuron.act_m = neuron.act

        # Set the current phase to plus
        self.phase = 'plus'

    # Handler for the end of the plus phase (last 25 msec of a cycle)
    def end_plus_phase(self):

        # Projections change their weights based on the formulas for error-driven learning
        for conn in self.projections:
            conn.learn()

        # Update the long-term average synaptic activity, done at the end of every cycle
        for layer in self.layers:
            for neuron in layer.neurons:
                neuron.update_avg_l()

        # Set the current phase to minus
        self.phase = 'minus'