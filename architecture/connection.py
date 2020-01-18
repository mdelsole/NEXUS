import numpy as np
import random


# A synapse is a connection between two neurons
class Synapse:
    # Index = position in the weight matrix

    def __init__(self, pre_neuron, post_neuron, w0, fw0, index=None):
        # TODO: Change to less confusing pre_neuron
        # The sending neuron
        self.pre = pre_neuron
        # The receiving neuron
        self.post = post_neuron

        # Initial weight value
        self.wt = w0
        # Initial fast weight value; used for fast/slow learning dynamic
        self.fwt = fw0

        # Change in synaptic weights due to learning
        self.dwt = 0.0


# Projection is a connection between two layers
class Projection:

    def __init__(self, pre_layer, post_layer, **kwargs):

        # The layer sending its activity
        self.pre = pre_layer
        # The layer receiving the activity
        self.post = post_layer
        self.synapses = []

        ########### Parameters ###########

        # Is this and inhibitory projection? TODO: Inhibitory projections not yet implemented
        self.inhib = True
        # Connection pattern for projections. 'full' or '1to1', with 1to1 requiring layers be the same size
        self.proj = 'full'

        ########### Random weight initialization parameters ###########

        # Shape of random weight initialization
        # TODO: Gaussian or uniform?
        self.rnd_type = 'uniform'
        # Mean of the random variable weight initialization
        self.rnd_mean = 0.5
        # Variance (+-range for uniform)
        self.rnd_var = 0.25

        ########### Learning rule parameters ###########

        # Set the learning rule to use (NEXUS or nothing)
        # TODO: Change to boolean?
        self.lrule = 'NEXUS'
        # Learning rate. Bigger means greater weight changes. 0.04 is good start, should decrease over time with age
        self.lrate = 0.04

        # Weighting of the error-driven learning (XCAL)
        self.m_lrn = 1.0

        # Threshold for the XCAL function to start having effect on LTD/LTP
        self.d_thr = 0.0001
        # Point at which the XCAL function reverses direction
        self.d_rev = 0.1

        # TODO: Sigmoid parameters
        self.sig_off = 1.0
        self.sig_gain = 6.0

        ########### Net input scaling parameters ###########

        # Absolute scaling weight; directly calculated by strength of the connection
        self.wt_scale_abs = 1.0
        # Relative scaling weight, weight normalized by the other projections
        self.wt_scale_rel = 1.0

        # Scale the layer's inputs relative to activity
        self.wt_scale_act = 1.0
        # Effective relative scaling weight, scaling relative to other projections
        self.wt_scale_rel_eff = None

        # Connect projected synapses
        self.projection_init()

        # Add this projection to the pre-layer's list of outgoing projections
        pre_layer.outgoing_projections.append(self)
        # Add this projection to the post-layer's list of incoming projections
        post_layer.incoming_projections.append(self)

        for key, value in kwargs.items():
            assert hasattr(self, key)
            setattr(self, key, value)


    ################  Synapse weight handlers  ################


    # Scaled weight = activity-scaled weight * other-projections-scaled weight
    @property
    def wt_scale(self):
        try:
            return self.wt_scale_act * self.wt_scale_rel_eff
        except TypeError as e:
            print('Error: did you correctly run the network.build() method?')
            raise e

    # Return a matrix of the synapse weights
    @property
    def weights(self):
        # For 1to1 projections
        if self.proj.lower() == '1to1':
            return np.array([[synapse.wt for synapse in self.synapses]])

        # For full projections
        else:
            # Weight matrix
            W = np.zeros((len(self.pre.neurons), len(self.post.neurons)))
            # Synapse iterator
            synapse_it = iter(self.synapses)

            # Add the weights to the weight matrix
            for i, pre_u in enumerate(self.pre.neurons):
                for j, post_u in enumerate(self.post.neurons):
                    W[i, j] = next(synapse_it).wt
            return W

    # Override the synapse weights if necessary
    @weights.setter
    def weights(self, value):
        # For 1to1 projections
        if self.proj.lower() == '1to1':
            assert len(value) == len(self.synapses)
            for wt, synapse in zip(value, self.synapses):
                synapse.wt = wt
                # TODO: fwt needs sig_inv?
                synapse.fwt = self.sig_inv(wt)

        # For full projections
        else:
            # Synapse iterator
            synapse_it = iter(self.synapses)
            assert len(value) * len(value[0]) == len(self.synapses)
            for i, pre_u in enumerate(self.pre.neurons):
                for j, post_u in enumerate(self.post.neurons):
                    synapse = next(synapse_it)
                    synapse.wt = value[i][j]
                    synapse.fwt = self.sig_inv(value[i][j])

    # Randomly initialize weights, according to specified distribution
    def _rnd_wt(self):
        # TODO: Gaussian or uniform?
        if self.rnd_type == 'uniform':
            return random.uniform(self.rnd_mean - self.rnd_var, self.rnd_mean + self.rnd_var)
        elif self.rnd_type == 'gaussian':
            return random.gauss(self.rnd_mean, np.sqrt(self.rnd_var))
        raise NotImplementedError


    ################  Time-step handler  ################


    # Advance the projection one time-step
    def step(self):
        # Transmit activity
        for synapse in self.synapses:
            # If activity is not forced
            if synapse.post.act_ext is None:
                # Scale the activity of the projection
                scaled_act = self.wt_scale_abs * self.wt_scale * synapse.wt * synapse.pre.act

                # Add the scaled activity to the post neuron's list of excitatory inputs
                synapse.post.add_excitatory(scaled_act)


    ################  Net input scaling for projections  ################


    # Compute net input scaling for projections
    def compute_netin_scaling(self):

        # Pre-layer average activity
        pre_avg_act = self.pre.avg_act_p_eff
        # Pre-layer size
        pre_size = len(self.pre.neurons)
        # Number of synapses in this projection
        num_synapses = len(self.synapses)

        # Constant
        sem_extra = 2.0
        # Estimated number of active neurons
        pre_act_n = max(1, int(pre_avg_act * pre_size + 0.5))

        # If it was 1to1
        if num_synapses == pre_size:
            self.wt_scale_act = 1.0 / pre_act_n

        # If it was full projection
        else:
            post_act_n_max = min(num_synapses, pre_act_n)
            post_act_n_avg = max(1, pre_avg_act * num_synapses + 0.5)
            post_act_n_exp = min(post_act_n_max, post_act_n_avg + sem_extra)
            self.wt_scale_act = 1.0 / post_act_n_exp


    ################  Initialize projections  ################


    # Connect projected synapses
    def projection_init(self):
        if self.proj == 'full':
            self._full_projection()
        if self.proj == '1to1':
            self._1to1_projection()

    # Fully project the pre-neurons to the post-neurons
    def _full_projection(self):
        # Store the neuron to neuron synapses
        self.synapses = []
        for i, pre_u in enumerate(self.pre.neurons):
            for j, post_u in enumerate(self.post.neurons):
                w0 = self._rnd_wt()
                fw0 = self.sig_inv(w0)
                self.synapses.append(Synapse(pre_u, post_u, w0, fw0, index=(i, j)))

    # Project one pre-neuron to one post-neuron for every neuron in the areas
    def _1to1_projection(self):
        # Store the neuron to neuron synapses
        self.synapses = []
        # 1to1 MUST have same layer size
        assert len(self.pre.neurons) == len(self.post.neurons)
        for i, (pre_u, post_u) in enumerate(zip(self.pre.neurons, self.post.neurons)):
            w0 = self._rnd_wt()
            fw0 = self.sig_inv(w0)
            self.synapses.append(Synapse(pre_u, post_u, w0, fw0, index=(i, i)))


    ################  Learning  ################


    # Execute "learning"
    def learn(self):

        # TODO: Needed? Probably, to turn learning off
        if self.lrule is not None:
            # Calculate synaptic weight change
            self.learning_rule()

            # Apply synaptic weight change
            self.apply_dwt()

        # Clip the weights to between 0.0-1.0 after weight change
        # TODO: Needed? Or should be done automatically?
        for synapse in self.synapses:
            synapse.wt = max(0.0, min(1.0, synapse.wt))

    # The NEXUS learning rule. Calculates synaptic weight change
    def learning_rule(self):
        for i, synapse in enumerate(self.synapses):
            # Calculate effective short-term average synaptic activity
            srs = synapse.post.avg_s_eff * synapse.pre.avg_s_eff
            # Calculate medium-term average synaptic activity
            srm = synapse.post.avg_m * synapse.pre.avg_m

            # Calculate the change in synaptic weight using the learning function
            synapse.dwt += (self.lrate * (self.m_lrn * self.xcal(srs, srm)
                                       + synapse.post.avg_l_lrn * self.xcal(srs, synapse.post.avg_l)))

    # Apply the change in weighs resulting from learning
    def apply_dwt(self):

        for synapse in self.synapses:
            synapse.dwt *= (1 - synapse.fwt) if (synapse.dwt > 0) else synapse.fwt
            synapse.fwt += synapse.dwt
            synapse.wt = self.sig(synapse.fwt)

            synapse.dwt = 0.0


    ################  Function calculations  ################


    # XCAL learning function
    def xcal(self, x, th):
        if x < self.d_thr:
            return 0
        elif x > th * self.d_rev:
            return x - th
        else:
            return -x * ((1 - self.d_rev)/self.d_rev)

    # TODO: Sigmoid activation function (need?)
    def sig(self, w):
        return 1 / (1 + (self.sig_off * (1 - w) / w) ** self.sig_gain)

    # TODO: Inverse sigmoid (need?)
    def sig_inv(self, w):

        # Clamp range
        if w <= 0.0:
            return 0.0
        elif w >= 1.0:
            return 1.0

        # Return inverse sigmoid
        return 1 / (1 + ((1 - w) / w) ** (1 / self.sig_gain) / self.sig_off)