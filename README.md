# NEXUS

**Nexus**: A specialized layer of the cell membrane involved in intercellular communication and adhesion; a synapse.

This is a WIP biologically-inspired approach to creating a neural network for object recognition, and eventually artificial imagination.

## Architecture: Algorithms and Mechanics
Singular neuron level:

- [x] Hodgkin-huxley model of neuron membrane potential updates
- [x] Adaptive exponential integrate-and-fire model for neuronal spiking activity

Synapse level:

- [x] Spike-timing-dependent plasticity
- [x] Long-term depression and long-term potentiation

Singular layer level:
- [x] Self-organizing learning via intrinsic biological properties of synapses
- [x] Inhibition

Cross-layer (network) level:

- [x] Cross-layer synapse projections
- [x] Bi-directional connectivity
- [ ] Error-driven backpropagation learning (still researching biological plausibility)

Code needs to be cleaned up and optimized, but it works.

## Visual Pathway
### Relevant Layers:
Ventral Stream ("What" recognition)
- [ ] LGN
- [ ] V1
- [ ] V2
- [ ] V4
- [ ] PIT
- [ ] AIT

Dorsal Stream ("Where/how" recognition, branching off of V1)
- [ ] ??? (Future work)

## UI Features
Not much, but hey I'm only one person

- [x] Graph of individual neuron test firing
- [ ] Runnable
- [x] Parameter modification
- [ ] Network visualization
- [x] Different view tabs

## Sources
* Hodgkin-Huxely Model: https://en.wikipedia.org/wiki/Hodgkin%E2%80%93Huxley_model
* Adaptive Exponential Model: https://www.ncbi.nlm.nih.gov/pubmed/16014787 
  * More usefully: http://www.scholarpedia.org/article/Adaptive_exponential_integrate-and-fire_model
* Spike-Timing-Dependent Plasticity: https://www.jneurosci.org/content/jneuro/28/13/3310.full.pdf
