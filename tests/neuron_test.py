from tests import graphs
from architecture import neuron

log_names = ('net_in', 'I_net', 'v_m', 'act', 'v_m_eq', 'adapt_curr')

receiver = neuron.Neuron(log_names=log_names)
receiver.show_config()

inputs = 10*[0.0] + 150*[1.0] + 40*[0.0]


for g_e in inputs:
    receiver.add_excitatory_inputs(g_e)
    receiver.calculate_net_input()
    receiver.step('minus')

graphs.unit_activity(receiver.logs)
