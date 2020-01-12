from tests import graphs
from architecture import neuron

log_names = ('net_input', 'I_net', 'v_m', 'act', 'v_m_eq', 'adapt_curr', 'avg_ss', 'avg_s', 'avg_m', 'avg_s_eff')

receiver = neuron.Neuron(log_names=log_names)
receiver.show_config()

# 0 - 10 ms: 0, 10 - 150 ms: 0.3, 150 - 190 ms: 0
inputs = 10*[0.0] + 150*[0.3] + 40*[0.0]

for g_e in inputs:
    receiver.add_excitatory(g_e)
    receiver.calculate_net_input()
    receiver.step('minus')

