from PyQt5 import QtWidgets, uic
from pyqtgraph import PlotWidget, plot
import pyqtgraph as pg
import sys  # We need sys so that we can pass argv to QApplication
import os

from architecture import neuron


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        ########### Get data ###########

        data = compute_neurons()
        names = ('net_input', 'v_m', 'I_net', 'act', 'v_m_eq', 'adapt_curr')
        colors = ((0, 0, 0), 'b', 'g', 'm', 'r', 'b')

        ########### Options ###########

        self.graphWidget = pg.PlotWidget()
        self.setCentralWidget(self.graphWidget)

        self.graphWidget.setXRange(0, len(data[names[0]]))
        self.graphWidget.setYRange(-0.1, 1)
        self.graphWidget.showGrid(x=True, y=True)

        # Set labels
        self.graphWidget.setLabel('left', 'Activity', color='black', size=15)
        self.graphWidget.setLabel('bottom', 'Time (ms)', color='black', size=15)

        ########### Plot data ###########

        self.graphWidget.setBackground('w')
        for name in names:
            self.plot(range(len(data[name])), data[name], name, colors[names.index(name)])

    def plot(self, x, y, plotname, color):
        pen = pg.mkPen(color=color, width=5)
        self.graphWidget.plot(x, y, name=plotname, pen=pen)


def compute_neurons():
    log_names = ('net_input', 'I_net', 'v_m', 'act', 'v_m_eq', 'adapt_curr', 'avg_ss', 'avg_s', 'avg_m', 'avg_s_eff')

    receiver = neuron.Neuron(log_names=log_names)
    receiver.show_config()

    # 0 - 10 ms: 0, 10 - 150 ms: 0.3, 150 - 190 ms: 0
    inputs = 10 * [0.0] + 150 * [0.3] + 40 * [0.0]

    for g_e in inputs:
        receiver.add_excitatory(g_e)
        receiver.calculate_net_input()
        receiver.step('minus')

    return receiver.logs


def main():
    app = QtWidgets.QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()