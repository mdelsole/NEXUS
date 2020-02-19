import sys
from PyQt5.QtWidgets import QWidget, QDesktopWidget, QTabWidget, QVBoxLayout

import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore
from pyqtgraph.parametertree import Parameter, ParameterTree
from pyqtgraph.parametertree import types as pTypes
from pyqtgraph.dockarea import *

from ui import network_runner, input_loader
import numpy as np
network = network_runner.network_runner()


class NexusGUI(QtGui.QWidget):
    def __init__(self):
        QtGui.QWidget.__init__(self)

        # Keep track of the docks
        self.displays = []

        # Keep track of display animation
        self.timer = QtCore.QTimer(self)
        self.input_list = [[input_loader.leftdiag, input_loader.leftdiag_2d],
                           [input_loader.vertical, input_loader.vertical_2d],
                           [input_loader.rightdiag, input_loader.rightdiag_2d],
                           [input_loader.horizontal, input_loader.horizontal_2d]]
        self.index = 0

        pg.setConfigOption('background', (190, 190, 190))

        self.setupGUI()

        ########### Control panel tree ###########

        self.objectGroup = ComponentGroupParam()

        self.control_tree_params = Parameter.create(name='control_tree_params', type='group',
            children=[
                dict(name='Duration', type='float', value=10.0, step=0.1, limits=[0.1, None]),
                dict(name='Real-time Animation', type='bool', value=True),
                dict(name='Animation Speed', type='float', value=1.0, dec=True, step=0.1, limits=[0.0001, None]),
                dict(name='Build Network', type='action'),
                dict(name='Train Network', type='action'),
                dict(name='Stop', type='action'),
            ])

        # Set the control panel tree
        self.control_panel_tree.setParameters(self.control_tree_params, showTop=False)

        # Actions
        self.control_tree_params.param('Build Network').sigActivated.connect(self.build_network)
        self.control_tree_params.param('Train Network').sigActivated.connect(self.train_network)
        self.control_tree_params.param('Stop').sigActivated.connect(self.stop)
        self.control_tree_params.sigTreeStateChanged.connect(self.control_tree_changed)

        ########### Builder tree ###########

        self.builder_tree_params = Parameter.create(name='params', type='group', children=[
            self.objectGroup,
        ])

        self.builder_tree_params.sigTreeStateChanged.connect(self.builder_tree_changed)

        self.builder_tree.setParameters(self.builder_tree_params, showTop=False)


    ################  GUI  ################


    def setupGUI(self):
        self.layout = QtGui.QVBoxLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(self.layout)
        self.splitter = QtGui.QSplitter()
        self.splitter.setOrientation(QtCore.Qt.Horizontal)
        self.splitter.setSizes([100, 200])


        # Add the splitter
        self.layout.addWidget(self.splitter)

        ########### Left side ###########

        # Initialize tab screen
        self.tabs = QTabWidget()
        self.tab1 = QWidget()
        self.tab2 = QWidget()
        self.tabs.resize(300, 200)

        # TODO: Make margin smaller

        # Add tabs
        self.tabs.addTab(self.tab1, "Control Panel")
        self.tabs.addTab(self.tab2, "Builder")
        self.splitter.addWidget(self.tabs)

        # Create the control_panel_tree
        self.control_panel_tree = ParameterTree(showHeader=True)

        # Add the parameter tree to tab1
        self.layout1 = QVBoxLayout()
        self.layout1.addWidget(self.control_panel_tree)
        self.tab1.setLayout(self.layout1)

        # Create the builder_tree
        self.builder_tree = ParameterTree(showHeader=True)

        # Add the parameter tree to tab1
        self.layout2 = QVBoxLayout()
        self.layout2.addWidget(self.builder_tree)
        self.tab2.setLayout(self.layout2)

        ########### Right side ###########

        self.area = DockArea()
        self.splitter.addWidget(self.area)

        ########### Initial windows ###########

        # We have the "input", the input layer, hidden

        self.init_dock = self.visualize_layer(input_loader.leftdiag_2d, "Input")
        self.init_dock_closed = False

    def visualize_layer(self, data, layer_name):

        d1 = Dock(layer_name, size=(400, 400))
        networkVisual = pg.GraphicsLayoutWidget()

        view = networkVisual.addViewBox()
        # Lock the aspect ratio so pixels are always square
        view.setAspectLocked(True)
        # Create image item
        img = pg.ImageItem(border='w')
        img.setLevels([])
        view.addItem(img)
        img.setImage(np.rot90(data,-1))

        d1.addWidget(networkVisual)

        # Add the dock to the area
        self.area.addDock(d1)
        self.displays.append(view.addedItems)
        return d1


    ################  Tree change handlers  ################


    ## If anything changes in the tree, print a message
    def control_tree_changed(param, changes):
        pass

    ## If anything changes in the tree, print a message
    def builder_tree_changed(self, *args):
        print("tree changes:", args[1])
        for param, change, data in args[1]:
            print(change)
            # if change == 'childAdded':
            if change == 'activated':
                if param.name() == 'Add layer to network':
                    x, name, new = param.parent().buildLayer()
                    if new:
                        self.visualize_layer(x, name)
                        print([layer.name for layer in network.layers])
                    for cl in self.builder_tree_params.param('Components'):
                        for child in cl.children():
                            print(child)
                            if child.type() == 'Projection':
                                print("Here")
                                child.__init__()
                elif param.name() == 'Add projection to network':
                    x = param.parent().addProjection()


    ################  Control panel actions  ################


    def train_network(self):
        self.timer.timeout.connect(self.train_handler)
        self.timer.start(1)


    def train_handler(self):
        # Change the input
        self.displays[0][0].setImage(np.rot90(self.input_list[self.index][1]))

        # Change the hidden layer
        x = network.train_network(self.input_list[self.index][0], self.input_list[self.index][0])
        self.displays[1][0].setImage(np.rot90(x,-1))

        # Cycle through
        self.index += 1
        if self.index > 3:
            self.index = 0

    def stop(self):
        self.timer.stop()

    def build_network(self):

        # First time, close placeholder network
        # if not self.init_dock_closed:
            # self.init_dock.close()

        network.build_network()


################  Parameters  ################


# Add a layer
class LayerParam(pTypes.GroupParameter):
    def __init__(self, **kwds):

        # Define the layer group
        self.defs = dict(name="Layer", autoIncrementName=True, type='Layer', renamable=True, removable=True, children=[
            dict(name='Show', type='bool', value=True),
            dict(name='Size', type='int', value=25),
            dict(name='Type', type='list', values=['INPUT', 'HIDDEN', 'OUTPUT'], value='INPUT'),
            dict(name='Layer Inhibition', type='bool', value=True),
            dict(name='Inhibitory Gain', type='float', value=2.0, step=0.1),
            dict(name='Feedforward inhibitory gain', type='float', value=1.0, step=0.1),
            dict(name='Feedbackward inhibitory gain', type='float', value=0.5, step=0.1),
            dict(name='Add layer to network', type='action'),
            NeuronGroup(),
        ])
        pTypes.GroupParameter.__init__(self, **self.defs)
        self.restoreState(kwds, removeChildren=False)

    def buildLayer(self):
        size = self['Size']
        name = self.name()
        type = self['Type']
        lay_inhib = self['Layer Inhibition']
        inhib_gain = self['Inhibitory Gain']
        ffi = self['Feedforward inhibitory gain']
        fbi = self['Feedbackward inhibitory gain']
        newLayer = network.add_layer(size=size, neuron_type=type, name=name, lay_inhib=lay_inhib,
                               inhib_gain=inhib_gain, ffi=ffi, fbi=fbi)
        return newLayer


    def add_to_network(self):
        self.buildLayer()


# Neuron specifications for the layer
class NeuronGroup(pTypes.GroupParameter):
    def __init__(self, **kwds):
        defs = dict(name="Neuron Settings", autoIncrementName=True, type='Neuron', renamable=True, removable=True, children=[
            dict(name='Number of Layers', type='int', value=5, limits=[1, None]),
            dict(name='Spacing', type='float', value=1.0, step=0.1),
        ])
        pTypes.GroupParameter.__init__(self, **defs)
        self.restoreState(kwds, removeChildren=False)

    def generate(self):
        prog = []
        for cmd in self:
            prog.append((cmd['Proper Time'], cmd['Neuron']))
        return prog


# Add a projection
class ProjectionParam(pTypes.GroupParameter):
    def __init__(self, **kwds):
        print("here", [layer.name for layer in network.layers])
        self.defs = dict(name="Projection", autoIncrementName=True, type='Projection', renamable=True, removable=True, children=[
            dict(name='Layer from', type='list', values=[layer.name for layer in network.layers]),
            dict(name='Layer to', type='list', values=[layer.name for layer in network.layers]),
            dict(name='Add projection to network', type='action'),
        ])
        pTypes.GroupParameter.__init__(self, **self.defs)
        self.restoreState(kwds, removeChildren=False)

    def addProjection(self):
        layer_from = self['Layer from']
        layer_to = self['Layer to']

        newLayer = network.add_projection(layer_from, layer_to)
        return newLayer

    def setLayers(self):
        self.__init__()


################  Headers  ################


class ComponentGroupParam(pTypes.GroupParameter):
    def __init__(self):
        pTypes.GroupParameter.__init__(self, name="Components")
        self.addChild(LayerHeaderParam())
        self.addChild(ProjectionHeaderParam())


class LayerHeaderParam(pTypes.GroupParameter):
    def __init__(self):
        pTypes.GroupParameter.__init__(self, name="Layers", addText="Add New Layer")

    def addNew(self):
        self.addChild(LayerParam())


class ProjectionHeaderParam(pTypes.GroupParameter):
    def __init__(self):
        pTypes.GroupParameter.__init__(self, name="Projections", addText="Add New Projection")

    def addNew(self):
        self.addChild(ProjectionParam())


################  App runner  ################


if __name__ == '__main__':
    pg.mkQApp()
    win = NexusGUI()

    # Set window title
    win.setWindowTitle("NEXUS")
    # Set window position
    win.move(0, 0)
    # Set window size to screen size
    ag = QDesktopWidget().availableGeometry()
    win.resize(ag.width() - 10, ag.height() - 50)

    # Show window
    win.show()

    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()