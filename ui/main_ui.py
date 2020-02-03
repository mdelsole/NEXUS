import numpy as np
import collections
import sys, os
from PyQt5.QtWidgets import QApplication, QWidget, QDesktopWidget, QTabWidget, QVBoxLayout

import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore
from pyqtgraph.parametertree import Parameter, ParameterTree
from pyqtgraph.parametertree import types as pTypes
import pyqtgraph.configfile
from pyqtgraph.dockarea import *

from ui import network_runner

network = network_runner.network_runner()

class NexusGUI(QtGui.QWidget):
    def __init__(self):
        QtGui.QWidget.__init__(self)

        # Handling display
        self.layers_displayed = []

        pg.setConfigOption('background', (190, 190, 190))

        self.setupGUI()

        ########### Control panel tree ###########

        self.objectGroup = ComponentGroupParam()

        self.control_tree_params = Parameter.create(name='control_tree_params', type='group',
            children=[
                dict(name='Duration', type='float', value=10.0, step=0.1, limits=[0.1, None]),
                dict(name='Reference Frame', type='list', values=[]),
                dict(name='Real-time Animation', type='bool', value=True),
                dict(name='Animation Speed', type='float', value=1.0, dec=True, step=0.1, limits=[0.0001, None]),
                dict(name='Build Network', type='action'),
                dict(name='Run Network', type='action'),
                dict(name='Display', type='action'),
                dict(name='Save', type='action'),
                dict(name='Load', type='action'),
            ])

        # Set the control panel tree
        self.control_panel_tree.setParameters(self.control_tree_params, showTop=False)

        # Actions
        self.control_tree_params.param('Build Network').sigActivated.connect(self.build_network)
        self.control_tree_params.param('Run Network').sigActivated.connect(self.run_network)
        self.control_tree_params.param('Display').sigActivated.connect(self.display)
        self.control_tree_params.param('Save').sigActivated.connect(self.save)
        self.control_tree_params.param('Load').sigActivated.connect(self.load)
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

        # Placeholder starter
        self.init_dock = self.visualize_layer(np.array(([0.0, 1.0], [0.0, 0.0])), "Input")
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
        img.setImage(data)

        d1.addWidget(networkVisual)

        # Add the dock to the area
        self.area.addDock(d1)
        return d1


    ################  Tree change handlers  ################


    ## If anything changes in the tree, print a message
    def control_tree_changed(param, changes):
        pass
        # print("tree changes:")

    ## If anything changes in the tree, print a message
    def builder_tree_changed(self, *args):
        print("tree changes:", args[1])
        for param, change, data in args[1]:
            if change == 'childAdded':
                for cl in self.builder_tree_params.param('Components'):
                    for child in cl.children():
                        if child.type() == 'Projection':
                            child.setLayers()



    ################  Control panel actions  ################

    def display(self):
        pass

    def run_network(self):
        network.train_network()

    def build_network(self):
        #x = network_runner.run(25)
        #self.visualize_layer(x, "V1")

        # First time, close placeholder network
        #if not self.init_dock_closed:
            #self.init_dock.close()

        for cl in self.builder_tree_params.param('Components'):
            for child in cl.children():
                if child.type() == 'Layer':
                    x, name, new = child.buildLayer()
                    if new:
                        self.visualize_layer(x, name)
                        print([layer.name for layer in network.layers])
                if child.type() == 'Projection':
                    pass


    ################  Save/Load  ################


    def save(self):
        fn = str(pg.QtGui.QFileDialog.getSaveFileName(self, "Save State..", "untitled.cfg", "Config Files (*.cfg)"))
        if fn == '':
            return
        # TODO: Change to builder_tree_params
        state = self.control_tree_params.saveState()
        pg.configfile.writeConfigFile(state, fn)

    def load(self):
        fn = str(pg.QtGui.QFileDialog.getOpenFileName(self, "Save State..", "", "Config Files (*.cfg)"))
        if fn == '':
            return
        state = pg.configfile.readConfigFile(fn)
        self.loadState(state)

    def loadState(self, state):
        # TODO: Change to builder_tree_params
        self.control_tree_params.param('Components').clearChildren()
        self.control_tree_params.restoreState(state, removeChildren=False)
        self.recalculate()


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


################  Parameters  ################


# Add a layer
class LayerParam(pTypes.GroupParameter):
    def __init__(self, **kwds):

        # Define the layer group
        defs = dict(name="Layer", autoIncrementName=True, type='Layer', renamable=True, removable=True, children=[
            dict(name='Show', type='bool', value=True),
            dict(name='Size', type='int', value=25),
            dict(name='Type', type='list', values=['INPUT', 'HIDDEN', 'OUTPUT'], value='INPUT'),
            dict(name='Layer Inhibition', type='bool', value=True),
            dict(name='Inhibitory Gain', type='float', value=2.0, step=0.1),
            dict(name='Feedforward inhibitory gain', type='float', value=1.0, step=0.1),
            dict(name='Feedbackward inhibitory gain', type='float', value=0.5, step=0.1),
            NeuronGroup(),
        ])
        pTypes.GroupParameter.__init__(self, **defs)
        self.restoreState(kwds, removeChildren=False)

    def buildLayer(self):
        size = self['Size']
        name = self.name()
        type = self['Type']
        lay_inhib = self['Layer Inhibition']
        inhib_gain = self['Inhibitory Gain']
        ffi = self['Feedforward inhibitory gain']
        fbi = self['Feedbackward inhibitory gain']
        newLayer = network.addLayer(size=size, neuron_type=type, name=name, lay_inhib=lay_inhib,
                               inhib_gain=inhib_gain, ffi=ffi, fbi=fbi)
        return newLayer


    def layerName(self):
        return self.name()


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
        print([layer.name for layer in network.layers])
        self.defs = dict(name="Projection", autoIncrementName=True, type='Projection', renamable=True, removable=True, children=[
            dict(name='Layer from', type='list', values=[layer.name for layer in network.layers]),
            dict(name='Layer to', type='list', values=[layer.name for layer in network.layers]),
        ])
        pTypes.GroupParameter.__init__(self, **self.defs)
        self.restoreState(kwds, removeChildren=False)

    def addProjection(self):
        layer_from = self['Layer from']
        layer_to = self['Layer to']

        newLayer = network.addLayer(size=layer_from, neuron_type=layer_to)
        return newLayer

    def setLayers(self):
        self.__init__()


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