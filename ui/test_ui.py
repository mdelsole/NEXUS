import numpy as np
import collections
import sys, os
from PyQt5.QtWidgets import QApplication, QWidget, QDesktopWidget, QTabWidget, QVBoxLayout

import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore
from pyqtgraph.parametertree import Parameter, ParameterTree
from pyqtgraph.parametertree import types as pTypes
import pyqtgraph.configfile
from pyqtgraph.python2_3 import xrange


class NexusGUI(QtGui.QWidget):
    def __init__(self):
        QtGui.QWidget.__init__(self)

        # Handling display
        self.layers_displayed = []

        self.setupGUI()

        self.objectGroup = ComponentGroupParam()

        self.params = Parameter.create(name='params', type='group', children=[
            dict(name='Duration', type='float', value=10.0, step=0.1, limits=[0.1, None]),
            dict(name='Reference Frame', type='list', values=[]),
            dict(name='Animate', type='bool', value=True),
            dict(name='Animation Speed', type='float', value=1.0, dec=True, step=0.1, limits=[0.0001, None]),
            dict(name='Build Network', type='action'),
            dict(name='Display', type='action'),
            dict(name='Save', type='action'),
            dict(name='Load', type='action'),
        ])

        self.control_panel_tree.setParameters(self.params, showTop=False)
        # When pressing the button, calls the display method, where we can put our images
        self.params.param('Display').sigActivated.connect(self.display)
        self.params.param('Save').sigActivated.connect(self.save)
        self.params.param('Load').sigActivated.connect(self.load)
        self.params.sigTreeStateChanged.connect(self.tree_changed)

        self.params2 = Parameter.create(name='params', type='group', children=[
            self.objectGroup,
        ])

        self.builder_tree.setParameters(self.params2, showTop=False)

    def setupGUI(self):
        self.layout = QtGui.QVBoxLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(self.layout)
        self.splitter = QtGui.QSplitter()
        self.splitter.setOrientation(QtCore.Qt.Horizontal)
        self.splitter.setSizes([int(self.height() * 0.6), int(self.height() * 0.4)])

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

        # Add a graph to the right side
        self.animationPlots = pg.GraphicsLayoutWidget()
        self.splitter.addWidget(self.animationPlots)

        # Add left graph to our animationPlots
        self.inertAnimationPlot = self.animationPlots.addPlot()

    def tree_changed(self, *args):
        areas = []
        for c in self.params.param('Components'):
            areas.extend(c.areaNames())
        self.params.param('Reference Frame').setLimits(areas)
        # self.setAnimation(self.params['Animate'])

    def save(self):
        fn = str(pg.QtGui.QFileDialog.getSaveFileName(self, "Save State..", "untitled.cfg", "Config Files (*.cfg)"))
        if fn == '':
            return
        # TODO: Change to params2
        state = self.params.saveState()
        pg.configfile.writeConfigFile(state, fn)

    def load(self):
        fn = str(pg.QtGui.QFileDialog.getOpenFileName(self, "Save State..", "", "Config Files (*.cfg)"))
        if fn == '':
            return
        state = pg.configfile.readConfigFile(fn)
        self.loadState(state)

    def loadState(self, state):
        # TODO: Change to params2
        self.params.param('Components').clearChildren()
        self.params.restoreState(state, removeChildren=False)
        self.recalculate()

    def display(self):
        pass


################  Headers  ################


class ComponentGroupParam(pTypes.GroupParameter):
    def __init__(self):
        pTypes.GroupParameter.__init__(self, name="Components")
        self.addChild(AreaHeaderParam())
        self.addChild(ProjectionHeaderParam())


class AreaHeaderParam(pTypes.GroupParameter):
    def __init__(self):
        pTypes.GroupParameter.__init__(self, name="Areas", addText="Add New Area")

    def addNew(self):
        self.addChild(AreaParam())


class ProjectionHeaderParam(pTypes.GroupParameter):
    def __init__(self):
        pTypes.GroupParameter.__init__(self, name="Projection", addText="Add New Projection")

    def addNew(self):
        self.addChild(ProjectionParam())


################  Parameters  ################


# Add an area
class AreaParam(pTypes.GroupParameter):
    def __init__(self, **kwds):

        # Define the area group
        defs = dict(name="Area", autoIncrementName=True, renamable=True, removable=True, children=[
            dict(name='Initial Position', type='float', value=0.0, step=0.1),
            dict(name='Rest Mass', type='float', value=1.0, step=0.1, limits=[1e-9, None]),
            dict(name='Color', type='color', value=(100, 100, 150)),
            dict(name='Size', type='float', value=0.5),
            dict(name='Vertical Position', type='float', value=0.0, step=0.1),
            NeuronGroup(),
        ])
        pTypes.GroupParameter.__init__(self, **defs)
        self.restoreState(kwds, removeChildren=False)

    def buildAreas(self):
        x0 = self['Initial Position']
        y0 = self['Vertical Position']
        color = self['Color']
        m = self['Rest Mass']
        size = self['Size']
        prog = self.param('Neuron').generate()
        c = Area(x0=x0, m0=m, y0=y0, color=color, prog=prog, size=size)
        return {self.name(): c}

    def areaNames(self):
        return [self.name()]


pTypes.registerParameterType('Area', AreaParam)


# Add a projection
class ProjectionParam(pTypes.GroupParameter):
    def __init__(self, **kwds):
        defs = dict(name="Projection", autoIncrementName=True, renamable=True, removable=True, children=[
            dict(name='Number of Areas', type='int', value=5, limits=[1, None]),
            dict(name='Spacing', type='float', value=1.0, step=0.1),
        ])
        # defs.update(kwds)
        pTypes.GroupParameter.__init__(self, **defs)
        self.restoreState(kwds, removeChildren=False)


pTypes.registerParameterType('Projection', ProjectionParam)


class NeuronGroup(pTypes.GroupParameter):
    def __init__(self, **kwds):
        defs = dict(name="Neuron Settings", autoIncrementName=True, renamable=True, removable=True, children=[
            dict(name='Number of Areas', type='int', value=5, limits=[1, None]),
            dict(name='Spacing', type='float', value=1.0, step=0.1),
        ])
        pTypes.GroupParameter.__init__(self, **defs)
        self.restoreState(kwds, removeChildren=False)

    def generate(self):
        prog = []
        for cmd in self:
            prog.append((cmd['Proper Time'], cmd['Neuron']))
        return prog


pTypes.registerParameterType('NeuronGroup', NeuronGroup)


class Area(object):
    nAreas = 0

    def __init__(self, x0=0.0, y0=0.0, m0=1.0, v0=0.0, t0=0.0, color=None, prog=None, size=0.5):
        Area.nAreas += 1
        self.pen = pg.mkPen(color)
        self.brush = pg.mkBrush(color)
        self.y0 = y0
        self.x0 = x0
        self.v0 = v0
        self.m0 = m0
        self.t0 = t0
        self.prog = prog
        self.size = size


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