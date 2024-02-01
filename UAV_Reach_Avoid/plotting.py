#!/usr/bin/python3

import visvis as vv
import numpy as np

import time

def translateArrayToVV(array):
    scaling = (float(array[0][1]) - float(array[0][0]), float(array[2][1]) - float(array[2][0]), float(array[4][1]) - float(array[4][0]))
    translation = (float(array[0][0]) + scaling[0] * 0.5, float(array[2][0]) + scaling[0] * 0.5, float(array[4][0]) + scaling[0] * 0.5)
    return translation, scaling

class VisVisPlotter():
    def __init__(self, stateValuations, goalStates, deadlockStates, stepwise):
        self.app = vv.use()
        self.fig = vv.clf()
        self.ax = vv.cla()
        self.stateColor = (0,0,0.75,0.4)
        self.failColor = (0.8,0.8,0.8,1.0)  # (1,0,0,0.8)
        self.goalColor = (0,1,0,1.0)
        self.stateScaling = (2,2,2) #(.5,.5,.5)

        self.plotedStates = set()
        self.plotedMeshes = set()
        self.stepwise = stepwise

        auxStates = [0,1,2,25518]
        self.goals = set([(stateValuation.x, stateValuation.y, stateValuation.z) for stateId, stateValuation in stateValuations.items() if stateId in goalStates and not stateId in auxStates])
        self.fails = set([(stateValuation.x, stateValuation.y, stateValuation.z) for stateId, stateValuation in stateValuations.items() if stateId in deadlockStates and not stateId in auxStates])

    def plotScenario(self, saveScreenshot=True):
        for goal in self.goals:
            state = vv.solidBox(goal, scaling=self.stateScaling)
            state.faceColor = self.goalColor
        for fail in self.fails:
            state = vv.solidBox(fail, scaling=self.stateScaling)
            state.faceColor = self.failColor

        self.ax.SetLimits((-16,16),(-10,10),(-7,7))
        self.ax.SetView({'zoom':0.025, 'elevation':20, 'azimuth':30})
        if saveScreenshot: vv.screenshot("000.png", sf=3, bg='w', ob=vv.gcf())

    def run(self):
        self.ax.SetLimits((-16,16),(-10,10),(-7,7))
        self.app.Run()

    def clear(self):
        axes = vv.gca()
        axes.Clear()

    def plotStates(self, states, coloring="", removeMeshes=False):
        if self.stepwise and removeMeshes:
            self.clear()
            self.plotScenario(saveScreenshot=False)
        if not coloring:
            coloring = self.stateColor
        plotedRegions = set()
        for state in states:
            if state in self.plotedStates: continue # what are the implications of this?
            coordinates = (state.x, state.y, state.z)
            print(f"plotting {state} at {coordinates}")
            if coordinates in plotedRegions: continue
            state = vv.solidBox(coordinates, scaling=(1,1,1))#(0.5,0.5,0.5))
            state.faceColor = coloring
            plotedRegions.add(coordinates)
            if self.stepwise:
                self.plotedMeshes.add(state)
        self.plotedStates.update(states)

    def takeScreenshot(self, iteration, prefix=""):
        self.ax.SetLimits((-16,16),(-10,10),(-7,7))
        #config = [(90, 00), (45, 00), (0, 00), (-45, 00), (-90, 00)]
        config = [(45, -150), (45, -100), (45, -60), (45, 60), (45, 120)]
        for elevation, azimuth in config:
            filename = f"{prefix}{'_' if prefix else ''}{iteration:03}_{elevation}_{azimuth}.png"
            print(f"Saving Screenshot to {filename}")
            self.ax.SetView({'zoom':0.025, 'elevation':elevation, 'azimuth':azimuth})
            vv.screenshot(filename, sf=3, bg='w', ob=vv.gcf())

    def turnCamera(self):
        config = [(45, -150), (45, -100), (45, -60), (45, 60), (45, 120)]
        self.ax.SetLimits((-16,16),(-10,10),(-7,7))
        for elevation, azimuth in config:
            self.ax.SetView({'zoom':0.025, 'elevation':elevation, 'azimuth':azimuth})


if __name__ == '__main__':
    main()
