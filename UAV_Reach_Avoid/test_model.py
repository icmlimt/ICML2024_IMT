#!/usr/bin/python3

import re, sys, os, shutil, fileinput, subprocess, argparse
from dataclasses import dataclass, field
#import visvis as vv # DISABLED IN DOCKER IMAGE
import numpy as np
import argparse

from translate import translateTransitions, readLabels
#from plotting import VisVisPlotter # DISABLED IN DOCKER IMAGE
from simulation import Simulator, Verdict

import time

TEMPEST_BINARY=os.environ.get("TEMPEST_BINARY", "~/projects/tempest-devel/ranking_release/bin/storm")

def tic():
    #Homemade version of matlab tic and toc functions: https://stackoverflow.com/a/18903019
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    if 'startTime_for_tictoc' in globals():
        print("Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds.")
    else:
        print("Toc: start time not set")


def convert(tuples):
    return dict(tuples)

def getBasename(filename):
    return os.path.basename(filename)

def traFileWithIteration(filename, iteration):
    return os.path.splitext(filename)[0] + f"_{iteration:03}.tra"

def copyFile(filename, newFilename):
    shutil.copy(filename, newFilename)


def execute(command, verbose=False):
    if verbose: print(f"Executing {command}")
    os.system(command)

@dataclass(frozen=True)
class State:
    id: int
    x: float
    x_vel: float
    y: float
    y_vel: float
    z: float
    z_vel: float

def default_value():
    return {'action' : None, 'choiceValue' : None}


@dataclass(frozen=True)
class StateValue:
    ranking: float
    choices: dict = field(default_factory=default_value)

@dataclass(frozen=False)
class TestResult:
    prob_pes_min: float
    prob_pes_max: float
    prob_pes_avg: float
    prob_opt_min: float
    prob_opt_max: float
    prob_opt_avg: float
    min_min: float
    min_max: float

    def csv(self, ws=" "):
        return f"{self.prob_pes_min:0.04f}{ws}{self.prob_pes_max:0.04f}{ws}{self.prob_pes_avg:0.04f}{ws}{self.prob_opt_min:0.04f}{ws}{self.prob_opt_max:0.04f}{ws}{self.prob_opt_avg:0.04f}{ws}{self.min_min:0.04f}{ws}{self.min_max:0.04f}{ws}"

def parseStrategy(strategyFile, allStateActionPairs, time_index=0):
    strategy = dict()
    with open(strategyFile) as strategyLines:
        for line in strategyLines:
            line = line.replace("(","").replace(")","").replace("\n", "")
            explode = re.split(",|=", line)
            stateId = int(explode[0]) + 3
            if stateId < 3: continue
            if int(explode[1]) != time_index: continue
            try:
                strategy[stateId] = allStateActionPairs[stateId].index(explode[2])
            except KeyError as e:
                pass

    return strategy

def queryStrategy(strategy, stateId):
    try:
        return strategy[stateId]
    except:
        return -1


def callTempest(files, reward, bound=3):
    property_str = "!(\"failed\" | \"reached\")"
    if True:
        prop =  f"filter(min, Pmax=? [ true U<={bound} \"failed\" ], {property_str} );"
        prop += f"filter(max, Pmax=? [ true U<={bound} \"failed\" ], {property_str} );"
        prop += f"filter(avg, Pmax=? [ true U<={bound} \"failed\" ], {property_str} );"
        prop += f"filter(min, Pmin=? [ true U<={bound} \"failed\" ], {property_str} );"
        prop += f"filter(max, Pmin=? [ true U<={bound} \"failed\" ], {property_str} );"
        prop += f"filter(avg, Pmin=? [ true U<={bound} \"failed\" ], {property_str} );"
        prop += f"filter(min, Rmin=? [ C<={bound} ], {property_str}  );"
        prop += f"filter(min, Rmax=? [ C<={bound} ], {property_str}  );"
    else:
        prop =  f"filter(min, Rmin=? [ C<={bound} ], {property_str}  );"
        prop += f"filter(max, Rmin=? [ C<={bound} ], {property_str}  );"
        prop += f"filter(avg, Rmin=? [ C<={bound} ], {property_str}  );"
        prop += f"filter(min, Rmax=? [ C<={bound} ], {property_str}  );"
        prop += f"filter(max, Rmax=? [ C<={bound} ], {property_str}  );"
        prop += f"filter(avg, Rmax=? [ C<={bound} ], {property_str}  );"
    command = f"{TEMPEST_BINARY} --io:explicit {files} --io:staterew MDP_Abstraction_interval.lab.{reward} --prop '{prop}' "

    results = list()
    try:
        output = subprocess.check_output(command, shell=True).decode("utf-8").split('\n')
        for line in output:
            if "Result" in line and not len(results) >= 10:
                range_value = re.search(r"(.*:).*\[(-?\d+\.?\d*), (-?\d+\.?\d*)\].*", line)
                if range_value:
                    results.append(float(range_value.group(2)))
                    results.append(float(range_value.group(3)))
                else:
                    value = re.search(r"(.*:)(.*)", line)
                    results.append(float(value.group(2)))
    except subprocess.CalledProcessError as e:
        print(e.output)
    #results.append(-1)
    #results.append(-1)
    return TestResult(*(tuple(results)))

def parseRanking(filename, allStates):
    state_ranking = dict()
    try:
        with open(filename, "r") as f:
            filecontent = f.readlines()
        for line in filecontent:
            stateId = int(re.findall(r"^\d+", line)[0])
            values = re.findall(r":(-?\d+\.?\d*),?", line)
            ranking_value = float(values[0])
            choices = {i : float(value) for i,value in enumerate(values[1:])}
            state = allStates[stateId]
            value = StateValue(ranking_value, choices)
            state_ranking[state] = value
        if len(state_ranking) == 0: return
        all_values = [x.ranking for x in state_ranking.values()]
        max_value = max(all_values)
        min_value = min(all_values)
        new_state_ranking = {}
        for state, value in state_ranking.items():
            choices = value.choices
            try:
                new_value = (value.ranking - min_value) / (max_value - min_value)
            except ZeroDivisionError as e:
                new_value = 0.0
            new_state_ranking[state] = StateValue(new_value, choices)
        state_ranking = new_state_ranking
    except EnvironmentError:
        print("TODO file not available. Exiting.")
        sys.exit(1)
    return {state: values for state, values in sorted(state_ranking.items(), key=lambda item: item[1].ranking)}

def parseStateValuations(filename):
    all_states = dict()
    maxStateId = -1
    for i in [0,1,2]:
        dummy_values = [i] * 7
        all_states[i] = State(*dummy_values)
    with open(filename) as stateValuations:
        for line in stateValuations:
            values = re.findall(r"(-?\d+\.?\d*),?", line)
            values = [int(values[0])] + [float(v) for v in values[1:]]
            all_states[values[0]] = State(*values)
            if values[0] > maxStateId: maxStateId = values[0]
    dummy_values = [maxStateId + 1] * 7
    all_states[maxStateId + 1] = State(*dummy_values)
    return all_states

def parseResults(allStates):
    state_to_values = dict()
    with open("prob_results_maximize") as maximizer, open("prob_results_minimize") as minimizer:
        for max_line, min_line in zip(maximizer, minimizer):
            max_values = re.findall(r"(-?\d+\.?\d*),?", max_line)
            min_values = re.findall(r"(-?\d+\.?\d*),?", min_line)
            if max_values[0] != min_values[0]:
                print("min/max files do not match.")
                assert(False)
            stateId = int(max_values[0])
            min_result = float(min_values[1])
            max_result = float(max_values[1])
            value = (min_result, max_result, max_result - min_result)
            state_to_values[stateId] = value
    return state_to_values


def removeActionFromTransitionFile(stateId, chosenActionIndex, filename, iteration):
    stateIdRegex = re.compile(f"^{stateId}\s")
    for line in fileinput.input(filename, inplace = True):
        if not stateIdRegex.match(line):
            print(line, end="")
        else:
            explode = line.split(" ")
            if int(explode[1]) == chosenActionIndex:
                print(line, end="")

def removeActionsFromTransitionFile(stateActionPairsToTrim, filename, iteration):
    stateIdsRegex = re.compile("|".join([f"^{stateId}\s" for stateId, actionIndex in stateActionPairsToTrim.items()]))
    for line in fileinput.input(filename, inplace = True):
        result = stateIdsRegex.match(line)
        if not result:
            print(line, end="")
        else:
            actionIndex = stateActionPairsToTrim[int(result[0])]
            explode = line.split(" ")
            if int(explode[1]) == actionIndex:
                print(line, end="")

def getTopNStates(rankedStates, n, threshold):
    if n != 0:
        return dict(list(rankedStates.items())[-n:])
    else:
        return {state:value for state,value in rankedStates.items() if value.ranking >= threshold}

def getNRandomStates(rankedStates, n, testedStates):
    stateIds = [state.id for state in rankedStates.keys()]
    notYetTestedStates = np.array([stateId for stateId in stateIds if stateId not in testedStates])
    if len(notYetTestedStates) >= n:
        return notYetTestedStates[np.random.choice(len(notYetTestedStates), size=n, replace=False)]
    else:
        return notYetTestedStates

def main(traFile, labFile, straFile, horizonBound, refinementSteps, refinementBound, ablationTesting, plotting=False, stepwisePlotting=False):

    all_states = parseStateValuations("MDP_state_valuations")
    deadlockStates, reachedStates, maxStateId = readLabels(labFile)
    stateToActions, allStateActionPairs = translateTransitions(traFile, deadlockStates, reachedStates, maxStateId)
    strategy = parseStrategy(straFile, stateToActions)

    #if plotting: plotter = VisVisPlotter(all_states, reachedStates, deadlockStates, stepwisePlotting) # DISABLED IN DOCKER IMAGE
    #if plotting: plotter.plotScenario() # DISABLED IN DOCKER IMAGE


    copyFile("MDP_" + traFile, "MDP_" + os.path.splitext(getBasename(traFile))[0] + f"_000.tra")

    iteration = 0
    #testsPerIteration = refinementSteps
    #refinementThreshold =
    numTestedStates = 0
    totalIterations = 52

    testedStates = list()
    while iteration < totalIterations:
        print(f"{iteration:03}", end="\t")
        sys.stdout.flush()
        currentTraFile = traFileWithIteration("MDP_" + traFile, iteration)
        nextTraFile = traFileWithIteration("MDP_" + traFile, iteration+1)
        testResult = callTempest(f"{currentTraFile} MDP_{labFile}",  "saferew", horizonBound)
        state_ranking = parseRanking("action_ranking", all_states)
        copyFile("action_ranking", f"action_ranking_{iteration:03}")
        copyFile("prob_results_maximize", f"prob_results_maximize_{iteration:03}")
        copyFile("prob_results_minimize", f"prob_results_minimize_{iteration:03}")

        if not ablationTesting:
            importantStates = getTopNStates(state_ranking, refinementSteps, refinementBound)
            statesToTest = [state.id for state in importantStates.keys()]
            statesToPlot = importantStates
        else:
            statesToTest = list(getNRandomStates(state_ranking, refinementSteps, testedStates))
            testedStates += statesToTest
            statesToPlot = {all_states[stateId]:StateValue(0,{}) for stateId in statesToTest}


        copyFile(currentTraFile, nextTraFile)
        stateActionPairsToTrim = dict()
        for testState in statesToTest:
            chosenActionIndex = queryStrategy(strategy, testState)
            if chosenActionIndex != -1:
                stateActionPairsToTrim[testState] = chosenActionIndex
        stateEstimates = parseResults(all_states)
        results = [0,0,0]

        failureStates = list()
        validatedStates = list()
        for state, estimates in stateEstimates.items():
            if state in deadlockStates or state in reachedStates:
                continue
            if estimates[0] > 0.05:
                results[1] += 1
                failureStates.append(all_states[state])
                #print(f"{state}: {estimates}")
            elif estimates[1] <= 0.05:
                results[0] += 1
                validatedStates.append(all_states[state])
            else:
                results[2] += 1


        removeActionsFromTransitionFile(stateActionPairsToTrim, nextTraFile, iteration)
        print(f"{numTestedStates}\t{testResult.csv(' ')}\t{results[0]}\t{results[1]}\t{results[2]}\t{sum(results)}")
        if results[2] == 0:
            toc()
            sys.exit(0)
        numTestedStates += len(statesToTest)
        iteration += 1

        if plotting: plotter.plotStates(failureStates, coloring=(0.8,0.0,0.0,0.6), removeMeshes=True)
        if plotting: plotter.plotStates(validatedStates, coloring=(0.0,0.8,0.0,0.6))
        if plotting: plotter.takeScreenshot(iteration, prefix="stepwise_0.05")

def randomTesting(traFile, labFile, straFile, bound, maxQueries, plotting=False):
    all_states = parseStateValuations("MDP_state_valuations")
    deadlockStates, reachedStates, maxStateId = readLabels(labFile)
    stateToActions, allStateActionPairs = translateTransitions(traFile, deadlockStates, reachedStates, maxStateId)
    strategy = parseStrategy(straFile, stateToActions)

    #if plotting: plotter = VisVisPlotter(all_states, reachedStates, deadlockStates, stepwisePlotting)  # DISABLED IN DOCKER IMAGE
    #if plotting: plotter.plotScenario() # DISABLED IN DOCKER IMAGE

    passingStates = list()

    randomTestingSimulator = Simulator(allStateActionPairs, strategy, deadlockStates, reachedStates, bound)
    i = 0
    print("Starting with random testing.")
    numQueries = 0
    failureStates = list()
    while numQueries <= maxQueries:
        if i >= 500:
            if plotting: plotter.plotStates(failureStates, coloring=(0.8,0.0,0.0,0.6))
            if plotting: plotter.takeScreenshot(iteration, prefix="random_testing")
            if plotting: plotter.turnCamera()
            if plotting: input("")
            print(f"{numQueries} {len(failureStates)} ")
            i = 0
        testCase, testResult, queriesForThisTestCase = randomTestingSimulator.runTest()
        i += queriesForThisTestCase
        numQueries += queriesForThisTestCase
        stateValuation = all_states[testCase]
        if testResult == Verdict.FAIL:
            failureStates.append(stateValuation)

    print(f"{numQueries} {len(failureStates)} ")

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tra', type=str, required=True,  help='Path to .tra file.')
    parser.add_argument('--lab', type=str, required=True,  help='Path to .lab file.')
    parser.add_argument('--rew', type=str, required=True,  help='Path to .rew file.')
    parser.add_argument('--stra', type=str, required=True, help='Path to strategy file.')

    refinement = parser.add_mutually_exclusive_group(required=True)
    refinement.add_argument('--refinement-steps', type=int,   default=0, help='Amount of refinement steps per iteration, mutually exclusive with refinement-bound.')
    refinement.add_argument('--refinement-bound', type=float, default=0, help='Threshold value for states to be tested, mutually exclusive with refinement-steps.')

    parser.add_argument('--bound', type=int, required=False, default=3, help='(optional) Safety Horizon Bound, defaults to 3.')
    parser.add_argument('--threshold', type=float, required=False, default=0.05, help='(optional) Safety Threshold, defaults to 0.05.')

    random_testing = parser.add_mutually_exclusive_group()
    random_testing.add_argument('-a', '--ablation', action='store_true', help="(optional) Run ablation testing for the importance ranking, i.e. model-based random testing.")
    random_testing.add_argument('-r', '--random', type=int, default=0, help='(optional) The amount of queries allowed for random testing.')

    parser.add_argument('-p', '--plotting', action='store_true', help='(optional) Enable plotting.')
    parser.add_argument('--stepwise', action='store_true', help='(optional) Remove states before plotting the next iteration.')
    return parser.parse_args()

if __name__ == '__main__':
    args = parseArgs()

    traFile = args.tra
    labFile = args.lab
    straFile = args.stra
    rewFile = args.rew

    ablationTesting = args.ablation
    plotting = args.plotting
    stepwisePlotting = args.stepwise

    maxQueriesForRandomTesting = args.random
    horizonBound = args.bound

    refinementSteps = args.refinement_steps
    refinementBound = args.refinement_bound

    tic()
    try:
        if maxQueriesForRandomTesting == 0: #awkward way to test for this...
            main(traFile, labFile, straFile, horizonBound, refinementSteps, refinementBound, ablationTesting, plotting, stepwisePlotting)
        else:
            randomTesting(traFile, labFile, straFile, horizonBound, maxQueriesForRandomTesting, plotting)

    except SystemExit as e:
        pass
    except Exception as e:
        print(e)
    toc() ## FIXME!
