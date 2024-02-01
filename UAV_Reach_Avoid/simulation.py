import sys
from enum import Flag, auto

import numpy as np


class Verdict(Flag):
    INCONCLUSIVE = auto()
    PASS = auto()
    FAIL = auto()



class Simulator():
    def __init__(self, allStateActionPairs, strategy, deadlockStates, reachedStates, bound=3, numSimulations=1):
        self.allStateActionPairs = { ( pair.state_id, pair.action_id ) : pair.next_state_probabilities for pair in allStateActionPairs }
        self.strategy = strategy
        self.deadlockStates = deadlockStates
        self.reachedStates = reachedStates

        #print(f"Deadlock: {self.deadlockStates}")
        #print(f"GoalStates: {self.reachedStates}")


        self.bound = bound
        self.numSimulations = numSimulations

        allStates = set([state.state_id for state in allStateActionPairs])
        allStates = allStates.difference(set(deadlockStates))
        allStates = allStates.difference(set(reachedStates))
        self.allStates = np.array(list(allStates))

    def _pickRandomTestCase(self):
        testCase = np.random.choice(self.allStates, 1)[0]
        #self.allStates = np.delete(self.allStates, testCase)
        return testCase

    def _simulate(self, initialStateId):
        i = 0

        actionId = self.strategy[initialStateId]
        nextStatePair = (initialStateId, actionId)

        while i < self.bound:
            i += 1
            nextStateProbabilities = self.allStateActionPairs[nextStatePair]
            weights = list()
            nextStateIds = list()
            for nextStateId, probability in nextStateProbabilities.items():
                weights.append(probability)
                nextStateIds.append(nextStateId)
            nextStateId = np.random.choice(nextStateIds, 1, p=weights)[0]
            if nextStateId in self.deadlockStates:
                return Verdict.FAIL, i
            if nextStateId in self.reachedStates:
                return Verdict.PASS, i
            nextStatePair = (nextStateId, self.strategy[nextStateId])
        return Verdict.INCONCLUSIVE, i

    def runTest(self):
        testCase = self._pickRandomTestCase()

        histogram = [0,0,0]
        for i in range(self.numSimulations):
            result, numQueries = self._simulate(testCase)
            if result == Verdict.FAIL:
                return testCase, Verdict.FAIL, numQueries
            return testCase, Verdict.INCONCLUSIVE, numQueries
