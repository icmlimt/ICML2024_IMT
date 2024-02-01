#!/usr/bin/python3

import sys
#import re
import json
import numpy as np
from collections import deque

from dataclasses import dataclass


@dataclass(frozen=False)
class StateAction:
    state_id: int
    action_id: int
    next_state_probabilities: dict
    action_name: str

    def normalizeDistribution(self):
        weight = np.sum([value for key, value in self.next_state_probabilities.items()])
        self.next_state_probabilities = { next_state_id : (probability / weight) for next_state_id, probability in self.next_state_probabilities.items() }


def translateTransitions(traFile, deadlockStates, reachedStates, maxStateId):
    current_state_id = -1
    current_action_id = -1
    all_state_action_pairs = list()
    with open(traFile) as transitions:
        next(transitions)
        for line in transitions:
            line = line.replace("\n","")
            explode = line.split(" ")
            if len(explode) < 2: continue
            interval = json.loads(explode[3])
            probability = (interval[0] + interval[1])/2
            state_id = int(explode[0])
            action_id = int(explode[1])
            next_state_id = int(explode[2])
            if len(explode) >= 5:
                action_name = explode[4]
            else:
                action_name = ""
            #print(f"State : {state_id} with action {action_id} leads with {probability} to {next_state_id}.")
            if state_id in [0, 1, 2]:
                continue

            next_state_probabilities = {next_state_id: probability}
            if current_state_id != state_id:
                new_state_action_pair = StateAction(state_id, action_id, next_state_probabilities, action_name)
                all_state_action_pairs.append(new_state_action_pair)
                current_state_id = state_id
                current_action_id = action_id

            elif current_action_id != action_id:
                new_state_action_pair = StateAction(state_id, action_id, next_state_probabilities, action_name)
                all_state_action_pairs.append(new_state_action_pair)
                current_action_id = action_id
            else:
                all_state_action_pairs[-1].next_state_probabilities[next_state_id] = probability

    # we need to sort the deadlock and reached states to insert them while building the .tra file
    deadlockStates = [(state, 0) for state in deadlockStates]
    reachedStates = [(state, maxStateId) for state in reachedStates]
    final_states = deadlockStates + reachedStates
    final_states = deque(sorted(final_states, key=lambda tuple: tuple[0], reverse=True))

    with open("MDP_" + traFile, "w") as new_transitions_file:
        new_transitions_file.write("mdp\n")
        new_transitions_file.write(f"0 0 {maxStateId} 1.0\n")
        for entry in all_state_action_pairs:
            entry.normalizeDistribution()
            source_state = int(entry.state_id)
            while final_states and int(final_states[-1][0]) < source_state:
                final_state = final_states.pop()
                if int(final_state[0]) == 0: continue
                new_transitions_file.write(f"{final_state[0]} 0 {final_state[1]} 1.0\n")
            for next_state_id, probability in entry.next_state_probabilities.items():
                new_transitions_file.write(f"{entry.state_id} {entry.action_id} {next_state_id} {probability}\n")
        new_transitions_file.write(f"{maxStateId} 0 {maxStateId} 1.0\n")

    state_to_actions = dict()
    for state_action_pair in all_state_action_pairs:
        if state_action_pair.state_id in state_to_actions:
            state_to_actions[state_action_pair.state_id].append(state_action_pair.action_name)
        else:
            state_to_actions[state_action_pair.state_id] = [state_action_pair.action_name]
    return state_to_actions, all_state_action_pairs

def readLabels(labFile):
    deadlockStates = list()
    reachedStates = list()
    with open(labFile) as states:
        newLabFile = "MDP_" + labFile
        newLabels = open(newLabFile, "w")
        optRewards = open(newLabFile + ".optrew", "w")
        safetyRewards = open(newLabFile + ".saferew", "w")
        labels = ["init", "deadlock", "reached", "failed"]
        next(states)
        newLabels.write("#DECLARATION\ninit deadlock reached failed\n#END\n")
        maxStateId = -1
        for line in states:
            line = line.replace(":","").replace("\n", "")
            explode = line.split(" ")
            newLabel = f"{explode[0]} "
            if int(explode[0]) > maxStateId: maxStateId = int(explode[0])
            if int(explode[0]) == 0:
                safetyRewards.write(f"{explode[0]} -100\n")
                optRewards.write(f"{explode[0]} -100\n")
            #if "3" in explode:
            #    safetyRewards.write(f"{explode[0]} -100\n")
            #    optRewards.write(f"{explode[0]} -100\n")
            elif "2" in explode:
                optRewards.write(f"{explode[0]} 100\n")
            else:
                optRewards.write(f"{explode[0]} -1\n")
            for labelIndex in explode[1:]:
                # sink states should not be deadlock states anymore:
                if labelIndex == "1":
                    deadlockStates.append(int(explode[0]))
                    continue
                if labelIndex == "2":
                    reachedStates.append(int(explode[0]))
                    continue
                newLabel += f"{labels[int(labelIndex)]} "
            newLabels.write(newLabel + "\n")
        return deadlockStates, reachedStates, maxStateId + 1

def main(traFile, labFile):
    deadlockStates, reachedStates, maxStateId = readLabels(labFile)
    translateTransitions(traFile, deadlockStates, reachedStates, maxStateId)

if __name__ == '__main__':
    traFile = sys.argv[1]
    labFile = sys.argv[2]
    main(traFile, labFile)
