import gym
from stable_baselines3 import DQN
from gym_minigrid.wrappers import RGBImgObsWrapper, ImgObsWrapper, MiniWrapper
from gym_minigrid.minigrid import isSlippery, Goal, isOneWay
from subprocess import call
from os import listdir, system
from os.path import isfile, join, getctime
import argparse
from gym_minigrid.policyRepairEnv import *
from statistics import mean

import time, re, sys, csv, os
from pathlib import Path
from copy import deepcopy

import subprocess
LOG_MODE = True

TEMPEST_BINARY=os.environ.get("TEMPEST_BINARY", "~/projects/tempest-devel/ranking_release/bin/storm")


@dataclass(frozen=False)
class StateSet:
    iteration: int
    failureStates = set()
    successStates = set()
    undecidedStates = set()
    tested = set()
    totalTestStates = int
    def addFailureState(self, state):
        self.failureStates.add(state)
    def addSuccessState(self, state):
        self.successStates.add(state)
    def addUndecidedState(self, state):
        self.undecidedStates.add(state)
    def addTestedStates(self, states):
        self.tested.update(states)
    def addFailureStates(self, states):
        self.failureStates.update(states)
    def addSuccessStates(self, states):
        self.successStates.update(states)
    def addUndecidedStates(self, states):
        self.undecidedStates.update(states)

    def __str__(self):
        state = f"Iteration: {self.iteration:04}"
        counter = [0] * 5
        Sft = "\tS_ft = "
        Sfi = "\tS_fi = "
        Sst = "\tS_st = "
        Ssi = "\tS_si = "
        Stu = "\tS_tu = "
        for testedState in self.tested:
            if testedState in self.failureStates:
                Sft += f"\t{testedState}"
                counter[0] += 1
            elif testedState in self.successStates:
                Sst += f"\t{testedState}"
                counter[2] += 1
            else:
                Stu += f"\t{testedState}"
                counter[4] += 1
        for failureState in self.failureStates:
            if not failureState in self.tested:
                Sfi += f"\t{failureState}\t"
                counter[1] += 1
        for successState in self.successStates:
            if not successState in self.tested:
                Ssi += f"\t{successState}\t"
                counter[3] += 1
        return f"{state}\n{counter[0]}{Sft}\n{counter[1]}{Sfi}\n{counter[2]}{Sst}\n{counter[3]}{Ssi}\n{counter[4]}{Stu}"

    def stateStatistics(self):
        return (self.totalTestStates - len(self.failureStates) - len(self.successStates), len(self.failureStates), len(self.successStates))


@dataclass(frozen=False)
class TestResult:
    init_check_pes_min: float
    init_check_pes_max: float
    init_check_pes_avg: float
    init_check_opt_min: float
    init_check_opt_max: float
    init_check_opt_avg: float
    all_states_tested: float
    count_undecided_states: int
    count_proven_failure_states: int
    count_proven_good_states: int
    num_queries: int

    def __str__(self):
        return f"""Test Result:
    init_check_pes_min: {self.init_check_pes_min}
    init_check_pes_max: {self.init_check_pes_max}
    init_check_pes_avg: {self.init_check_pes_avg}
    init_check_opt_min: {self.init_check_opt_min}
    init_check_opt_max: {self.init_check_opt_max}
    init_check_opt_avg: {self.init_check_opt_avg}
    all_tested: {"True" if self.all_states_tested == 1.0 else "False"}
    count_undecided_states (for previous iteration):  {self.count_undecided_states}
    count_proven_failure_states (for this iteration): {self.count_proven_failure_states}
    count_proven_good_states (for this iteration):    {self.count_proven_good_states}
"""
    def csv(self, ws=" "):
        return f"{self.init_check_pes_min:0.04f}{ws}{self.init_check_pes_max:0.04f}{ws}{self.init_check_pes_avg:0.04f}{ws}{self.init_check_opt_min:0.04f}{ws}{self.init_check_opt_max:0.04f}{ws}{self.init_check_opt_avg:0.04f}{ws}{int(self.count_undecided_states)}{ws}{int(self.count_proven_failure_states)}{ws}{int(self.count_proven_good_states)}{ws}{self.num_queries}"

    def updateStateStatistics(self, stats, num_queries):
        self.count_undecided_states = stats[0]
        self.count_proven_failure_states = stats[1]
        self.count_proven_good_states = stats[2]
        self.num_queries = num_queries

    @staticmethod
    def csv_header(ws=" "):
        return f"iteration{ws}pessimistic_min{ws}pessimistic_max{ws}pessimistic_avg{ws}optimistic_min{ws}optimistic_max{ws}optimistic_avg{ws}undecided_states{ws}proven_failure{ws}proven_good{ws}num_queries"

def min_mean_max(list):
    if not list: return ""
    return min(list), mean(list), max(list)

def move_files(envname, iterations, directory, plotting):
    files_to_move_regex = re.compile("(results_m.*|action_ranking.*|.*prism|.*heatmap.*|.*fixed.*|.*tested_states.*|.*tested_tiles.*|\d+.png|full.png|output.csv|boilerplate_tikz.pdf)$")
    if plotting: concat_images(envname, iterations)
    if os.path.exists(directory):
        print(f"{directory} already exists, please move files manually")
        return
    os.makedirs(directory)
    for f in os.listdir('.'):
        if files_to_move_regex.match(f):
            #print(f"Moving {f} to {directory}/{f}")
            try:
                Path(f).rename(f"{directory}/{f}")
            except Exception as e:
                print(e)

def concat_images(envname, iterations):
    for i in range(0, iterations):
        command = f"montage {envname}*{i:03}*heatmap_C*png {envname}*{i:03}*fixed_map*png {envname}*{i+1:03}*tested_*png  -tile 1x -geometry +0+0 {i:03}.png"
        print(f"Executing {command}")
        system(command)
    iterations -= 1
    command = f"montage <000-{iterations-1:03}>.png  -tile x1 -geometry +0+0 full.png"
    print(f"Executing {command}")
    system(command)


def compare_min_max(iteration):
    state_to_values = {}
    done_states = list()
    try:
        with open(f"results_maximize", "r") as f:
            max_lines = f.readlines()
        with open(f"results_minimize", "r") as f:
            min_lines = f.readlines()
        for max_line, min_line in zip(max_lines, min_lines):
            stateMapping = convert(re.findall(r"([a-zA-Z]*[a-zA-Z])=(\d+)?", max_line))
            min_stateMapping = convert(re.findall(r"([a-zA-Z]*[a-zA-Z])=(\d+)?", min_line))
            if stateMapping != min_stateMapping:
                print("min/max files do not match.")
                sys.exit(1)
            if "[AgentDone" in max_line:
                done_states.append(State(int(stateMapping["xAgent"]), int(stateMapping["yAgent"]), int(stateMapping["viewAgent"])))
                continue
            max_result = float(re.search(r"Result:([+-]?(\d*\.\d+)|\d+)", max_line)[0].replace("Result:",""))
            min_result = float(re.search(r"Result:([+-]?(\d*\.\d+)|\d+)", min_line)[0].replace("Result:",""))
            state = State(int(stateMapping["xAgent"]), int(stateMapping["yAgent"]), int(stateMapping["viewAgent"]))
            value = (min_result, max_result, max_result - min_result)
            state_to_values[state] = value
        for state in done_states:
            del state_to_values[state]
    except EnvironmentError:
        print("Error: file not available. Exiting.")
        sys.exit(1)
    system(f"mv results_maximize results_maximize_{iteration}")
    system(f"mv results_minimize results_minimize_{iteration}")
    return state_to_values



def translate_grid_to_prism(env, filename, oneways=False):
    with open(f"{filename}.grid", "w") as infile:
        infile.write(env.printGrid(init=True))
    command = f"./Minigrid2PRISM/build/main -i {filename}.grid -o {filename}.prism -v 'agent'"
    if oneways: command += " -f"
    LOG(f"Executing '{command}'")
    system(command)

def call_tempest(filename, bound, rewardStructure, threshold, use_docker=False, safety=False):
    if True:
        if safety:
            prop =  f"filter(min, Pmin=? [ G !\"AgentIsInLava\" ], !\"AgentIsInLava\" & !\"AgentIsInGoal\" );"
            prop += f"filter(max, Pmin=? [ G !\"AgentIsInLava\" ], !\"AgentIsInLava\" & !\"AgentIsInGoal\" );"
            prop += f"filter(avg, Pmin=? [ G !\"AgentIsInLava\" ], !\"AgentIsInLava\" & !\"AgentIsInGoal\" );"
            prop += f"filter(min, Pmax=? [ G !\"AgentIsInLava\" ], !\"AgentIsInLava\" & !\"AgentIsInGoal\" );"
            prop += f"filter(max, Pmax=? [ G !\"AgentIsInLava\" ], !\"AgentIsInLava\" & !\"AgentIsInGoal\" );"
            prop += f"filter(avg, Pmax=? [ G !\"AgentIsInLava\" ], !\"AgentIsInLava\" & !\"AgentIsInGoal\" );"
        else:
            property_str = "!(\"AgentCannotTurn\" |\"AgentIsOnOneWay\") & !(xAgent=25&yAgent=25)"
            #prop =  f"filter(min, R{{\"{rewardStructure}\"}}min=? [ C<={bound} ], !\"AgentIsInLava\" & !\"AgentIsInGoal\" );"
            #prop += f"filter(max, R{{\"{rewardStructure}\"}}min=? [ C<={bound} ], !\"AgentIsInLava\" & !\"AgentIsInGoal\" );"
            #prop += f"filter(avg, R{{\"{rewardStructure}\"}}min=? [ C<={bound} ], !\"AgentIsInLava\" & !\"AgentIsInGoal\" );"
            #prop += f"filter(min, R{{\"{rewardStructure}\"}}max=? [ C<={bound} ], !\"AgentIsInLava\" & !\"AgentIsInGoal\" );"
            #prop += f"filter(max, R{{\"{rewardStructure}\"}}max=? [ C<={bound} ], !\"AgentIsInLava\" & !\"AgentIsInGoal\" );"
            #prop += f"filter(avg, R{{\"{rewardStructure}\"}}max=? [ C<={bound} ], !\"AgentIsInLava\" & !\"AgentIsInGoal\" );"
            prop =  f"filter(min, R{{\"{rewardStructure}\"}}min=? [ C<={bound} ], {property_str}  );"
            prop += f"filter(max, R{{\"{rewardStructure}\"}}min=? [ C<={bound} ], {property_str}  );"
            prop += f"filter(avg, R{{\"{rewardStructure}\"}}min=? [ C<={bound} ], {property_str}  );"
            prop += f"filter(min, R{{\"{rewardStructure}\"}}max=? [ C<={bound} ], {property_str}  );"
            prop += f"filter(max, R{{\"{rewardStructure}\"}}max=? [ C<={bound} ], {property_str}  );"
            prop += f"filter(avg, R{{\"{rewardStructure}\"}}max=? [ C<={bound} ], {property_str}  );"
        prop += f"filter(forall, Pmax>=1 [ \"decidedStates\" ], !\"AgentIsInLava\" & !\"AgentIsInGoal\");"
        prop += f"filter(count, Pmax>=1 [ !\"decidedStates\" ], !\"AgentIsInLava\" & !\"AgentIsInGoal\");"
        prop += f"filter(count, R{{\"{rewardStructure}\"}}max<{threshold} [ C<={bound} ], !\"AgentIsInLava\" & !\"AgentIsInGoal\");"
        prop += f"filter(count, R{{\"{rewardStructure}\"}}min>= {threshold} [ C<={bound} ], !\"AgentIsInLava\" & !\"AgentIsInGoal\");"
        prop += f"filter(count, Pmax>=1 [ \"decidedStates\" ], !\"AgentIsInLava\" & !\"AgentIsInGoal\");"
        prop += f"R{{\"{rewardStructure}\"}}max=? [ C<={bound} ];"
        command = f"{TEMPEST_BINARY} --prism {filename}.prism --prop '{prop}' --buildchoicelab --buildstateval --build:explchecks; mv action_ranking action_ranking_{bound}"
    #LOG(f"Executing '{command}'")

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
    results.append(0) # placeholder for num_queries
    return TestResult(*tuple(results))

def plot_heatmaps(env, name, bound, rewardStructure, previousFixedStates, fixedStates, statesValuesDict, nr_bins=1, iteration=0):
    #env.printHeatMap(name + f"_{rewardStructure}_{iteration:03}", bound, nr_bins)
    #env.printHeatMapReduced(name + f"_{rewardStructure}_{iteration:03}", bound, nr_bins)
    previousFixedStates = env.printFixedMap(name + f"_{rewardStructure}_{iteration:03}", bound, previousFixedStates, fixedStates)
    system(f"mv action_ranking_{bound} action_ranking_{bound}_{iteration:03}")
    return previousFixedStates

def simulate_from_state(env, policy, state=None, n=20, visualize=False):
    if visualize: env.render(mode="human")
    #input("Hit Enter to simulate 20 steps...")
    observation = env.reset(state=state)
    action, info = policy.predict(observation, deterministic=True)
    for i in range(0,n):
        action = env.actions(action)
        observation, _, terminated, info = env.step(action)
        action, _ = policy.predict(observation, deterministic=True)
        if visualize: env.render(mode="human")
        # If we test for safety, lava is a safety violation
        # If we test for performance, running into lava within the time bound is a performance violation, since the agent cannot reach the goal
        if terminated and info["ran_into_lava"]:
            return True, i+1
        # Reaching the goal means no safety violation, or no performance violation when within the time budget
        if terminated and info["reached_goal"]:
            return False, i+1
    return not info["reached_goal"] or info['ran_into_lava'], i+1

def test_important_state(env, policy, state, state_actions_to_trim=None, visualize=False):
    #print(f"Testing state {state}...")
    #print(f"Testing: state: {state}... \tAction:", end=""); sys.stdout.flush()
    observation = env.reset(state=state)
    action, info = policy.predict(observation, deterministic=True)
    #print(f"{env.Actions(action).name}, info: {info}")
    if state_actions_to_trim is not None:
        state_actions_to_trim.append((state,action))
    #env.check_if_best_action(action, state)
    if visualize:
        simulate_from_state(env, policy, state, action)
    return state_actions_to_trim

def test_all_states(env, policy, important_states, visualize=False):
    state_actions_to_trim = []
    for state in reversed(important_states):
        state_actions_to_trim = test_important_state(env, policy, state, state_actions_to_trim, visualize)
    return state_actions_to_trim

def update_prism_file(env, name, state_actions_to_trim, decided_states, out):
    fixed_formula, fixed_updates = "formula AgentIsFixed = ", list()
    slippery_left_formula, slippery_left_updates = "formula AgentSlipperyTurnLeftAllowed = ", list()
    slippery_right_formula, slippery_right_updates = "formula AgentSlipperyTurnRightAllowed = ", list()
    slippery_forward_formula, slippery_forward_updates = "formula AgentSlipperyMoveForwardAllowed = ", list()

    windy_left_formula, windy_left_updates = "formula AgentWindyTurnLeftAllowed = ", list()
    windy_right_formula, windy_right_updates = "formula AgentWindyTurnRightAllowed = ", list()
    windy_forward_formula, windy_forward_updates = "formula AgentWindyMoveForwardAllowed = ", list()
    windy_standstill_formula, windy_standstill_updates = "formula AgentWindyStandstillAllowed = ", list()
    commands = []
    for sa in state_actions_to_trim:
        state_string = f"(xAgent={sa[0].pos_x}&yAgent={sa[0].pos_y}&viewAgent={sa[0].dir})"
        cell = env.grid.get(sa[0].pos_x,sa[0].pos_y)
        if isinstance(cell, Goal): continue
        action_name = env.Actions(sa[1]).name
        #print(f"Fixing action at {sa[0]} to {sa[1]}({action_name}), type = {type(cell)}")
        if not isSlippery(cell): # and not isWindy(cell):
            fixed_updates.append(state_string)
            command =  f"[fixed_{action_name}] xAgent={sa[0].pos_x}&yAgent={sa[0].pos_y}&viewAgent={sa[0].dir} "
            if action_name == "left":
                if sa[0].dir > 0:
                    update = (sa[0].dir - 1) % 4
                else:
                    update=3
                command += f" -> (viewAgent'={update});"
            elif action_name == "right":
                update = (sa[0].dir + 1) % 4
                command += f" -> (viewAgent'={update});"
            elif action_name == "forward":
                if sa[0].dir == 0:
                    if sa[0].pos_x < env.grid.width - 2:
                        command += f"&!AgentCannotMoveEast -> (xAgent'=xAgent+1);"
                    if sa[0].pos_x == env.grid.width - 2:
                        command += " -> true;"
                elif sa[0].dir == 1:
                    if sa[0].pos_y < env.grid.height - 2:
                        command += f"&!AgentCannotMoveSouth -> (yAgent'=yAgent+1);"
                    if sa[0].pos_y == env.grid.height - 2:
                        command += " -> true;"
                elif sa[0].dir == 2:
                    if sa[0].pos_x > 1:
                        command += f"&!AgentCannotMoveWest -> (xAgent'=xAgent-1);"
                    if sa[0].pos_x == 1:
                        command += " -> true;"
                elif sa[0].dir == 3:
                    if sa[0].pos_y > 1:
                        command += f"&!AgentCannotMoveNorth -> (yAgent'=yAgent-1);"
                    if sa[0].pos_y == 1:
                        command += " -> true;"
            else:
                command += " -> true;"
            commands.append(command)
        elif isSlippery(cell):
            if action_name == "left":
                slippery_forward_updates.append(f"!{state_string}")
                slippery_right_updates.append(f"!{state_string}")
            elif action_name == "right":
                slippery_forward_updates.append(f"!{state_string}")
                slippery_left_updates.append(f"!{state_string}")
            elif action_name == "forward":
                slippery_left_updates.append(f"!{state_string}")
                slippery_right_updates.append(f"!{state_string}")
            else:
                slippery_forward_updates.append(f"!{state_string}")
                slippery_left_updates.append(f"!{state_string}")
                slippery_right_updates.append(f"!{state_string}")
        #elif isWindy(cell):
        #    if action_name == "left":
        #        windy_forward_updates.append(f"!{state_string}")
        #        windy_right_updates.append(f"!{state_string}")
        #        windy_standstill_updates.append(f"!{state_string}")
        #    elif action_name == "right":
        #        windy_forward_updates.append(f"!{state_string}")
        #        windy_left_updates.append(f"!{state_string}")
        #        windy_standstill_updates.append(f"!{state_string}")
        #    elif action_name == "forward":
        #        windy_left_updates.append(f"!{state_string}")
        #        windy_right_updates.append(f"!{state_string}")
        #        windy_standstill_updates.append(f"!{state_string}")
        #    else:
        #        windy_forward_updates.append(f"!{state_string}")
        #        windy_left_updates.append(f"!{state_string}")
        #        windy_right_updates.append(f"!{state_string}")
    fixed_formula += " | ".join(fixed_updates)
    slippery_left_formula += " & ".join(slippery_left_updates)
    slippery_right_formula += " & ".join(slippery_right_updates)
    slippery_forward_formula += " & ".join(slippery_forward_updates)
    #windy_left_formula += " & ".join(windy_left_updates)
    #windy_right_formula += " & ".join(windy_right_updates)
    #windy_forward_formula += " & ".join(windy_forward_updates)
    #windy_standstill_formula += " & ".join(windy_standstill_updates)

    decided_states_label = "label \"decidedStates\" = ("
    first = True
    for decided_states_iter, decided_state in enumerate(decided_states):
        if first:
            first = False
        else:
            if decided_states_iter % 50 == 0:
                decided_states_label += " )|( "
            else:
                decided_states_label += " | "
        decided_states_label += f"(xAgent={decided_state.pos_x}&yAgent={decided_state.pos_y}&viewAgent={decided_state.dir})"
    decided_states_label += ");\n"
    with open(f'{name}.prism', 'r') as file :
      filedata = file.read()
    filedata = filedata.replace('endmodule', "\n".join(commands) + "\n\nendmodule\n")
    if len(fixed_updates) > 0: filedata = re.sub(r"^formula AgentIsFixed = ", fixed_formula + " |", filedata, flags=re.MULTILINE)
    if len(slippery_left_updates) > 0: filedata = re.sub(r"^formula AgentSlipperyTurnLeftAllowed = ", slippery_left_formula + " &", filedata, flags=re.MULTILINE)
    if len(slippery_right_updates) > 0: filedata = re.sub(r"^formula AgentSlipperyTurnRightAllowed = ", slippery_right_formula + " &", filedata, flags=re.MULTILINE)
    if len(slippery_forward_updates) > 0: filedata = re.sub(r"^formula AgentSlipperyMoveForwardAllowed = ", slippery_forward_formula + " &", filedata, flags=re.MULTILINE)
    if len(decided_states) > 0: filedata = re.sub(r"^label \"decidedStates\" =.*;", decided_states_label, filedata, flags=re.MULTILINE)
    #if len(windy_left_updates) > 0: filedata = re.sub(r"^formula AgentWindyTurnLeftAllowed = ", windy_left_formula + " &", filedata, flags=re.MULTILINE)
    #if len(windy_right_updates) > 0: filedata = re.sub(r"^formula AgentWindyTurnRightAllowed = ", windy_right_formula + " &", filedata, flags=re.MULTILINE)
    #if len(windy_forward_updates) > 0: filedata = re.sub(r"^formula AgentWindyMoveForwardAllowed = ", windy_forward_formula + " &", filedata, flags=re.MULTILINE)
    #if len(windy_standstill_updates) > 0: filedata = re.sub(r"^formula AgentWindyStandstillAllowed = ", windy_standstill_formula + " &", filedata, flags=re.MULTILINE)
    with open(f'{out}.prism', 'w') as file:
      file.write(filedata)
# ........................................................................... #

def LOG(text : str) -> None:
    if (LOG_MODE):
        print(text)

# ........................................................................... #

def main():
    LOG("> Start Application ...")

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, required=True, help='Environment (for now only "Testing-v0" possible)')
    parser.add_argument('--dir', type=str, required=False, help='directory where models are stored')
    parser.add_argument('--dest', type=str, required=False, default="", help='(optional) directory where results are stored, suffix for fixed directory name, daefault=""')
    parser.add_argument('--policy', type=str, required=False, help='(optional) filename of the trained model you want to test\nif not specified load most recent file in dir')
    parser.add_argument('--reward', type=str, required=False, default="SafetyNoBFS", help='(optional) Reward Structure, defaults to SafetyNoBFS')
    parser.add_argument('--oneways', action='store_true', required=False, default=False, help='(optional) Whether oneways modelling should be used.')
    parser.add_argument('--bound', type=int, required=False, default=100, help='(optional) Cumulative Reward Bound, defaults to 100')
    parser.add_argument('--threshold', type=float, required=False, default=0, help='The threshold delta_varphi.')
    #parser.add_argument('--docker', action='store_true', required=False, default=False, help='(optional) Whether to use docker instead of local binary.')
    parser.add_argument('--visualize', action='store_true', required=False, default=False, help='(optional) Whether to use visualize 20 steps of the agent.')
    parser.add_argument('--randomMT', action='store_true', required=False, default=False, help='Whether to run EMT instead of IMT')
    parser.add_argument('--random', type=int, required=False, default=0, help='The number of time steps to execute one random test case for.')
    parser.add_argument('--safety', action='store_true', required=False, default=False, help='Compute safety estimates instead of performance.')


    parser.add_argument('--refinement-steps', type=int, required=False, default=5, help='(optional) Amount of refinement steps per iteration, defaults to 5.')
    #parser.add_argument('--bins', type=int, required=False, default=1, help='(optional) how many bins to be used for plotting the heatmaps, defaults to 1.')
    parser.add_argument('--plotting', action='store_true', required=False, default=False, help='(optional) Whether to plot the results for the individual iterations.')
    args = parser.parse_args()


    bound = args.bound
    envname = args.env
    rewardStructure = args.reward
    dest = args.dest

    plotting = args.plotting

    env = gym.make(args.env)
    env = RGBImgObsWrapper(env) # Get pixel observations
    env = ImgObsWrapper(env)    # Get rid of the 'mission' field
    env = MiniWrapper(env)      # Project specific changes
    if args.policy:
        policyname = args.policy

        LOG(f"> Load Model: {policyname}")

        policy = DQN.load(policyname)
        LOG(f"> Simulating from initial state.")
        env.reset()
        if args.visualize:
            simulate_from_state(env, policy)
            #sys.exit(0)

        dest_directory = f'{envname}_{policyname.replace("/", "_").replace(".zip","")}{"_" + dest if dest else ""}'

    i = 0
    prism_file_name = f"{envname}_trimmed_{i:03}"
    env.reset()
    translate_grid_to_prism(env, prism_file_name, args.oneways)

    fixedStates = set()
    states_values = list()

    previousFixedStates = list()
    previousTestedStates = {}
    previousTestedTiles = list()
    previousImpliedTiles = dict()


    #for _ in range(0,1):
    if not args.policy:
        print("No policy provided, plotting heatmap and exiting!")
        call_tempest(prism_file_name, bound, rewardStructure, args.threshold)
        env.fillStateRanking("action_ranking_{}".format(bound))
        env.printHeatMap(prism_file_name + f"_{rewardStructure}", bound, 1)
        #env.printHeatMapReduced(prism_file_name + f"_{rewardStructure}", bound, 1)
        sys.exit(0)

    previous_important_states = list()
    all_test_results = list()
    total_states_to_be_tested = -1
    decided_states = set()
    important_states = list()
    numQueries = 0
    iterationResult = StateSet(i)
    if args.random > 0: #awkward way to test this, but means that a single cli flag is sufficient
        with open(f'output.csv', 'w') as file:
            file.write("num_queries num_failing\n")
            numFailingInstances = 0
            numQueriesTotal = 0
            while numQueriesTotal < 2000: # hardcoded value 1
                pos = env.place_agent()
                dir = env.agent_dir
                testState = State(*pos, dir)
                failed, numQueries = simulate_from_state(env, policy, state=testState, n=args.random, visualize=args.visualize)
                numQueriesTotal += numQueries
                if failed:
                    numFailingInstances += 1
                result = f"{numQueriesTotal} {numFailingInstances}"
                file.write(result + "\n")
                print(result)
        move_files(envname, 0, dest_directory, plotting)
        sys.exit(0)

    while True:
        iterationResult.addTestedStates(important_states)
        decided_states.clear()
        test_result = call_tempest(prism_file_name, bound, rewardStructure, args.threshold, safety=args.safety) # computeEstimates
        if i == 0: iterationResult.totalTestStates = test_result.count_undecided_states

        csv_test_results = test_result.csv()
        states_values_dict = compare_min_max(i) # computeEstimates part 2
        for state, values in states_values_dict.items():
            if values[1] < args.threshold:
                iterationResult.addFailureState(state) # line 8
                decided_states.add(state)
            elif values[0] >= args.threshold:
                iterationResult.addSuccessState(state) # line 12
                decided_states.add(state)
            else:
                iterationResult.addUndecidedState(state) # line 12
        env.fillStateRanking("action_ranking_{}".format(bound), decided_states)
        if plotting: env.printHeatMap(prism_file_name + f"_{rewardStructure}_{i:03}", bound, 1)
        if args.randomMT:
            important_states = env.random_states(args.refinement_steps, decided_states)
        else:
            important_states = env.top_n_states(args.refinement_steps, states_values_dict, args.threshold)
        #print(iterationResult)
        numQueriesForThisIteration = len(important_states)
        numQueries += numQueriesForThisIteration
        state_actions_to_trim = test_all_states(env, policy, important_states, args.visualize)

        all_test_results.append(test_result)
        test_result.updateStateStatistics(iterationResult.stateStatistics(), numQueries)
        print(f"CSV: {test_result.csv()}")
        if abs(test_result.init_check_opt_avg - test_result.init_check_pes_avg) == 0 or test_result.all_states_tested > 0 or len(env.state_ranking) == 0:
            print(abs(test_result.init_check_opt_avg - test_result.init_check_pes_avg) == 0)
            print(test_result.all_states_tested > 0)
            print(len(env.state_ranking) == 0)
            if plotting: env.plotTestedStates(prism_file_name, bound, iterationResult)
            #env.plotTestedStatesMap(prism_file_name, bound, previousTestedTiles, previous_important_states, states_values_dict, previousImpliedTiles, args.threshold)
            if plotting: env.previousFixedStates = plot_heatmaps(env, prism_file_name, bound, rewardStructure, previousFixedStates, state_actions_to_trim, states_values_dict, 1, i)
            print("... Aborting!")
            break

        #previousTestedTiles, previousImpliedTiles, counter = env.plotTestedStatesMap(prism_file_name, bound, previousTestedTiles, previous_important_states, states_values_dict, previousImpliedTiles, args.threshold)

        #test_result.count_proven_failure_states = counter[0]
        #test_result.count_proven_good_states = counter[2]
        #if total_states_to_be_tested < 0:
        #    total_states_to_be_tested = test_result.count_undecided_states
        #else:
        #    test_result.count_undecided_states = total_states_to_be_tested - test_result.count_proven_failure_states - test_result.count_proven_good_states

        i += 1
        previous_file_name = prism_file_name
        prism_file_name = f"{envname}_trimmed_{i:03}"
        update_prism_file(env, previous_file_name, state_actions_to_trim, decided_states, prism_file_name) # restrictMDP
        states_values.append(states_values_dict)

        #print("Start plotting ... ", end=""); sys.stdout.flush()
        if plotting: previousFixedStates = plot_heatmaps(env, prism_file_name, bound, rewardStructure, previousFixedStates, state_actions_to_trim, states_values_dict, 1, i)
        if plotting: env.plotTestedStates(prism_file_name, bound, iterationResult)
        #print("... Done", end="")

        nextIterationResult = StateSet(i)
        nextIterationResult.addFailureStates(iterationResult.failureStates)
        nextIterationResult.addSuccessStates(iterationResult.successStates)
        nextIterationResult.addUndecidedStates(iterationResult.undecidedStates)
        nextIterationResult.addTestedStates(iterationResult.tested)
        nextIterationResult.totalTestStates = iterationResult.totalTestStates
        iterationResult = nextIterationResult
        previous_important_states = important_states
        #input("")

    final_test_results = list()
    #for iteration in range(len(all_test_results) - 1):
    #    test_result = all_test_results[iteration]
    #    test_result.count_undecided_states = all_test_results[iteration+1].count_undecided_states
    #    final_test_results.append(test_result)

    with open(f'output.csv', 'w') as file:
        file.write(TestResult.csv_header(ws=" "))
        file.write("\n")
        print(TestResult.csv_header(ws="\t"))
        for i, test_result in enumerate(all_test_results):
            file.write(f"{i} ")
            file.write(test_result.csv(ws=" "))
            file.write("\n")
            print(i, "\t\t", test_result.csv(ws="\t\t"))
    if plotting: system(f"lualatex -shell-escape boilerplate_tikz.tex --jobname {dest_directory}")

    env.close()
    LOG("> Finished Application!")
    move_files(envname, i, dest_directory, plotting)



# ........................................................................... #

if __name__ == '__main__':
    main()
