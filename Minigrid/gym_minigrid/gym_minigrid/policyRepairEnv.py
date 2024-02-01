from gym_minigrid.minigrid import MiniGridEnv, MissionSpace, HeatMapTile, HeatMapTileReduced, Grid, Wall, Lava, Goal, SlipperyNorth, SlipperyEast, SlipperySouth, SlipperyWest, FixedMapTile, RGBTile
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
import operator
import re
import sys
import numpy as np

from copy import deepcopy


@dataclass(frozen=True)
class State:
    pos_x: int
    pos_y: int
    dir: int

    def __str__(self):
        return f"[{self.pos_x}, {self.pos_y}, dir={self.dir}]"
def default_value():
    return {'action' : None, 'choiceValue' : None}


@dataclass(frozen=True)
class StateValue:
    ranking: float
    choices: dict = field(default_factory=default_value)

def transform(action_list):
    result = []
    for action in action_list:
        if action == 'left' :
            result.append(0)
        elif action == 'right' :
            result.append(1)
        elif action in ['east','west','south','north']:
            result.append(2)
        else:
            result.append(-1)
    return result

def convert(tuples):
    return dict(tuples)
class PolicyRepairEnv(MiniGridEnv):
    def __init__(self, width=9, height=9, agent_start_pos=(1, 1), agent_start_dir=0, **kwargs):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.state_ranking = dict()
        self.reset_state = None

        mission_space = MissionSpace(
            mission_func=lambda: "get to the green goal square"
        )

        super().__init__(
            mission_space=mission_space,
            width=width,
            height=height,
            max_steps=4 * width * height,
            **kwargs
        )




    def fillStateRanking(self, file_name, decided_states, match=""):
        self.state_ranking.clear()
        done_states = list()
        if match:
            skip_line = re.compile(f"(\[|&\s){match}")
        try:
            with open(file_name, "r") as f:
                file_content = f.readlines()
            for line in file_content:
                if match and skip_line.match(line): continue
                stateMapping = convert(re.findall(r"([a-zA-Z]*[a-zA-Z])=(\d+)?", line))
                if "[AgentDone" in line:
                    done_states.append(State(int(stateMapping["xAgent"]), int(stateMapping["yAgent"]), int(stateMapping["viewAgent"])))
                #print("stateMapping", stateMapping)
                choices = convert(re.findall(r"[a-zA-Z_]*(left|right|east|west|north|south|forward|done)[a-zA-Z_]*:(-?\d+\.?\d*)", line))
                #print("choices", choices)
                ranking_value = float(re.search(r"Value:([+-]?(\d*\.\d+)|\d+)", line)[0].replace("Value:",""))
                #print("ranking_value", ranking_value)
                state = State(int(stateMapping["xAgent"]), int(stateMapping["yAgent"]), int(stateMapping["viewAgent"]))
                if state in decided_states: continue
                value = StateValue(ranking_value, choices)
                self.state_ranking[state] = value
            for state in done_states:
                del self.state_ranking[state]
            if len(self.state_ranking) == 0: return
            all_values = [x.ranking for x in self.state_ranking.values()]
            max_value = max(all_values)
            min_value = min(all_values)
            new_state_ranking = {}
            for state, value in self.state_ranking.items():
                choices = value.choices
                try:
                    new_value = (value.ranking - min_value) / (max_value - min_value)
                except ZeroDivisionError as e:
                    new_value = 0.0
                new_state_ranking[state] = StateValue(new_value, choices)
            self.state_ranking = new_state_ranking
        except EnvironmentError:
            print("TODO file not available. Exiting.")
            sys.exit(1)

    def printHeatMap(self, envName, bound, nr_bins=2):
        self.reset()
        heat_map = Grid(self.grid.width, self.grid.height)
        ordered_state_ranking = sorted(self.state_ranking.items(), key=lambda x: x[1].ranking)
        ranking_values = [x[1].ranking for x in ordered_state_ranking]
        for x in range(self.grid.width):
            for y in range(self.grid.height):
                cell = self.grid.get(x,y)
                if cell is not None  and cell.type == "lava":
                    heat_map.set(x, y, Lava())
                elif  cell is not None  and cell.type == "goal":
                    heat_map.set(x, y, Goal())
                elif cell is not None  and cell.type == "wall":
                    heat_map.set(x, y, Wall())
                else:
                    tile = dict()
                    for dir in range(4):
                        try:
                            tile[dir] = (self.state_ranking[State(x,y,dir)]).ranking
                        except:
                            tile[dir] = 0.0
                    heat_map.set(x, y, HeatMapTile(tile, ranking_values, nr_bins))

        img = heat_map.render(self.tile_size, self.agent_pos, self.agent_dir, None, None, False)
        plt.imsave("{}_heatmap_C{:03}_bin_size_{}.png".format(envName,bound, nr_bins), img)

    def plotTestedStates(self, envName, bound, stateSets):
        black = np.array([0,0,0])
        red = np.array([255,0,0])
        green = np.array([0,255,0])
        blue = np.array([51,153,205])
        teal = np.array([102, 255, 255])
        implied_result_colouring_factor = 0.6
        #empty_tile = {0:blue*implied_result_colouring_factor,1:blue*implied_result_colouring_factor,2:blue*implied_result_colouring_factor,3:blue*implied_result_colouring_factor}
        empty_tile = {0:black,1:black,2:black,3:black}

        self.reset()
        tested_states_map = Grid(self.grid.width, self.grid.height)
        tested_tiles_map = Grid(self.grid.width, self.grid.height)

        tested_tiles_map_dict = {}
        for x in range(self.grid.width):
            for y in range(self.grid.height):
                cell = self.grid.get(x,y)
                if cell is not None and cell.type in ["lava", "wall"]: #"slipperynorth", "slipperyeast", "slipperysouth", "slipperywest",
                    if cell.type == "lava": tested_tiles_map.set(x, y, Lava())
                    elif cell.type == "wall": tested_tiles_map.set(x, y, Wall())
                    #elif cell.type == "goal": tested_tiles_map.set(x, y, Goal())
                    elif cell.type == "slipperynorth": tested_tiles_map.set(x, y, SlipperyNorth())
                    elif cell.type == "slipperyeast": tested_tiles_map.set(x, y, SlipperyEast())
                    elif cell.type == "slipperysouth": tested_tiles_map.set(x, y, SlipperySouth())
                    elif cell.type == "slipperywest": tested_tiles_map.set(x, y, SlipperyWest())
                else:
                    tested_tiles_map_dict[(x,y)] = deepcopy(empty_tile)
                    for dir in range(0,4):
                        state = State(x,y,dir)
                        if state in stateSets.undecidedStates and not state in stateSets.tested:
                            tested_tiles_map_dict[(x, y)][dir] = blue
                        if state in stateSets.tested:
                            tested_tiles_map_dict[(x, y)][dir] = teal
                        if state in stateSets.failureStates:
                            tested_tiles_map_dict[(x, y)][dir] = red
                        elif state in stateSets.successStates:
                            tested_tiles_map_dict[(x, y)][dir] = green
                        if not state in stateSets.tested:
                            tested_tiles_map_dict[(x, y)][dir] = tested_tiles_map_dict[(x, y)][dir] * implied_result_colouring_factor

        for key, value in tested_tiles_map_dict.items():
            tested_tiles_map.set(key[0], key[1], FixedMapTile(value))
        img = tested_tiles_map.render(self.tile_size, self.agent_pos, self.agent_dir, None, None, False)
        plt.imsave("{}_tested_tiles_C{:03}.png".format(envName,bound), img)

    def plotTestedStatesMap(self, envName, bound, previousFixedTiles, fixedStates, statesValuesDict, previousImpliedTiles, threshold):
        black = np.array([0,0,0])
        red = np.array([255,0,0])
        green = np.array([0,255,0])
        blue = np.array([0,0,255])
        implied_result_colouring_factor = 0.6
        empty_tile = {0:black,1:black,2:black,3:black}

        self.reset()
        tested_states_map = Grid(self.grid.width, self.grid.height)
        tested_tiles_map = Grid(self.grid.width, self.grid.height)

        counter = [0,0,0]

        tested_tiles_map_dict = {}
        for x in range(self.grid.width):
            for y in range(self.grid.height):
                cell = self.grid.get(x,y)
                if cell is not None and cell.type in ["lava", "goal", "wall"]: #"slipperynorth", "slipperyeast", "slipperysouth", "slipperywest",
                    if cell.type == "lava": tested_tiles_map.set(x, y, Lava())
                    elif cell.type == "wall": tested_tiles_map.set(x, y, Wall())
                    elif cell.type == "goal": tested_tiles_map.set(x, y, Goal())
                    elif cell.type == "slipperynorth": tested_tiles_map.set(x, y, SlipperyNorth())
                    elif cell.type == "slipperyeast": tested_tiles_map.set(x, y, SlipperyEast())
                    elif cell.type == "slipperysouth": tested_tiles_map.set(x, y, SlipperySouth())
                    elif cell.type == "slipperywest": tested_tiles_map.set(x, y, SlipperyWest())
                else:
                    tested_tiles_map_dict[(x,y)] = deepcopy(empty_tile)

        for state, value in previousImpliedTiles.items():
            tested_tiles_map_dict[(state.pos_x, state.pos_y)][state.dir] = value

        for state, performance_estimates in statesValuesDict.items():
            cell = self.grid.get(state.pos_x,state.pos_y)
            if cell is not None and cell.type in ["lava", "goal", "wall"]: continue
            minimum = performance_estimates[0]
            maximum = performance_estimates[1]
            coloring = black
            #print(f"Looking at {state}: min: {minimum}, max: {maximum}, \tgreen: {minimum >= 0 and maximum >= 0}, \tred: {minimum < 0 and maximum < 0}")
            if minimum >= threshold and maximum >= threshold:
                coloring = green * implied_result_colouring_factor;
            elif minimum < threshold and maximum < threshold:
                coloring = red * implied_result_colouring_factor;
            tested_tiles_map_dict[(state.pos_x, state.pos_y)][state.dir] = coloring
            previousImpliedTiles[state] = coloring


        for (state, value) in previousFixedTiles:
            try:
                performance_estimates = statesValuesDict[state]
                minimum = performance_estimates[0]
                maximum = performance_estimates[1]
                if   minimum <  0 and maximum >= 0: new_coloring = blue
                elif minimum >= 0 and maximum >= 0: new_coloring = green
                elif minimum <  0 and maximum <= 0: new_coloring = red
                tested_tiles_map_dict[(state.pos_x, state.pos_y)][state.dir] = new_coloring
            except:
                #print("Filling in from previous step", state, " ", value)
                tested_tiles_map_dict[(state.pos_x, state.pos_y)][state.dir] = value
        for state in fixedStates:
            if isinstance(self.grid.get(state.pos_x,state.pos_y), Goal): continue
            try:
                performance_estimates = statesValuesDict[state]
                minimum = performance_estimates[0]
                maximum = performance_estimates[1]
                if   minimum <  0 and maximum >= 0: new_coloring = blue
                elif minimum >= 0 and maximum >= 0: new_coloring = green
                elif minimum <  0 and maximum <= 0: new_coloring = red
                tested_tiles_map_dict[(state.pos_x, state.pos_y)][state.dir] = new_coloring
                previousFixedTiles.append((state, new_coloring))
            except Exception as e:
                pass
                #print(e)

        for key, value in tested_tiles_map_dict.items():
            tested_tiles_map.set(key[0], key[1], FixedMapTile(value))
            for t in range(0,4):
                color = value[t]
                if np.array_equal(color, green * implied_result_colouring_factor) or np.array_equal(color, green):
                    counter[2] += 1
                elif np.array_equal(color, red * implied_result_colouring_factor) or np.array_equal(color, red):
                    counter[0] += 1
        img = tested_tiles_map.render(self.tile_size, self.agent_pos, self.agent_dir, None, None, False)
        plt.imsave("{}_tested_tiles_C{:03}.png".format(envName,bound), img)
        print(f"This iteration tested {sum(counter)} states: \t{counter[0]} proven system errors, \t{counter[1]} with positive performance estimate and \t{counter[2]} states with proven positive performance. PrevFixedTiles: {len(previousFixedTiles)}, fixedStates: {len(fixedStates)}")
        return previousFixedTiles, previousImpliedTiles, counter

    def printTestedStatesMap(self, envName, bound, previousFixedStates, fixedStates, statesValuesDict, previousTestedStatesDict, previousTestedTilesDict):
        black = np.array([0,0,0])
        red = np.array([255,0,0])
        green = np.array([0,255,0])
        blue = np.array([0,0,255])
        empty_tile = {0:black,1:black,2:black,3:black}

        self.reset()
        tested_states_map = Grid(self.grid.width, self.grid.height)
        tested_tiles_map = Grid(self.grid.width, self.grid.height)

        tested_states_map_dict = previousTestedStatesDict
        tested_tiles_map_dict = previousTestedTilesDict
        for x in range(self.grid.width):
            for y in range(self.grid.height):
                cell = self.grid.get(x,y)
                if cell is not None and cell.type in ["slipperynorth", "slipperyeast", "slipperysouth", "slipperywest", "lava", "goal", "wall"] or (tested_tiles_map_dict[(x,y)] == empty_tile).all():
                    if cell.type == "lava":
                        tested_states_map.set(x, y, Lava())
                        tested_tiles_map.set(x, y, Lava())
                    elif cell.type == "wall":
                        tested_states_map.set(x, y, Wall())
                        tested_tiles_map.set(x, y, Wall())
                    elif cell.type == "goal":
                        tested_states_map.set(x, y, Goal())
                        tested_tiles_map.set(x, y, Goal())
                    elif cell.type == "slipperynorth":
                        tested_states_map.set(x, y, SlipperyNorth())
                        tested_tiles_map.set(x, y, SlipperyNorth())
                    elif cell.type == "slipperyeast":
                        tested_states_map.set(x, y, SlipperyEast())
                        tested_tiles_map.set(x, y, SlipperyEast())
                    elif cell.type == "slipperysouth":
                        tested_states_map.set(x, y, SlipperySouth())
                        tested_tiles_map.set(x, y, SlipperySouth())
                    elif cell.type == "slipperywest":
                        tested_states_map.set(x, y, SlipperyWest())
                        tested_tiles_map.set(x, y, SlipperyWest())
                #else:
                #    tested_states_map_dict[(x,y)] = deepcopy(black)
                #    tested_tiles_map_dict[(x,y)] = deepcopy(empty_tile)

        for entry in fixedStates:
            state = entry[0]
            if isinstance(self.grid.get(state.pos_x,state.pos_y), Goal): continue
            previousFixedStates.append((state, entry[1]))
        for (state, action) in previousFixedStates:
            try:
                #print("FixedState Entry:", statesValuesDict[state], " at ", state)
                new_coloring = black
                performance_estimates = statesValuesDict[state]
                minimum = performance_estimates[0]
                maximum = performance_estimates[1]
                if   minimum <= 0 and maximum >= 0: new_coloring = blue
                elif minimum >= 0 and maximum >= 0: new_coloring = green
                elif minimum <= 0 and maximum <= 0: new_coloring = red
                if (tested_states_map_dict[(state.pos_x, state.pos_y)] == black).all() or (tested_states_map_dict[(state.pos_x, state.pos_y)] == new_coloring).all():
                    tested_states_map_dict[(state.pos_x, state.pos_y)] = new_coloring
                else:
                    print(state, " already has a colouring:" , tested_states_map_dict[(state.pos_x, state.pos_y)], " wanted to colour it with ", new_coloring)

                tested_tiles_map_dict[(state.pos_x, state.pos_y)][state.dir] = new_coloring
            except:
                pass
                #print("need old colour for state:", state)

        for key, value in tested_states_map_dict.items():
            tested_states_map.set(key[0], key[1], RGBTile(color=value)) # Plotting a wall since it can have any colour
        for key, tile in tested_tiles_map_dict.items():
            tested_tiles_map.set(key[0], key[1], FixedMapTile(tile))

        #img = tested_states_map.render(self.tile_size, self.agent_pos, self.agent_dir, None, None, False)
        #plt.imsave("{}_tested_states_map_C{:03}.png".format(envName,bound), img)
        img = tested_tiles_map.render(self.tile_size, self.agent_pos, self.agent_dir, None, None, False)
        plt.imsave("{}_tested_tiles_map_C{:03}.png".format(envName,bound), img)
        return tested_states_map_dict, tested_tiles_map_dict


    def printFixedMap(self, envName, bound, previousFixedStates, fixedStates):
        prevFactor = 0.8
        black = np.array([0,0,0])
        forward = np.array([180, 142, 173])
        left = np.array([163, 190, 140])
        right = np.array([136, 192, 208])
        colourMapping = {"new": {0: left, 1: right, 2:  forward},
                         "previous": {0: left * prevFactor, 1: right * prevFactor, 2: forward * prevFactor} }

        self.reset()
        fixed_map = Grid(self.grid.width, self.grid.height)

        new_fixed_map_dict = {}
        empty_tile = {0:black,1:black,2:black,3:black}
        for x in range(self.grid.width):
            for y in range(self.grid.height):
                cell = self.grid.get(x,y)
                if cell is not None and cell.type in ["lava", "goal", "wall"]: #"slipperynorth", "slipperyeast", "slipperysouth", "slipperywest",
                    if cell.type == "lava": fixed_map.set(x, y, Lava())
                    elif cell.type == "wall": fixed_map.set(x, y, Wall())
                    elif cell.type == "goal": fixed_map.set(x, y, Goal())
                    elif cell.type == "slipperynorth": fixed_map.set(x, y, SlipperyNorth())
                    elif cell.type == "slipperyeast": fixed_map.set(x, y, SlipperyEast())
                    elif cell.type == "slipperysouth": fixed_map.set(x, y, SlipperySouth())
                    elif cell.type == "slipperywest": fixed_map.set(x, y, SlipperyWest())
                    elif cell.type == "onewaynorth": fixed_map.set(x, y, OneWayNorth())
                    elif cell.type == "onewayeast": fixed_map.set(x, y, OneWayEast())
                    elif cell.type == "onewaysouth": fixed_map.set(x, y, OneWaySouth())
                    elif cell.type == "onewaywest": fixed_map.set(x, y, OneWayWest())
                else:
                    new_fixed_map_dict[(x,y)] = deepcopy(empty_tile)

        for (state, action) in previousFixedStates:
            #print(f"Fixing old action at {state} -> {action}")
            try:
                new_fixed_map_dict[(state.pos_x, state.pos_y)][state.dir] = colourMapping["previous"].get(int(action), np.array([40,40,40]))
            except:
                pass
                #print(state)
        for entry in fixedStates:
            state = entry[0]
            if isinstance(self.grid.get(state.pos_x,state.pos_y), Goal): continue
            #print(f"Fixing new action at {state} -> {entry[1]}")
            try:
                new_fixed_map_dict[(state.pos_x, state.pos_y)][state.dir] = colourMapping["new"].get(int(entry[1]), np.array([70,70,70]))
            except:
                print(entry)
            previousFixedStates.append((state, entry[1]))
        for key, value in new_fixed_map_dict.items():
            fixed_map.set(key[0], key[1], FixedMapTile(value))

        img = fixed_map.render(self.tile_size, self.agent_pos, self.agent_dir, None, None, False)
        plt.imsave("{}_fixed_map_C{:03}.png".format(envName,bound), img)
        return previousFixedStates

    def printHeatMapReduced(self, envName, bound, nr_bins=2):
        self.reset()
        heat_map = Grid(self.grid.width, self.grid.height)
        ordered_state_ranking = sorted(self.state_ranking.items(), key=lambda x: x[1].ranking)
        ranking_values = [x[1].ranking for x in ordered_state_ranking]
        for x in range(self.grid.width):
            for y in range(self.grid.height):
                cell = self.grid.get(x,y)
                if cell is not None  and cell.type == "lava":
                    heat_map.set(x, y, Lava())
                elif  cell is not None  and cell.type == "goal":
                    heat_map.set(x, y, Goal())
                elif cell is not None  and cell.type == "wall":
                    heat_map.set(x, y, Wall())
                else:
                    ranking_max = 0
                    for dir in range(4):
                        try:
                            ranking = self.state_ranking[State(x,y,dir)].ranking
                            if ranking > ranking_max:
                                ranking_max = ranking
                        except:
                            pass
                    heat_map.set(x, y, HeatMapTileReduced(ranking_max, ranking_values, nr_bins))

        img = heat_map.render(self.tile_size, self.agent_pos, self.agent_dir, None, None, False)
        plt.imsave("{}_heatmap_reduced_C{:03}_bin_size_{}.png".format(envName,bound, nr_bins), img)

    def reset(self, seed=None, state=None):
        return super().reset(state=state, seed=seed)

    def top_n_states(self, n, states_values_dict, threshold):
        untested_states = {state: state_value for state, state_value in self.state_ranking.items() if states_values_dict[state][1] >= threshold and states_values_dict[state][0] < threshold}
        ordered_state_ranking = sorted(untested_states.items(), key=lambda x: (x[1].ranking, len(x[1].choices)))
        states = [x[0] for x in ordered_state_ranking if x[1].ranking > 0.3]
        #states = [x[0] for x in ordered_state_ranking if len(x[1].choices) > 1]
        return states[-n:]

    def random_states(self, n, decided_states):
        # only test state of it actually has a choice
        states = [key for key, state_value in self.state_ranking.items() if len(state_value.choices) > 1]
        states = [x for x in states if x not in decided_states]
        try:
            return np.random.choice(states, size=n, replace=False)
        except:
            return states


    def check_if_best_action(self, action, state):
        print(f"\nChecking {state} with \n\tranking/choices: {self.state_ranking[state]}")
        max_value = max(self.state_ranking[state].choices.values())
        best_actions = [k for k,v in self.state_ranking[state].choices.items() if v == max_value]
        best_action_enum = transform(best_actions)
        if action in best_action_enum:
            print(f"\t{action}, best action was taken!\t".center(80, '✓'))
        else:
            print(f"\t{action }, non-optimal action!\t".center(80, '⛌'))

    def print_ranking_value(self, bound, x, y, dir):
        state = State(x,y,dir)
        print("Bound: {}, State: {}, {}".format(bound, state, self.state_ranking[state].ranking))

    def print_readable_ranking(self, bound):
        for y in range(1, self.grid.height-1):
            print("--------------\n")
            row1 = "|"
            row2 = "|"
            row3 = "|"
            for x in range(1, self.grid.width-1):
                row1 += "       {:.9f}      |".format(self.state_ranking[State(x,y,3)].ranking)
                row2 += "{:.9f}  {:.9f}|".format(self.state_ranking[State(x,y,2)].ranking, self.state_ranking[State(x,y,0)].ranking)
                row3 += "       {:.9f}      |".format(self.state_ranking[State(x,y,1)].ranking)
            print(row1)
            print(row2)
            print(row3)
