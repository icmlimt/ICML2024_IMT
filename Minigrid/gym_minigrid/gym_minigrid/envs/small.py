import numpy as np
from gym import spaces
from gym_minigrid.minigrid import Goal, Grid, MiniGridEnv, MissionSpace, Floor, Adversary, Box, SlipperyNorth, run_BFS_reward
from gym_minigrid.Task import DoRandom, TaskManager, DoNothing, GoTo, PlaceObject, PickUpObject, Task

from copy import deepcopy
import sys

deliveryStation1Tile = Floor("blue")
deliveryStation2Tile = Floor("green")
deliveryStation3Tile = Floor("purple")
deliveryStationRLTile = Floor("red")
deliveryRegion1Tile = Floor("lightblue")
deliveryRegion2Tile = Floor("lightgreen")
deliveryRegion3Tile = Floor("lightpurple")

class DeliveryStationsSmallNoAg(MiniGridEnv):
    def __init__(self, agent_start_pos=(1, 1), agent_start_dir=0, highlight:bool=False,
                 reward_type:str='Dense', number_adversaries:int=3, max_steps=200,
                 shield_gamma=1.0, shield_horizon=30, collision_penalty=0.1, **kwargs):
        assert reward_type == 'Sparse' or reward_type == 'Dense', "Must select either 'Sparse' or 'Dense' reward"
        assert 0 <= number_adversaries <= 3, "Must select either 0, 1, 2 or 3 adversaries"

        self.collision_penalty = collision_penalty
        self.shield_gamma = shield_gamma
        self.shield_horizon = shield_horizon
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.number_adversaries = number_adversaries
        self.reward_type = reward_type


        mission_space = MissionSpace(
            mission_func=lambda: "Avoid crashing into the other agents."
        )

        super().__init__(
            mission_space=mission_space,
            width=13,
            height=13,
            # Set this to True for maximum speed
            highlight=highlight,
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs
        )


    def _gen_grid(self, width, height):
        self.use_purple_adversary = self.number_adversaries >= 3
        self.use_blue_adversary = self.number_adversaries >= 2
        self.use_green_adversary = self.number_adversaries >= 1

        # Create an empty grid
        self.grid = Grid(width, height)

        # add walls
        self.grid.wall_rect(0, 0, width, height) # outside border
        self.grid.horz_wall(2, 2, width - 4) # top wall
        self.grid.horz_wall(2, height - 3, width - 4) # bottom wall
        #self.grid.horz_wall(2, int(height/2), width - 4) # middle wall

        # find where to spawn boxes
        self.potential_box_locations_red = []
        self.potential_box_locations_green = []
        self.potential_box_locations_blue = []
        self.potential_box_locations_purple = []
        for i in range(1,width-1):
            for j in range(1,4):
                if j != 2 or (i == 1 or i == width-2):
                    self.grid.set_background(i,j,deliveryRegion1Tile)
            for j in range(height - 4, height-1):
                if j != height-3 or (i == 1 or i == width-2):
                    self.grid.set_background(i,j,deliveryRegion2Tile)

            #if i != 1 and i != width-2:
            #    if i == 2 or i == 3 or i == 4:
            #        self.potential_box_locations_red.append((i, int(height/2)+1))
            #        self.potential_box_locations_red.append((i, int(height/2)-1))

        self.potential_box_locations_red.append((3, int(height/2)+1))
        self.potential_box_locations_red.append((3, int(height/2)-1))
        self.potential_box_locations_red.append((5, int(height/2)-1))
        self.potential_box_locations_red.append((5, int(height/2)+1))
        self.potential_box_locations_red.append((4, int(height/2)-1))
        self.potential_box_locations_red.append((4, int(height/2)+1))

        self.potential_box_locations_blue.append((2, int(height/2)-2))
        self.potential_box_locations_blue.append((2, int(height/2)-1))
        self.potential_box_locations_blue.append((2, int(height/2)+1))
        self.potential_box_locations_blue.append((2, int(height/2)+2))
        self.potential_box_locations_blue.append((4, int(height/2)-2))
        self.potential_box_locations_blue.append((4, int(height/2)-1))
        self.potential_box_locations_blue.append((4, int(height/2)+1))
        self.potential_box_locations_blue.append((4, int(height/2)+2))
        self.potential_box_locations_blue.append((6, int(height/2)-2))
        self.potential_box_locations_blue.append((6, int(height/2)-1))
        self.potential_box_locations_blue.append((6, int(height/2)+1))
        self.potential_box_locations_blue.append((6, int(height/2)+2))
        self.potential_box_locations_blue.append((8, int(height/2)-2))
        self.potential_box_locations_blue.append((8, int(height/2)-1))
        self.potential_box_locations_blue.append((8, int(height/2)+1))
        self.potential_box_locations_blue.append((8, int(height/2)+2))

        self.potential_box_locations_green.append((2, int(height/2)-2))
        self.potential_box_locations_green.append((2, int(height/2)-1))
        self.potential_box_locations_green.append((2, int(height/2)+1))
        self.potential_box_locations_green.append((2, int(height/2)+2))
        self.potential_box_locations_green.append((4, int(height/2)-2))
        self.potential_box_locations_green.append((4, int(height/2)-1))
        self.potential_box_locations_green.append((4, int(height/2)+1))
        self.potential_box_locations_green.append((4, int(height/2)+2))
        #self.potential_box_locations_green.append((6, int(height/2)-2))
        #self.potential_box_locations_green.append((6, int(height/2)-1))
        #self.potential_box_locations_green.append((6, int(height/2)+1))
        #self.potential_box_locations_green.append((6, int(height/2)+2))
        #self.potential_box_locations_green.append((8, int(height/2)-2))
        #self.potential_box_locations_green.append((8, int(height/2)-1))
        #self.potential_box_locations_green.append((8, int(height/2)+1))
        #self.potential_box_locations_green.append((8, int(height/2)+2))

        #self.potential_box_locations_purple.append((2, int(height/2)-2))
        #self.potential_box_locations_purple.append((2, int(height/2)-1))
        #self.potential_box_locations_purple.append((2, int(height/2)+1))
        #self.potential_box_locations_purple.append((2, int(height/2)+2))
        #self.potential_box_locations_purple.append((4, int(height/2)-2))
        #self.potential_box_locations_purple.append((4, int(height/2)-1))
        #self.potential_box_locations_purple.append((4, int(height/2)+1))
        #self.potential_box_locations_purple.append((4, int(height/2)+2))
        self.potential_box_locations_purple.append((6, int(height/2)-2))
        self.potential_box_locations_purple.append((6, int(height/2)-1))
        self.potential_box_locations_purple.append((6, int(height/2)+1))
        self.potential_box_locations_purple.append((6, int(height/2)+2))
        self.potential_box_locations_purple.append((8, int(height/2)-2))
        self.potential_box_locations_purple.append((8, int(height/2)-1))
        self.potential_box_locations_purple.append((8, int(height/2)+1))
        self.potential_box_locations_purple.append((8, int(height/2)+2))
        # get all possible spawn locations
        self.potential_spawn_locations_blue = []
        self.potential_spawn_locations_green = []
        self.potential_spawn_locations_purple = []
        for i in range(1,width-1):
            for j in range(1, height-1):
                # add a buffer around the agent spawn location
                if abs(i - self.agent_start_pos[0]) <= 1 or abs(j - self.agent_start_pos[1]) <= 1:
                    continue
                obj = self.grid.get(i,j)
                bg = self.grid.get_background(i, j)
                if obj is None:
                    if bg is None or bg.color == 'lightblue':
                        self.potential_spawn_locations_blue.append((i,j))
                    if bg is None or bg.color == 'lightgreen':
                        self.potential_spawn_locations_green.append((i,j))
                        self.potential_spawn_locations_purple.append((i,j))


        self.red_delivery_station = (2,1)
        self.grid.set_background(*self.red_delivery_station,deliveryStationRLTile)

        self.blue_delivery_station = (8,1)
        self.grid.set_background(*self.blue_delivery_station,deliveryStation1Tile)

        self.green_delivery_station = (2,height - 2)
        self.grid.set_background(*self.green_delivery_station,deliveryStation2Tile)

        self.purple_delivery_station = (8,height - 2)
        self.grid.set_background(*self.purple_delivery_station,deliveryStation3Tile)

        self.adversaries = {}
        if self.use_blue_adversary:
            random_start = self.get_viable_location(self.potential_spawn_locations_blue)
            self.add_adversary(*random_start, "blue", np.random.randint(low=0, high=4), [DoNothing()])
            self.add_blue_box()


        if self.use_green_adversary:
            random_start = self.get_viable_location(self.potential_spawn_locations_green)
            self.add_adversary(*random_start, "green", np.random.randint(low=0, high=4), [DoNothing()])
            self.add_green_box()

        if self.use_purple_adversary:
            random_start = self.get_viable_location(self.potential_spawn_locations_purple)
            self.add_adversary(*random_start, "purple", np.random.randint(low=0, high=4), [DoNothing()])
            self.add_purple_box()

        # learning agent boxes
        red_random_box_location1 = self.get_viable_location(self.potential_box_locations_red)
        self.grid.set(*red_random_box_location1,Box("red"))
        if self.reward_type == 'Dense':
            self.learning_agent_manager = TaskManager([
                PickUpObject(red_random_box_location1, self.grid.get(*red_random_box_location1)),
                PlaceObject((2, 1), self.grid.get(*red_random_box_location1)),
                DoNothing()
            ])
            current_objective = self.learning_agent_manager.tasks[0].obj_position
            self.bfs_reward = run_BFS_reward(self.grid, current_objective)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "Avoid crashing into the other agents."

    def add_blue_box(self):
        random_box_location1 = self.get_viable_location(self.potential_box_locations_blue)
        self.grid.set(*random_box_location1, Box("blue"))
        self.adversaries["blue"].task_manager.tasks.insert(0,PickUpObject(random_box_location1, self.grid.get(*random_box_location1)))
        self.adversaries["blue"].task_manager.tasks.insert(1, PlaceObject(self.blue_delivery_station, self.grid.get(*random_box_location1)))

    def add_green_box(self):
        random_box_location1 = self.get_viable_location(self.potential_box_locations_green)
        self.grid.set(*random_box_location1, Box("green"))
        self.adversaries["green"].task_manager.tasks.insert(0,PickUpObject(random_box_location1, self.grid.get(*random_box_location1)))
        self.adversaries["green"].task_manager.tasks.insert(1, PlaceObject(self.green_delivery_station, self.grid.get(*random_box_location1)))

    def add_purple_box(self):
        random_box_location1 = self.get_viable_location(self.potential_box_locations_purple)
        self.grid.set(*random_box_location1, Box("purple"))
        self.adversaries["purple"].task_manager.tasks.insert(0,PickUpObject(random_box_location1, self.grid.get(*random_box_location1)))
        self.adversaries["purple"].task_manager.tasks.insert(1, PlaceObject(self.purple_delivery_station, self.grid.get(*random_box_location1)))

    def add_red_box(self):
        #sys.exit("adding additional red boxes, even though this should terminate the episode. \n Remove if multiple boxes should be picked up")
        print("WARN: Adding red box after {} steps.".format(self.step_count))
        random_box_location1 = self.get_viable_location(self.potential_box_locations_red)
        self.grid.set(*random_box_location1, Box("red"))
        if self.reward_type == 'Dense':
            self.learning_agent_manager.tasks.insert(0, PickUpObject(random_box_location1, self.grid.get(*random_box_location1)))
            self.learning_agent_manager.tasks.insert(1, PlaceObject(self.red_delivery_station, self.grid.get(*random_box_location1)))


    def step(self, action):
        delete_list = list()
        for position, box in self.background_boxes.items():
            if self.grid.get(*position) is None:
                self.grid.set(*position, box)
                self.grid.set_background(*position, None)
                delete_list.append(tuple(position))
            #else:
            #    print("Cannot remove box at {}, {} from backgroud".format(*position))
        for position in delete_list:
            #print("Removed box at {}, {} from backgroud".format(*position))
            del self.background_boxes[position]
        self.reward = 0.0

        obs, reward, terminated, truncated, info = super().step(action)
        self.violation_deque.appendleft("Agent Movement:\n{}".format(self.printGrid()))

        # need to augment this just like the agent, and we can hardcode a policy to follow
        for adversary in self.adversaries.values():
            blocked_positions = [adv.cur_pos for adv in self.adversaries.values()]
            agent_pos = self.agent_pos
            advsersary_action = self.get_adversary_action(adversary)
            self.move_adversary(adversary, advsersary_action, blocked_positions, agent_pos)

        if self.reward_type == 'Dense':
            #starting_pos = self.agent_pos
            #best_action = self.learning_agent_manager.get_best_action(self.agent_pos, self.dir_vec, self.carrying, self)
            #if action == best_action:
            #    self.reward += 0.1
            #else:
            #    self.reward -= 0.1
            self.reward += self.bfs_reward[self.agent_pos[0] + self.grid.width * self.agent_pos[1]]

        # get obs from env
        #obs, reward, terminated, truncated, info = super().step(action)

        # reward function and done function are custom
        reward = self.reward
        #print("Task[0]: {}".format(self.learning_agent_manager.tasks[0]))

        if self.reward_type == 'Dense':
            # current_pos = self.agent_pos
            # goal_pos = self.learning_agent_manager.tasks[0].obj_position
            # starting_distance = abs(starting_pos[0] - goal_pos[0]) + abs(starting_pos[1] - goal_pos[1])
            # ending_distance = abs(current_pos[0] - goal_pos[0]) + abs(current_pos[1] - goal_pos[1])
            # reward += 0.1 * (starting_distance - ending_distance)
            # if self.learning_agent_manager.tasks[0].completed(self.agent_pos, self.dir_vec, self.carrying, self):
            #     self.learning_agent_manager.tasks.pop(0)

            if self.learning_agent_manager.tasks[0].completed(self.agent_pos, self.dir_vec, self.carrying, self):
                self.learning_agent_manager.tasks.pop(0)
                current_objective = self.learning_agent_manager.tasks[0]
                if hasattr(current_objective, 'obj_position'):
                    self.bfs_reward = run_BFS_reward(self.grid, current_objective.obj_position)

        # respawn boxes as needed
        maybe_red_box = self.grid.get(*self.red_delivery_station)
        if maybe_red_box and isinstance(maybe_red_box, Box) and maybe_red_box.color == 'red':
            self.grid.set(*self.red_delivery_station, None)
            self.add_red_box()

        # respawn green box
        if self.use_green_adversary:
            maybe_green_box = self.grid.get(*self.green_delivery_station)
            if maybe_green_box and isinstance(maybe_green_box, Box) and maybe_green_box.color == 'green':
                self.grid.set(*self.green_delivery_station, None)
                self.add_green_box()

        # respawn blue box
        if self.use_blue_adversary:
            maybe_blue_box = self.grid.get(*self.blue_delivery_station)
            if maybe_blue_box and isinstance(maybe_blue_box, Box) and maybe_blue_box.color == 'blue':
                self.grid.set(*self.blue_delivery_station, None)
                self.add_blue_box()

        # respawn purple box
        if self.use_purple_adversary:
            maybe_purple_box = self.grid.get(*self.purple_delivery_station)
            if maybe_purple_box and isinstance(maybe_purple_box, Box) and maybe_purple_box.color == 'purple':
                self.grid.set(*self.purple_delivery_station, None)
                self.add_purple_box()

        #if terminated:
        #    print("Terminated true @ {}".format(self.step_count))
        return obs, reward, terminated, truncated, info

    def seed(self, seed):
        np.random.seed(seed)
        return seed

    def get_adversary_action(self, adversary):
        return adversary.task_manager.get_best_action(adversary.cur_pos, adversary.dir_vec(), adversary.carrying, self)

    def get_viable_location(self, locs):
        tried = list()
        locations = deepcopy(locs)
        viable = False
        counter = 0
        while not viable:
            if len(locations) == 0:
                print("All locations: ", locs)
                print("Tried: ", tried)
                print(self.printGrid())
                sys.exit("No viable locations.")
            location = locations[np.random.randint(low=0, high=len(locations))]
            tried.append(location)
            if self.grid.get(*location) is None and self.agent_pos != location:
                viable = True
            else:
                locations.remove(location)

            counter += 1
            if counter >= 10:
                print(location)
                print(locations)
                print(self.printGrid())
                input("")
        return location

    # Moves the adversary according to current policy, code copy pasted from minigrid.step
    def move_adversary(self, adversary, action, blocked_positions, agent_pos):
        # fetch current location and forward location
        cur_pos = adversary.cur_pos
        current_cell = self.grid.get(*adversary.cur_pos)
        fwd_pos = cur_pos + adversary.dir_vec()
        fwd_cell = self.grid.get(*fwd_pos)


        if action == self.actions.left:
            adversary.adversary_dir -= 1
            if adversary.adversary_dir < 0:
                adversary.adversary_dir += 4

        # Rotate right
        elif action == self.actions.right:
            adversary.adversary_dir = (adversary.adversary_dir + 1) % 4

        # Move forward
        elif action == self.actions.forward and isinstance(current_cell, SlipperyNorth):
            print("TODO: actual slippery obs, for now: Teleporting the agent")
            # TODO also what if blocked, see below
            adversary.cur_pos = (3, 3)

        elif action == self.actions.forward and not isinstance(current_cell, SlipperyNorth):
            if tuple(fwd_pos) == agent_pos:
                self.reward -= self.collision_penalty
                self.safety_violations_this_episode += 1
                #print("Safety Violation @ {},{}, by adversary".format(*fwd_pos))
                #for g in self.violation_deque:
                #    print(g)
                #input("Hit Enter.")
                self.violation_deque.clear()


            elif (fwd_cell is None or fwd_cell.can_overlap()) and not tuple(fwd_pos) in blocked_positions:
                if isinstance(fwd_cell, Box):
                    self.grid.set_background(*fwd_pos,fwd_cell)
                    self.background_boxes[tuple(fwd_pos)] = fwd_cell  # np.array is not hashable
                adversary.cur_pos = tuple(fwd_pos)

        # Pick up an object
        elif action == self.actions.pickup:
            if fwd_cell and fwd_cell.can_pickup():
                if adversary.carrying is None:
                    adversary.carrying = fwd_cell
                    adversary.carrying.cur_pos = np.array([-1, -1])
                    self.grid.set(fwd_pos[0], fwd_pos[1], None)

        # Drop an object
        elif action == self.actions.drop:
            if not fwd_cell and adversary.carrying:
                self.grid.set(fwd_pos[0], fwd_pos[1], adversary.carrying)
                adversary.carrying.cur_pos = fwd_pos
                adversary.carrying = None

        # Toggle/activate an object
        elif action == self.actions.toggle:
            if fwd_cell:
                fwd_cell.toggle(self, fwd_pos)

        # Done action (not used by default)
        elif action == self.actions.done:
            pass

        else:
            raise ValueError(f"Unknown action: {action}")

        # finally update the env with these changes

        self.grid.set(*cur_pos, None)
        self.grid.set(*adversary.cur_pos, adversary)
        self.violation_deque.appendleft("{} Movement:\n{}".format(adversary.color, self.printGrid()))

    def get_adversaries(self):
        return [adversary.color for adversary in self.adversaries.values()]

    def get_property_map(self, epoch_string):
        return { "blue" : ("SafetyBlue", '<' + epoch_string + 'SafetyBlue, PreSafety, gamma={}> <<Agent>> Pmax=? [G<{} !"crash" ];'.format(self.shield_gamma, self.shield_horizon)),
                 "green": ("SafetyGreen", '<' + epoch_string + 'SafetyGreen, PreSafety, gamma={}> <<Agent>> Pmax=? [G<{} !"crash" ];'.format(self.shield_gamma, self.shield_horizon)),
                 "purple": ("SafetyPurple", '<' + epoch_string + 'SafetyPurple, PreSafety, gamma={}> <<Agent>> Pmax=? [G<{} !"crash" ];'.format(self.shield_gamma, self.shield_horizon))
                 }

    def get_shield_info(self):
        adversary_info = {color : ((adv.cur_pos[1], adv.cur_pos[0]), adv.adversary_dir) for (color,adv) in self.adversaries.items()}
        return self.agent_pos, self.agent_dir, adversary_info
