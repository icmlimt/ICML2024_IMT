import numpy as np
from gym import spaces
from gym_minigrid.minigrid import Goal, Grid, MiniGridEnv, MissionSpace, Floor, Adversary, Box, SlipperyNorth
from gym_minigrid.Task import DoRandom, TaskManager, DoNothing, GoTo, PlaceObject, PickUpObject, Task

deliveryStation1Tile = Floor("blue")
deliveryStation2Tile = Floor("green")
deliveryStationRLTile = Floor("red")
deliveryRegion1Tile = Floor("lightblue")
deliveryRegion2Tile = Floor("lightgreen")

class DeliveryStations(MiniGridEnv):

    """
    ### Description

    This environment is an empty room, and the goal of the agent is to reach the
    green goal square, which provides a sparse reward. A small penalty is
    subtracted for the number of steps to reach the goal. This environment is
    useful, with small rooms, to validate that your RL algorithm works
    correctly, and with large rooms to experiment with sparse rewards and
    exploration. The random variants of the environment have the agent starting
    at a random position for each episode, while the regular variants have the
    agent always starting in the corner opposite to the goal.

    ### Mission Space

    "get to the green goal square"

    ### Action Space

    | Num | Name         | Action       |
    |-----|--------------|--------------|
    | 0   | left         | Turn left    |
    | 1   | right        | Turn right   |
    | 2   | forward      | Move forward |
    | 3   | pickup       | Unused       |
    | 4   | drop         | Unused       |
    | 5   | toggle       | Unused       |
    | 6   | done         | Unused       |

    ### Observation Encoding

    - Each tile is encoded as a 3 dimensional tuple:
        `(OBJECT_IDX, COLOR_IDX, STATE)`
    - `OBJECT_TO_IDX` and `COLOR_TO_IDX` mapping can be found in
        [gym_minigrid/minigrid.py](gym_minigrid/minigrid.py)
    - `STATE` refers to the door state with 0=open, 1=closed and 2=locked

    ### Rewards

    A reward of '1' is given for success, and '0' for failure.

    ### Termination

    The episode ends if any one of the following conditions is met:

    1. The agent reaches the goal.
    2. Timeout (see `max_steps`).

    ### Registered Configurations

    - `MiniGrid-Empty-5x5-v0`
    - `MiniGrid-Empty-Random-5x5-v0`
    - `MiniGrid-Empty-6x6-v0`
    - `MiniGrid-Empty-Random-6x6-v0`
    - `MiniGrid-Empty-8x8-v0`
    - `MiniGrid-Empty-16x16-v0`

    """

    def __init__(self, agent_start_pos=(1, 1), agent_start_dir=0, highlight:bool=False,
                 reward_type:str='Sparse', number_adversaries:int=2, max_steps=200,
                 shield_gamma=1.0, shield_horizon=30, collision_penalty=0.1, **kwargs):
        assert reward_type == 'Sparse' or reward_type == 'Dense', "Must select either 'Sparse' or 'Dense' reward"
        assert 0 <= number_adversaries <= 2, "Must select either 0, 1, or 2 adversaries"

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
        self.use_green_adversary = self.number_adversaries == 2
        self.use_blue_adversary = self.number_adversaries >= 1

        # Create an empty grid
        self.grid = Grid(width, height)

        # add walls
        self.grid.wall_rect(0, 0, width, height) # outside border
        self.grid.horz_wall(2, 2, width - 4) # top wall
        self.grid.horz_wall(2, height - 3, width - 4) # bottom wall
        self.grid.horz_wall(2, int(height/2), width - 4) # middle wall

        # find where to spawn boxes
        self.potential_box_locations_red = []
        self.potential_box_locations_green = []
        self.potential_box_locations_blue = []
        for i in range(1,width-1):
            for j in range(1,4):
                if j != 2 or (i == 1 or i == width-2):
                    self.grid.set_background(i,j,deliveryRegion1Tile)
            for j in range(height - 4, height-1):
                if j != height-3 or (i == 1 or i == width-2):
                    self.grid.set_background(i,j,deliveryRegion2Tile)

            if i != 1 and i != width-2:
                if i == 2 or i == 3 or i == 4:
                    self.potential_box_locations_red.append((i, int(height/2)+1))
                    self.potential_box_locations_red.append((i, int(height/2)-1))
                self.potential_box_locations_green.append((i, int(height/2)-1))
                self.potential_box_locations_blue.append((i, int(height/2)+1))

        # get all possible spawn locations
        self.potential_spawn_locations_blue = []
        self.potential_spawn_locations_green = []
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



        self.red_delivery_station = (2,1)
        self.grid.set_background(*self.red_delivery_station,deliveryStationRLTile)

        self.blue_delivery_station = (8,1)
        self.grid.set_background(*self.blue_delivery_station,deliveryStation1Tile)

        self.green_delivery_station = (6,height - 2)
        self.grid.set_background(*self.green_delivery_station,deliveryStation2Tile)

        self.adversaries = {}
        if self.use_blue_adversary:
            random_start = self.get_viable_location(self.potential_spawn_locations_blue)
            self.add_adversary(*random_start, "blue", np.random.randint(low=0, high=4), [DoNothing()])
            self.add_blue_box()


        if self.use_green_adversary:
            random_start = self.get_viable_location(self.potential_spawn_locations_green)
            self.add_adversary(*random_start, "green", np.random.randint(low=0, high=4), [DoNothing()])
            self.add_green_box()

        # learning agent boxes
        red_random_box_location1 = self.get_viable_location(self.potential_box_locations_red)
        self.grid.set(*red_random_box_location1,Box("red"))
        if self.reward_type == 'Dense':
            self.learning_agent_manager = TaskManager([
                PickUpObject(red_random_box_location1, self.grid.get(*red_random_box_location1)),
                PlaceObject(self.red_delivery_station, self.grid.get(*red_random_box_location1)),
                DoNothing()
            ])

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
    def add_red_box(self):
        random_box_location1 = self.get_viable_location(self.potential_box_locations_red)
        self.grid.set(*random_box_location1, Box("red"))
        if self.reward_type == 'Dense':
            self.learning_agent_manager.tasks.insert(0,PickUpObject(random_box_location1, self.grid.get(*random_box_location1)))
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

        # need to augment this just like the agent, and we can hardcode a policy to follow
        for adversary in self.adversaries.values():
            blocked_positions = [adv.cur_pos for adv in self.adversaries.values()]
            agent_pos = self.agent_pos
            advsersary_action = self.get_adversary_action(adversary)
            self.move_adversary(adversary, advsersary_action, blocked_positions, agent_pos)

        if self.reward_type == 'Dense':
            # starting_pos = self.agent_pos
            best_action = self.learning_agent_manager.get_best_action(self.agent_pos, self.dir_vec, self.carrying, self)
            if action == best_action:
                self.reward += 0.1
            else:
                self.reward -= 0.1


        # reward function and done function are custom
        reward = self.reward

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


        return obs, reward, terminated, truncated, info

    def seed(self, seed):
        np.random.seed(seed)
        return seed

    def get_adversary_action(self, adversary):
        return adversary.task_manager.get_best_action(adversary.cur_pos, adversary.dir_vec(), adversary.carrying, self)

    def get_viable_location(self, locations):
        viable = False
        while not viable:
            location = locations[np.random.randint(low=0, high=len(locations))]
            if self.grid.get(*location) is None and self.agent_pos != location:
                viable = True
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


            elif (fwd_cell is None or fwd_cell.can_overlap()) and not tuple(fwd_pos) in blocked_positions:
                if isinstance(fwd_cell, Box):
                    print("set background: ", fwd_cell)
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

    def get_adversaries(self):
        return [adversary.color for adversary in self.adversaries.values()]

    def get_property_map(self):
        return { "blue" : ("SafetyAGBlue", 'concat(<_, PreSafety, gamma={}> <<Blue>> Pmax=? [ G !"BlueOnGreen"  ], <SafetyAGBlue, PreSafety, gamma={}> <<Agent>> Pmax=? [G<{} !"crash" ] );'.format(self.shield_gamma, self.shield_gamma, self.shield_horizon)),
                 "green": ("SafetyAGGreen", 'concat(<_, PreSafety, gamma={}> <<Green>> Pmax=? [ G !"GreenOnBlue"  ], <SafetyAGGreen, PreSafety, gamma={}> <<Agent>> Pmax=? [G<{} !"crash" ] );'.format(self.shield_gamma, self.shield_gamma, self.shield_horizon))
                 }

    def get_shield_info(self):
        adversary_info = {color : ((adv.cur_pos[1], adv.cur_pos[0]), adv.adversary_dir) for (color,adv) in self.adversaries.items()}
        return self.agent_pos, self.agent_dir, adversary_info



class DeliveryStationsWithRows(MiniGridEnv):

    """
    ### Description

    This environment is an empty room, and the goal of the agent is to reach the
    green goal square, which provides a sparse reward. A small penalty is
    subtracted for the number of steps to reach the goal. This environment is
    useful, with small rooms, to validate that your RL algorithm works
    correctly, and with large rooms to experiment with sparse rewards and
    exploration. The random variants of the environment have the agent starting
    at a random position for each episode, while the regular variants have the
    agent always starting in the corner opposite to the goal.

    ### Mission Space

    "get to the green goal square"

    ### Action Space

    | Num | Name         | Action       |
    |-----|--------------|--------------|
    | 0   | left         | Turn left    |
    | 1   | right        | Turn right   |
    | 2   | forward      | Move forward |
    | 3   | pickup       | Unused       |
    | 4   | drop         | Unused       |
    | 5   | toggle       | Unused       |
    | 6   | done         | Unused       |

    ### Observation Encoding

    - Each tile is encoded as a 3 dimensional tuple:
        `(OBJECT_IDX, COLOR_IDX, STATE)`
    - `OBJECT_TO_IDX` and `COLOR_TO_IDX` mapping can be found in
        [gym_minigrid/minigrid.py](gym_minigrid/minigrid.py)
    - `STATE` refers to the door state with 0=open, 1=closed and 2=locked

    ### Rewards

    A reward of '1' is given for success, and '0' for failure.

    ### Termination

    The episode ends if any one of the following conditions is met:

    1. The agent reaches the goal.
    2. Timeout (see `max_steps`).

    ### Registered Configurations

    - `MiniGrid-Empty-5x5-v0`
    - `MiniGrid-Empty-Random-5x5-v0`
    - `MiniGrid-Empty-6x6-v0`
    - `MiniGrid-Empty-Random-6x6-v0`
    - `MiniGrid-Empty-8x8-v0`
    - `MiniGrid-Empty-16x16-v0`

    """

    def __init__(self, size=8, agent_start_pos=(1, 1), agent_start_dir=0, **kwargs):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        mission_space = MissionSpace(
            mission_func=lambda: "Delivery packages while avoiding crashes."
        )

        super().__init__(
            mission_space=mission_space,
            width=13,
            height=13,
            max_steps=400000,
            # Set this to True for maximum speed
            see_through_walls=True,
            **kwargs
        )

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)
        for j in range(2, height - 2):
            if j % 3 == 1: continue
            if j % 3 == 2: continue
            self.grid.horz_wall(2, j, width - 4)


        self.grid.set(4,1,deliveryStationRLTile)
        self.grid.set(5,1,deliveryStationRLTile)

        self.grid.set(8,1,deliveryStation1Tile)
        self.grid.set(9,1,deliveryStation1Tile)

        self.grid.set(5,height - 2,deliveryStation2Tile)
        self.grid.set(6,height - 2,deliveryStation2Tile)

        self.grid.set(5,7,Adversary(2,"blue"))
        self.grid.set(3,4,Adversary(1,"green"))

        self.grid.set(3,7,Box("blue"))
        self.grid.set(4,7,Box("green"))
        self.grid.set(6,5,Box("blue"))
        self.grid.set(8,5,Box("green"))
        self.grid.set(7,5,Box("red"))
        self.grid.set(8,7,Box("green"))
        self.grid.set(4,4,Box("red"))


        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "Delivery packages while avoiding crashes."
