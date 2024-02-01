import numpy as np
from gym import spaces
from gym_minigrid.minigrid import Goal, Grid, MiniGridEnv, MissionSpace, Floor, Adversary, Box, SlipperyNorth
from gym_minigrid.Task import DoRandom, TaskManager, DoNothing, GoTo, PlaceObject, PickUpObject, Task

deliveryStation1Tile = Floor("blue")
deliveryStation2Tile = Floor("green")
deliveryStationRLTile = Floor("red")
deliveryRegion1Tile = Floor("lightblue")
deliveryRegion2Tile = Floor("lightgreen")

def random_boxes_in_region(env, amount, xTop, yTop, width, height, agent_pos, color="red"):
    totalBoxCount = amount
    while totalBoxCount > 0:
        x = np.random.choice(range(xTop, width - 1))
        y = np.random.choice(range(yTop, height - 1))
        if env.grid.get(x,y) == None and (x,y) != agent_pos:
            env.put_obj(Box(color),x,y)
            totalBoxCount -= 1

class WareHouse(MiniGridEnv):

    def __init__(self, agent_start_pos=(1, 7), agent_start_dir=0, sb3_mode=False, width=7, **kwargs):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        mission_space = MissionSpace(
            mission_func=lambda: "Avoid crashing into the other agents."
        )

        super().__init__(
            mission_space=mission_space,
            width=width,
            height=17,
            # Set this to True for maximum speed
            see_through_walls=True,
            **kwargs
        )

    def _gen_grid(self, width, height):
        # Create an empty grid
        assert(width > 6)
        self.grid = Grid(width, height)

        self.grid.set_background(1,7,deliveryRegion1Tile)
        self.grid.set_background(1,8,deliveryRegion1Tile)
        self.grid.set_background(1,9,deliveryRegion1Tile)

        self.adversaries = []
        self.add_adversary(5,5,"blue",1)
        self.add_adversary(10,9,"green",3)

        self.grid.wall_rect(0, 0, width, height)
        self.grid.vert_wall(1,1 , 6)
        self.grid.vert_wall(1,10, 6)
        
        cols = (self.width - 4) // 3
        for r in range(0, cols):
            print(r)
            self.grid.vert_wall(4 + r * 3,2 , 5)
            self.grid.vert_wall(4 + r * 3,10, 5)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()
        random_boxes_in_region(self, 1, 3,1, self.width-1, self.height-1, self.agent_pos, "red")
        random_boxes_in_region(self, 1, 3,1, self.width-1, self.height-1, self.agent_pos, "blue")
        random_boxes_in_region(self, 1, 3,1, self.width-1, self.height-1, self.agent_pos, "green")

        self.mission = "Avoid crashing into the other agents."

    def step(self, action):
        self.reward = 0.0


        # need to augment this just like the agent, and we can hardcode a policy to follow
        for adversary in self.adversaries:
            blocked_positions = [adv.cur_pos for adv in self.adversaries]
            agent_pos = self.agent_pos
            advsersary_action = self.get_adversary_action(adversary)
            self.move_adversary(adversary, advsersary_action, blocked_positions, agent_pos)

        # get obs from env
        obs, reward, terminated, truncated, info = super().step(action)

        # reward function and done function are custom
        reward = self.reward
        terminated = self.done

        return obs, reward, terminated, truncated, info

    # Todo make this an intelligent adversary policy
    def get_adversary_action(self, adversary):
        return adversary.task_manager.get_best_action(adversary.cur_pos, adversary.dir_vec(), adversary.carrying, self)


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
                self.reward -= 0.1

            elif (fwd_cell is None or fwd_cell.can_overlap()) and not tuple(fwd_pos) in blocked_positions:
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
        return [adversary.color for adversary in self.adversaries]

    def get_property_map(self):
        return { "blue" : ("SafetyAGBlue", 'concat(<_, PreSafety, lambda=0.89> <<Blue>> Pmax=? [ G !"BlueOnGreen"  ], <SafetyAGBlue, PreSafety, lambda=0.89> <<Agent>> Pmax=? [G<10 !"crash" ] );'), "green": ("SafetyAGGreen", 'concat(<_, PreSafety, lambda=0.89> <<Green>> Pmax=? [ G !"GreenOnBlue"  ], <SafetyAGGreen, PreSafety, lambda=0.89> <<Agent>> Pmax=? [G<10 !"crash" ] );') }

    def get_shield_info(self):
        #adv_positions = [(adv.color, adv.cur_pos[0], adv.cur_pos[1]) for adv in self.adversaries]
        adv_positions = {adv.color : adv.cur_pos for adv in self.adversaries}
        adv_directions = [adv.dir_vec() for adv in self.adversaries]
        return self.agent_pos, self.agent_dir, adv_positions, adv_directions
