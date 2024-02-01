from gym_minigrid.minigrid import Goal, Grid, MiniGridEnv, MissionSpace, SlipperyNorth, SlipperySouth, SlipperyEast, SlipperyWest, Wall, HeatMapTile, Lava, Floor
from gym_minigrid.policyRepairEnv import PolicyRepairEnv
#from gym_minigrid.minigrid import run_BFS_reward

class OptimalityTaxi(PolicyRepairEnv):
    def __init__(self, width=17, height=9, agent_start_pos=(3, 1), agent_start_dir=1, **kwargs):
        super().__init__(
            agent_start_pos=agent_start_pos,
            agent_start_dir=agent_start_dir,
            width=width,
            height=height,
            **kwargs
        )
        self.picked_up_passenger = False
        self.passenger_pos = (5,4)

    def _gen_grid(self, width, height):
        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        self.grid.vert_wall(2, 2, 5)
        self.grid.horz_wall(2, 2, 3)
        self.grid.horz_wall(2, 6, 4)

        self.grid.vert_wall(4, 2, 2)
        self.grid.vert_wall(4, 5, 2)

        self.grid.vert_wall(7, 2, 3)
        self.grid.vert_wall(8, 2, 3)

        self.grid.vert_wall(10, 2, 5)

        self.grid.vert_wall(12, 2, 2)
        self.grid.vert_wall(12, 5, 2)

        self.grid.vert_wall(14, 2, 2)
        self.grid.vert_wall(14, 5, 2)

        self.grid.vert_wall(1,2, 5, SlipperyNorth)
        self.put_obj(SlipperyEast(), 2, 1)

        self.put_obj(SlipperyEast(), 9, 1)
        self.put_obj(SlipperySouth(), 9, 4)

        self.put_obj(SlipperySouth(), 11, 3)
        self.put_obj(SlipperySouth(), 11, 6)

        self.put_obj(SlipperyNorth(), 13, 2)
        self.put_obj(SlipperyNorth(), 13, 5)

        self.put_obj(SlipperySouth(), 15, 3)
        self.put_obj(SlipperySouth(), 15, 6)

        goalPos = (width-2, 1)
        self.put_obj(Goal(), *goalPos)
        self.grid.set_background(*self.passenger_pos, Floor("green"))

        # Place the agent
        if not self.agent_pos or (self.agent_pos[0] == -1 and self.agent_pos[1] == -1 and self.agent_dir == -1):
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir

        self.mission = "get to the green goal square"


    def reset(self, seed=None, state=None):
        self.picked_up_passenger = False
        return super().reset(state=state, seed=seed)

    def step(self, action):
        # Get the position in front of the agent
        fwd_pos = self.front_pos
        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)
        obs, reward, terminated, truncated, info = super().step(action)

        reward = -1

        if self.agent_pos == self.passenger_pos and not self.picked_up_passenger:
            self.picked_up_passenger = True
            self.grid.set_background(*self.passenger_pos, Floor("red"))
            reward += 100

        #negative reward for stepping in lava, positive reward for reaching goal
        if action == self.actions.forward:
            if fwd_cell is not None and fwd_cell.type == "goal" and self.picked_up_passenger:
                reward += 100
            elif fwd_cell is not None and fwd_cell.type == "goal" and not self.picked_up_passenger:
                reward -= 100
            elif fwd_cell is not None and fwd_cell.type == "lava":
                reward -= 100

        reward = reward * 0.01

        info["is_success"] = info["reached_goal"] and self.picked_up_passenger
        info["picked_up"] = self.picked_up_passenger
        if self.render_mode == "human":
            print("step: {}, reward: {}, info: {}".format(self.total_timesteps, reward, info))
        return obs, reward, terminated, truncated, info
