from gym_minigrid.minigrid import Goal, Grid, MiniGridEnv, MissionSpace, SlipperyNorth, SlipperySouth, SlipperyEast, SlipperyWest, Wall, HeatMapTile, Lava, Floor
from gym_minigrid.policyRepairEnv import PolicyRepairEnv
from gym_minigrid.minigrid import run_BFS_reward

class OptimalityTesting(PolicyRepairEnv):
    def __init__(self, width=13, height=9, agent_start_pos=(1, 1), agent_start_dir=0, **kwargs):
        super().__init__(
            agent_start_pos=agent_start_pos,
            agent_start_dir=agent_start_dir,
            width=width,
            height=height,
            **kwargs
        )

    def _gen_grid(self, width, height):
        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        self.grid.horz_wall(2,2,2)
        self.grid.vert_wall(2,2,5)
        self.grid.vert_wall(4,5,3)

        self.grid.vert_wall(5,1,2)
        self.grid.vert_wall(5,4,2)
        self.grid.horz_wall(7,2,3)
        self.grid.vert_wall(7,2,5)
        self.grid.horz_wall(9,4,3)

        self.put_obj(SlipperyNorth(), 1,2)
        self.put_obj(SlipperySouth(), 6,2)
        self.put_obj(SlipperyNorth(), 6,4)

        goalPos = (width-2, height - 2)
        self.put_obj(Goal(), *goalPos)
        self.grid.set_background(3,3, Floor("green"))

        # Place the agent
        if not self.agent_pos or (self.agent_pos[0] == -1 and self.agent_pos[1] == -1 and self.agent_dir == -1):
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir

        self.bfs_reward = run_BFS_reward(self.grid, goalPos)
        self.mission = "get to the green goal square"



    def step(self, action):
        # Get the position in front of the agent
        fwd_pos = self.front_pos
        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)
        obs, reward, terminated, truncated, info = super().step(action)

        #bfs reward
        reward = self.bfs_reward[self.agent_pos[0] + self.grid.width * self.agent_pos[1]]
        #negative reward for stepping in lava, positive reward for reaching goal
        if action == self.actions.forward:
            if fwd_cell is not None and fwd_cell.type == "goal":
                reward += 100
            elif fwd_cell is not None and fwd_cell.type == "lava":
                reward -= 100

        if self.render_mode == "human":
            print("step: {}, reward: {}, info: {}".format(self.total_timesteps, reward, info))
        return obs, reward, terminated, truncated, info
