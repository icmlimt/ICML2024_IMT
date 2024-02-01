from gym_minigrid.minigrid import Goal, Grid, MiniGridEnv, MissionSpace, SlipperyNorth, SlipperySouth, SlipperyEast, SlipperyWest, Wall, HeatMapTile, Lava, Floor
from gym_minigrid.policyRepairEnv import PolicyRepairEnv
from gym_minigrid.minigrid import run_BFS_reward

class CityDeadend(PolicyRepairEnv):
    def __init__(self, width=15, height=13, agent_start_pos=(1, 1), agent_start_dir=1, **kwargs):
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

        self.grid.wall_rect(2, 3, 2, 2)
        self.grid.wall_rect(2, 6, 2, 2)
        self.grid.wall_rect(2, 9, 2, 1)

        self.grid.wall_rect(5, 3, 3, 2)
        self.grid.wall_rect(5, 6, 3, 2)
        self.grid.wall_rect(5, 9, 3, 1)

        self.grid.wall_rect(9, 3, 3, 2)
        self.grid.wall_rect(9, 6, 3, 2)
        self.grid.wall_rect(9, 9, 3, 1)

        self.grid.vert_wall(5, 10, 2)

        self.grid.vert_wall(1, 3, 1, SlipperyNorth)
        self.grid.vert_wall(1, 6, 1, SlipperyNorth)
        self.grid.vert_wall(1, 9, 1, SlipperyNorth)

        self.grid.horz_wall(2, 5, 1, SlipperyWest)
        self.grid.horz_wall(5, 5, 2, SlipperyWest)
        self.grid.horz_wall(9, 5, 1, SlipperyWest)

        self.grid.horz_wall(3, 8, 1, SlipperyEast)
        self.grid.horz_wall(6, 8, 2, SlipperyEast)
        self.grid.horz_wall(11, 8, 1, SlipperyEast)

        #self.grid.horz_wall(5, 4, 3, SlipperyWest)
        #self.grid.horz_wall(5, 5, 3, SlipperyWest)
        #self.grid.horz_wall(5, 8, 3, SlipperyEast)

        #self.grid.horz_wall(9, 4, 3, SlipperyWest)
        #self.grid.horz_wall(9, 5, 3, SlipperyWest)
        #self.grid.horz_wall(9, 8, 3, SlipperyWest)

        #self.put_obj(SlipperyEast(), 12,7)
        #self.put_obj(SlipperyWest(), 5, 7)

        goalPos = [(width-2, height - 2), (6, height - 2)]
        for pos in goalPos:
            self.put_obj(Goal(), *pos)

        # Place the agent
        if not self.agent_pos or (self.agent_pos[0] == -1 and self.agent_pos[1] == -1 and self.agent_dir == -1):
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir

        self.bfs_reward = run_BFS_reward(self.grid, goalPos[0])
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

        #reward = -1
        reward = self.bfs_reward[self.agent_pos[0] + self.grid.width * self.agent_pos[1]]
        #negative reward for stepping in lava, positive reward for reaching goal
        if action == self.actions.forward:
            if fwd_cell is not None and fwd_cell.type == "goal":
                reward += 100
            elif fwd_cell is not None and fwd_cell.type == "lava":
                reward -= 100

        reward = reward * 0.01

        if self.render_mode == "human":
            print("step: {}, reward: {}, info: {}".format(self.total_timesteps, reward, info))
        return obs, reward, terminated, truncated, info
