from gym_minigrid.minigrid import Goal, Grid, MiniGridEnv, MissionSpace, SlipperyNorth, SlipperySouth, SlipperyEast, SlipperyWest, Wall, HeatMapTile, Lava
from gym_minigrid.policyRepairEnv import PolicyRepairEnv
from gym_minigrid.minigrid import run_BFS_reward

class CliffWalking(PolicyRepairEnv):
    def __init__(self, width=9, height=9, nr_cliffs=1, agent_start_pos=(1, 1), agent_start_dir=0, training=False, **kwargs):
        self.nr_cliffs = nr_cliffs
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
        cliffPosTop = 1
        cliffPosBot = height - 2
        #self.grid.horz_wall(1,5)

        #north cliffs
        lava_pos = [6, 14, 22, 30]
        for i in range(self.nr_cliffs):
            self.put_obj(Lava(), lava_pos[i], 1)
            self.put_obj(SlipperySouth(), lava_pos[i], cliffPosTop + 1)
            self.put_obj(SlipperySouth(), lava_pos[i]+1, cliffPosTop + 1)
            self.put_obj(SlipperySouth(), lava_pos[i]-1, cliffPosTop + 1)
            self.put_obj(SlipperyEast(), lava_pos[i]+1, cliffPosTop)
            self.put_obj(SlipperyWest(), lava_pos[i]-1, cliffPosTop)

        #south cliffs
        lava_pos = [2, 10, 18, 26]
        for i in range(self.nr_cliffs):
            self.put_obj(Lava(), lava_pos[i], cliffPosBot)
            self.put_obj(SlipperyNorth(), lava_pos[i], cliffPosBot - 1)
            self.put_obj(SlipperyNorth(), lava_pos[i]+1, cliffPosBot - 1)
            self.put_obj(SlipperyNorth(), lava_pos[i]-1, cliffPosBot - 1)
            self.put_obj(SlipperyEast(), lava_pos[i]+1, cliffPosBot)
            self.put_obj(SlipperyWest(), lava_pos[i]-1, cliffPosBot)


        goalPos = (width-2, height - 2)
        self.put_obj(Goal(), *goalPos)

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
        #if self.grid.width > 16:
        #    pos = info["pos"]
        #    if pos[0] > 7:
        #        #reward += 10
        #        reward = self.bfs_reward[self.agent_pos[0] + self.grid.width * self.agent_pos[1]]
        #    else:
        #        reward = 3 * self.bfs_reward[self.agent_pos[0] + self.grid.width * self.agent_pos[1]]
        #else:
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
