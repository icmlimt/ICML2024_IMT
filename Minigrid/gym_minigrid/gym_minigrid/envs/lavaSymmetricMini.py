from gym_minigrid.minigrid import Goal, Grid, MiniGridEnv, MissionSpace, Lava, Wall, HeatMapTile
from gym_minigrid.policyRepairEnv import PolicyRepairEnv
from gym_minigrid.minigrid import run_BFS_reward

class LavaSymmetricMini(PolicyRepairEnv):

    def __init__(self, size=6, agent_start_pos=(4, 4), agent_start_dir=0, **kwargs):
        super().__init__(
            size=size,
            agent_start_pos=agent_start_pos,
            agent_start_dir=agent_start_dir,
            **kwargs
        )

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), 1, 1)
        self.put_obj(Lava(), 2, 4)
        self.put_obj(Lava(), 4, 2)
        #self.put_obj(Lava(), 2, 2)
        # Place the agent
        if not self.agent_pos or (self.agent_pos[0] == -1 and self.agent_pos[1] == -1 and self.agent_dir == -1):
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir

        self.bfs_reward = run_BFS_reward(self.grid, (width - 2, height - 2))
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
                reward -=100

        if self.render_mode == "human":
            print("step: {}, reward: {}, info: {}\t\t\tthat terminated: {}".format(self.total_timesteps, reward, info, terminated or truncated))
        return obs, reward, terminated, truncated, info
