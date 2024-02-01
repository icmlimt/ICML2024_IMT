from gym_minigrid.minigrid import Goal, Grid, MiniGridEnv, MissionSpace, Lava, Wall, HeatMapTile
from gym_minigrid.window import Window 
from gym_minigrid.minigrid import run_BFS_reward

class Testing(MiniGridEnv):

    def __init__(self, size=9, agent_start_pos=(1, 1), agent_start_dir=0, **kwargs):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.state_ranking = dict()

        mission_space = MissionSpace(
            mission_func=lambda: "get to the green goal square"
        )
        
        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            max_steps=4 * size * size,
            # Set this to True for maximum speed
            see_through_walls=True,
            **kwargs
        )

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        for i in range(3,width - 4):
            self.put_obj(Lava(), i, 2)
        for i in range(width - 5, width - 1):
            self.put_obj(Lava(), i, 4)
        for i in range(1, width - 5):
            self.put_obj(Lava(), i, 6)

        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), width - 2, height - 2)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()
        
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
