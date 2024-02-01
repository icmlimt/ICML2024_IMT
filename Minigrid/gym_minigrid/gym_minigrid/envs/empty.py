from gym_minigrid.minigrid import Goal, Grid, MiniGridEnv, MissionSpace, Lava
from gym_minigrid.minigrid import run_BFS_reward

class EmptyEnv(MiniGridEnv):
    """
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

        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), width - 2, height - 2)
        self.put_obj(Lava(), 3, 2)
        self.put_obj(Lava(), 4, 2)
        self.put_obj(Lava(), 5, 2)

        self.put_obj(Lava(), 9, 3)

        self.put_obj(Lava(), 5, 4)
        self.put_obj(Lava(), 6, 4)

        self.put_obj(Lava(), 7, 6)
        self.put_obj(Lava(), 8, 6)

        self.put_obj(Lava(), 2, 6)
        self.put_obj(Lava(), 3, 6)
        self.put_obj(Lava(), 4, 6)

        self.put_obj(Lava(), 6, 8)
        self.put_obj(Lava(), 7, 8)

        self.put_obj(Lava(), 1, 9)
        self.put_obj(Lava(), 2, 9)
        self.put_obj(Lava(), 3, 9)
        #self.put_obj(Lava(), 3, 6)

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
        #if fwd_pos[0] == 4 and fwd_pos[1] == 3 and action == self.actions.forward:
        #    reward += 10
        #bfs reward
        reward += 1 * self.bfs_reward[self.agent_pos[0] + self.grid.width * self.agent_pos[1]]
        #negative reward for stepping in lava, positive reward for reaching goal
        if action == self.actions.forward:
            if fwd_cell is not None and fwd_cell.type == "goal":
                reward += 100
            elif fwd_cell is not None and fwd_cell.type == "lava":
                reward += -10

        if self.render_mode == "human":
            print("{}, rew: {}, info: {}\tterminated: {}".format(self.total_timesteps, reward, info, terminated or truncated))
        return obs, reward, terminated, truncated, info
