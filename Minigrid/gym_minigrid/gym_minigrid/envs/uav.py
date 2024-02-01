from gym_minigrid.minigrid import Goal, Grid, MiniGridEnv, MissionSpace, WindyNorth, WindySouth, WindyEast, WindyWest, Wall, HeatMapTile, Lava, Floor
from gym_minigrid.policyRepairEnv import PolicyRepairEnv
from gym_minigrid.minigrid import run_BFS_reward

class UAV(PolicyRepairEnv):
    def __init__(self, width=20, height=23, agent_start_pos=(1, 1), agent_start_dir=1, training=False, **kwargs):
        super().__init__(
            agent_start_pos=agent_start_pos,
            agent_start_dir=agent_start_dir,
            width=width,
            height=height,
            **kwargs
        )
        self.training = training
        self.noise = 0.1

    def _gen_grid(self, width, height):
        self.grid.grid = [WindyNorth()] * width * height
        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        self.grid.wall_rect(3,1,3,5, Lava)
        self.grid.wall_rect(4,2,1,3)

        self.grid.wall_rect(3,9,3,5, Lava)
        self.grid.wall_rect(4,10,1,3)

        self.grid.wall_rect(10,1,3,3, Lava)
        self.grid.wall_rect(11,2,1,1)

        self.grid.wall_rect(10, 7,3,5, Lava)
        self.grid.wall_rect(11, 8,1,3)

        self.grid.wall_rect(2, height-4,height-7 ,3, Lava)
        self.grid.wall_rect(3, height-3,height-9 ,1)

        #self.grid.horz_wall(1, height-2, width-2, obj_type=Lava)

        goalPos = [(width-2, 1)]
        for pos in goalPos:
            self.put_obj(Goal(), *pos)

        # Place the agent
        if self.training:
            self.place_agent()
            p = self.agent_pos
            self.put_obj(WindyNorth(), *p)
        elif not self.agent_pos or (self.agent_pos[0] == -1 and self.agent_pos[1] == -1 and self.agent_dir == -1):
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
        #fwd_cell = self.grid.get(*fwd_pos)
        #print(self.agent_pos)
        obs, reward, terminated, truncated, info = super().step(action)

        #reward = -1

        reward += self.bfs_reward[self.agent_pos[0] + self.grid.width * self.agent_pos[1]]

        #negative reward for stepping in lava, positive reward for reaching goal
        #if action == self.actions.forward:
        #current_cell = self.grid.get(*self.agent_pos)
        #print(current_cell)
        #if current_cell is not None and current_cell.type == "goal":
        #    reward += 100
        #elif current_cell is not None and current_cell.type == "lava":
        #    reward -= 100

        reward = reward * 0.01

        if self.render_mode == "human":
            print("step: {}, reward: {}, info: {}".format(self.total_timesteps, reward, info))
        return obs, reward, terminated, truncated, info
