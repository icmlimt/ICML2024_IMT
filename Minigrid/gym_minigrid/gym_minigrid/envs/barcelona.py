from gym_minigrid.minigrid import Goal, Grid, MiniGridEnv, MissionSpace, OneWayNorth, OneWaySouth, OneWayEast, OneWayWest, Wall, HeatMapTile, Lava, Floor
from gym_minigrid.policyRepairEnv import PolicyRepairEnv
from gym_minigrid.minigrid import run_BFS_reward

class Barcelona(PolicyRepairEnv):
    def __init__(self, width=27, height=27, agent_start_pos=(1, 1), agent_start_dir=1, training=False, **kwargs):
        super().__init__(
            agent_start_pos=agent_start_pos,
            agent_start_dir=agent_start_dir,
            width=width,
            height=height,
            **kwargs
        )
        self.actions = MiniGridEnv.MovementOnlyActions
        self.training = training
        self.training = False

    def _gen_grid(self, width, height):
        self.grid.grid = [Floor("black")] * width * height
        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)


        for x in range(2, 24, 6):
            for y in range(2, 24, 6):
                if x == 2 and y == 2: continue
                elif x == 8 and y == 20: continue
                elif x == 20 and y == 8: continue
                self.grid.wall_rect(x, y, 5, 5)
                self.grid.wall_rect(x+1, y+1, 3, 3)
                self.grid.wall_rect(x+2, y+2, 1, 1)

        self.grid.wall_rect(2,2,5,2)
        self.grid.wall_rect(2,5,5,2)

        self.grid.wall_rect(8,20,3,5)
        self.grid.wall_rect(9,21,1,3)
        self.grid.wall_rect(12,20,1,5)

        self.grid.wall_rect(22, 10, 3,3)
        self.grid.wall_rect(23, 11, 1,1)
        self.grid.horz_wall(20,8,5)
        self.grid.vert_wall(20,8,5)

        for x in range(2, 25, 6):
            for y in [1, 13, 19]:
                if y == 13 and x == 14: continue
                self.grid.horz_wall(x, y, 5, OneWayEast)
        for x in range(2, 25, 6):
            for y in [7, 25]:
                self.grid.horz_wall(x, y, 5, OneWayWest)

        for y in range(2, 25, 6):
            for x in [7]:
                self.grid.vert_wall(x, y, 5, OneWaySouth)
        for y in range(2, 25, 6):
            for x in [13, 25]:
                self.grid.vert_wall(x, y, 5, OneWayNorth)

        self.grid.vert_wall(1, 2, 5, OneWayNorth)
        self.grid.vert_wall(19, 14, 5, OneWaySouth)

        # Small streets:
        self.grid.horz_wall(14, 10, 5, OneWayEast)
        self.grid.horz_wall(2, 22, 5, OneWayWest)
        self.grid.vert_wall(15, 20, 5, OneWayNorth)

        self.grid.set(1,4, Floor("black"))
        self.grid.set(7,4, Floor("black"))

        self.grid.set(13,10, Floor("black"))

        self.grid.set(25,9, Floor("black"))
        self.grid.set(21,13, Floor("black"))

        self.grid.set(7,22, Floor("black"))

        self.grid.set(11,19, Floor("black"))
        self.grid.set(11,25, Floor("black"))

        self.grid.set(15,19, Floor("black"))
        self.grid.set(15,25, Floor("black"))

        goalPos = [(16, 14)]
        for pos in goalPos:
            self.put_obj(Goal(), *pos)

        # Place the agent
        if self.training:
            self.place_agent()
            p = self.agent_pos
            # Do not train the agent from deadend initial states
            while (20 <= p[0] and p[0] <= 26 and 20 <= p[1] and p[1] <= 26):
                self.place_agent()
                p = self.agent_pos
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
