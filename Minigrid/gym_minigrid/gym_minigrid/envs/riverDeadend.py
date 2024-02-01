from gym_minigrid.minigrid import Goal, Grid, MiniGridEnv, MissionSpace, SlipperyNorth, SlipperySouth, SlipperyEast, SlipperyWest, Wall, HeatMapTile, Lava, Floor
from gym_minigrid.policyRepairEnv import PolicyRepairEnv
#from gym_minigrid.minigrid import run_BFS_reward

class RiverDeadend(PolicyRepairEnv):
    def __init__(self, width=15, height=15, agent_start_pos=(1, 1), agent_start_dir=1, training=False, **kwargs):
        super().__init__(
            agent_start_pos=agent_start_pos,
            agent_start_dir=agent_start_dir,
            width=width,
            height=height,
            **kwargs
        )
        self.training = training

    def _gen_grid(self, width, height):
        self.grid.grid = [Floor("black")] * width * height
        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        for x in range(2, 13, 3):
            for y in range(2, 13, 3):
                self.grid.wall_rect(x, y, 2, 2)

        for x in range(2, 14, 3):
            for y in range(1, 14, 3):
                self.grid.horz_wall(x, y, 1, SlipperyWest)

        for x in range(1, 14, 3):
            for y in range(2, 14, 3):
                self.grid.vert_wall(x, y, 1, SlipperyNorth)


        deadends = [(7,13),(10,7)]
        for deadend in deadends:
            self.put_obj(Wall(), *deadend)
        goalPos = [(width-2, height - 2)]
        for pos in goalPos:
            self.put_obj(Goal(), *pos)

        # Place the agent
        if self.training:
            self.place_agent()
            p = self.agent_pos
            # Do not train the agent from deadend initial states
            while (p[1] >= 11 and p[0] <= 7) or (8 <= p[0] and p[0] <= 10 and 5 <= p[1] and p[1] <= 7):
                self.place_agent()
                p = self.agent_pos
        elif not self.agent_pos or (self.agent_pos[0] == -1 and self.agent_pos[1] == -1 and self.agent_dir == -1):
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir

        self.mission = "get to the green goal square"


    def reset(self, seed=None, state=None):
        return super().reset(state=state, seed=seed)

    def step(self, action):
        # Get the position in front of the agent
        fwd_pos = self.front_pos
        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)
        obs, reward, terminated, truncated, info = super().step(action)

        reward = -1


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
