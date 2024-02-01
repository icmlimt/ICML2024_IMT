from gym_minigrid.minigrid import Goal, Grid, MiniGridEnv, MissionSpace, SlipperyNorth, SlipperySouth, SlipperyEast, SlipperyWest, Wall, HeatMapTile, Lava
from gym_minigrid.policyRepairEnv import PolicyRepairEnv
from gym_minigrid.minigrid import run_BFS_reward

class Taxi(PolicyRepairEnv):

    def __init__(self, size=15, nr_cliffs=1, agent_start_pos=(1, 1), agent_start_dir=0, nr_structures=1, **kwargs):
        self.nr_structures = nr_structures
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
        self.grid.horz_wall(1,14)
        
        x_start = [1, 15, 29, 43, 57, 71, 85, 99, 113, 127, 141, 155]

        structure_user = [
            self.structure_0, 
            self.structure_1, 
            self.structure_2, 
            self.structure_3, 
            self.structure_4, 
            self.structure_5, 
            self.structure_6, 
            self.structure_7, 
            self.structure_8, 
            self.structure_0, 
            self.structure_1, 
            self.structure_2, 
            self.structure_3
        ]

        for i in range(self.nr_structures):
            structure_user[i](x_start[i], width, height)
                
            for j in range(2,15):
                self.put_obj(Wall(), x_start[i]-1, j)
        
        # Goal 
        goalPos = (width-2, 1)
        self.put_obj(Goal(), *goalPos)

        # Place the agent
        if not self.agent_pos or (self.agent_pos[0] == -1 and self.agent_pos[1] == -1 and self.agent_dir == -1):
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir

        self.bfs_reward = run_BFS_reward(self.grid, goalPos)
        self.mission = "get to the green goal square"

    def structure_0(self, x_start, height, width):
        for i in range(2, 13):
            self.put_obj(Wall(), x_start+1, i)
        
        self.put_obj(SlipperyNorth(), x_start, 2)
        self.put_obj(SlipperyWest(), x_start+1, 1)
        
        for i in range(x_start+2,x_start+12):
            self.put_obj(Wall(), i, 2)
            self.put_obj(Wall(), i, 8)
            self.put_obj(Wall(), i, 12)

        for i in range(x_start+3, x_start+13):
            self.put_obj(Wall(), i, 6)
            self.put_obj(Wall(), i, 10)
        
        for i in [x_start+3, x_start+7, x_start+11]:
            self.put_obj(Wall(), i, 4)
            self.put_obj(Wall(), i, 5)
        
        for i in [x_start+5, x_start+9]:
            self.put_obj(Wall(), i, 3)
            self.put_obj(Wall(), i, 4)

    def structure_1(self, x_offset, height, width):
        self.grid.vert_wall(x=x_offset + 1, y=2, length=11)
        self.grid.vert_wall(x=x_offset + 7, y=4, length=10)
        self.grid.vert_wall(x=x_offset + 9, y=2, length=10)
        self.grid.vert_wall(x=x_offset + 11, y=8, length=6)
        self.grid.horz_wall(x=x_offset + 1, y=2, length=11)
        self.grid.horz_wall(x=x_offset + 3, y=4, length=5)
        self.grid.horz_wall(x=x_offset + 3, y=10, length=5)
        self.grid.horz_wall(x=x_offset + 1, y=12, length=5)
        self.put_obj(SlipperyNorth(), x_offset, 2)
        self.put_obj(SlipperyWest(), x_offset+1, 1)
        # ?? self.put_obj(SlipperyNorth(), x_offset+12, 2)

    def structure_2(self, x_offset, height, width): 
        self.grid.horz_wall(x=x_offset + 1, y=1, length=5)
        self.grid.horz_wall(x=x_offset + 1, y=12, length=7)
        self.grid.horz_wall(x=x_offset + 3, y=7, length=5)
        self.grid.horz_wall(x=x_offset + 3, y=10, length=3)
        self.grid.horz_wall(x=x_offset + 7, y=3, length=4)
        self.grid.horz_wall(x=x_offset + 10, y=6, length=2)
        self.grid.horz_wall(x=x_offset + 10, y=10, length=2)
        self.grid.horz_wall(x=x_offset + 9, y=12, length=2)
        self.grid.horz_wall(x=x_offset + 9, y=8, length=2)
        self.grid.horz_wall(x=x_offset + 7, y=6, length=2)
        self.grid.horz_wall(x=x_offset + 7, y=10, length=2)
        self.grid.vert_wall(x=x_offset + 1, y=1, length=3)
        self.grid.vert_wall(x=x_offset + 1, y=5, length=8)
        self.grid.vert_wall(x=x_offset + 3, y=3, length=8)
        self.grid.vert_wall(x=x_offset + 5, y=1, length=3)
        self.grid.vert_wall(x=x_offset + 5, y=5, length=3)
        self.grid.vert_wall(x=x_offset + 7, y=3, length=5)
        self.grid.vert_wall(x=x_offset + 7, y=9, length=4)
        self.grid.vert_wall(x=x_offset + 10, y=5, length=8)

        self.put_obj(Wall(), x_offset + 12, 12)
        self.put_obj(Wall(), x_offset + 12, 8)
        self.put_obj(Wall(), x_offset + 12, 4)
        self.put_obj(SlipperyNorth(), x_offset + 0, 5)
        self.put_obj(SlipperyWest(), x_offset + 1, 4)
        self.put_obj(SlipperySouth(), x_offset + 2, 3)
        self.put_obj(SlipperyNorth(), x_offset + 2, 5)
        self.put_obj(SlipperySouth(), x_offset + 8, 12)
        self.put_obj(SlipperyWest(), x_offset + 3, 13)
        self.put_obj(SlipperySouth(), x_offset + 11, 12)
        self.put_obj(SlipperyWest(), x_offset + 7, 8)
        # ??? self.put_obj(SlipperyNorth(), x_offset + 11, 3)
        # ??? self.put_obj(SlipperyNorth(), x_offset + 12, 3)

    def structure_3(self, x_offset, height, width):
        self.grid.horz_wall(x=x_offset + 1, y=2, length=5)
        self.grid.horz_wall(x=x_offset + 1, y=8, length=5)
        self.grid.horz_wall(x=x_offset + 1, y=12, length=5)
        self.grid.horz_wall(x=x_offset + 7, y=12, length=5)
        self.grid.horz_wall(x=x_offset + 7, y=8, length=5)
        self.grid.horz_wall(x=x_offset + 9, y=5, length=4)
        self.grid.horz_wall(x=x_offset + 9, y=10, length=4)
        self.grid.horz_wall(x=x_offset + 0, y=6, length=2)
        self.grid.vert_wall(x=x_offset + 3, y=2, length=5)
        self.grid.vert_wall(x=x_offset + 3, y=8, length=5)
        self.grid.vert_wall(x=x_offset + 5, y=5, length=3)
        self.grid.vert_wall(x=x_offset + 7, y=1, length=3)
        self.grid.vert_wall(x=x_offset + 7, y=7, length=6)
        self.grid.vert_wall(x=x_offset + 9, y=2, length=5)
        self.grid.vert_wall(x=x_offset + 11, y=1, length=3)
        self.put_obj(Wall(), 6, 5)
        self.put_obj(SlipperyWest(), x_offset + 1, 1)
        self.put_obj(SlipperyNorth(), x_offset, 2)
        self.put_obj(SlipperyEast(), x_offset + 1, 7)
        self.put_obj(SlipperyWest(), x_offset + 3, 7)
        self.put_obj(SlipperyWest(), x_offset + 7, 4)
        self.put_obj(SlipperySouth(), x_offset + 6, 12)



    def structure_4(self, x_offset, height, width): 
        self.grid.horz_wall(x=x_offset + 1, y=1, length=4)
        self.grid.horz_wall(x=x_offset + 1, y=3, length=4)
        self.grid.horz_wall(x=x_offset + 1, y=6, length=4)
        self.grid.horz_wall(x=x_offset + 1, y=10, length=4)
        self.grid.horz_wall(x=x_offset + 1, y=12, length=6)
        self.grid.horz_wall(x=x_offset + 3, y=8, length=4)
        self.grid.horz_wall(x=x_offset + 8, y=2, length=2)
        self.grid.horz_wall(x=x_offset + 8, y=6, length=2)
        self.grid.horz_wall(x=x_offset + 8, y=10, length=2)
        self.grid.horz_wall(x=x_offset + 8, y=12, length=4)
        self.grid.horz_wall(x=x_offset + 10, y=4, length=2)
        self.grid.horz_wall(x=x_offset + 10, y=8, length=2)
        self.grid.vert_wall(x=x_offset + 1, y=3, length=8)
        self.grid.vert_wall(x=x_offset + 6, y=2, length=11)
        self.grid.vert_wall(x=x_offset + 8, y=2, length=9)
        self.grid.vert_wall(x=x_offset + 11, y=1, length=12)
        self.put_obj(SlipperyWest(), x_offset + 1, 2)
        self.put_obj(SlipperyNorth(), x_offset, 3)
        self.put_obj(SlipperyNorth(), x_offset, 12)
        self.put_obj(SlipperyWest(), x_offset + 1, 11)
        self.put_obj(SlipperyNorth(), x_offset + 7, 2)
        self.put_obj(SlipperyWest(), x_offset + 8, 1)
        # ??? self.put_obj(SlipperySouth(), x_offset + 5, 3)


    def structure_5(self, x_offset, height, width): 
        self.grid.horz_wall(x=x_offset + 1, y=1, length=7)
        self.grid.horz_wall(x=x_offset + 0, y=3, length=3)
        self.grid.horz_wall(x=x_offset + 1, y=5, length=2)
        self.grid.horz_wall(x=x_offset + 1, y=7, length=2)
        self.grid.horz_wall(x=x_offset + 1, y=9, length=5)
        self.grid.horz_wall(x=x_offset + 1, y=12, length=2)
        self.grid.horz_wall(x=x_offset + 4, y=3, length=4)
        self.grid.horz_wall(x=x_offset + 4, y=5, length=6)
        self.grid.horz_wall(x=x_offset + 6, y=7, length=6)
        self.grid.horz_wall(x=x_offset + 6, y=11, length=2)
        self.grid.horz_wall(x=x_offset + 6, y=12, length=2)
        self.grid.horz_wall(x=x_offset + 9, y=3, length=3)
        self.grid.horz_wall(x=x_offset + 9, y=12, length=3)

        self.grid.vert_wall(x=x_offset + 2, y=3, length=5)
        self.grid.vert_wall(x=x_offset + 1, y=9, length=4)
        self.grid.vert_wall(x=x_offset + 4, y=5, length=5)
        self.grid.vert_wall(x=x_offset + 4, y=11, length=2)

        self.grid.vert_wall(x=x_offset + 7, y=1, length=3)
        self.grid.vert_wall(x=x_offset + 7, y=7, length=6)
        self.grid.vert_wall(x=x_offset + 9, y=2, length=4)
        self.grid.vert_wall(x=x_offset + 11, y=5, length=3)
        self.grid.vert_wall(x=x_offset + 11, y=9, length=4)

        self.put_obj(Wall(), x_offset + 11, 1)
        self.put_obj(SlipperyNorth(), x_offset + 3, 3)
        self.put_obj(SlipperyNorth(), x_offset + 3, 5)
        self.put_obj(SlipperyWest(), x_offset + 4, 2)
        self.put_obj(SlipperyWest(), x_offset + 4, 4)
        self.put_obj(SlipperySouth(), x_offset, 7)
        self.put_obj(SlipperyNorth(), x_offset, 9)
        self.put_obj(SlipperySouth(), x_offset + 3, 12)
        self.put_obj(SlipperySouth(), x_offset + 5, 12)
        self.put_obj(SlipperySouth(), x_offset + 8, 12)
        self.put_obj(SlipperySouth(), x_offset + 12, 12)
        self.put_obj(SlipperyWest(), x_offset + 4, 13)
        self.put_obj(SlipperyWest(), x_offset + 6, 13)

       
    def structure_6(self, x_offset, height, width): 
        self.grid.horz_wall(x=x_offset + 3, y=2, length=5)
        self.grid.horz_wall(x=x_offset + 3, y=4, length=3)
        self.grid.horz_wall(x=x_offset + 3, y=8, length=3)
        self.grid.horz_wall(x=x_offset + 3, y=12, length=3)
        self.grid.horz_wall(x=x_offset + 5, y=6, length=3)
        self.grid.horz_wall(x=x_offset + 5, y=10, length=3)
        self.grid.horz_wall(x=x_offset + 7, y=3, length=2)
        self.grid.horz_wall(x=x_offset + 9, y=1, length=2)
        self.grid.horz_wall(x=x_offset + 9, y=5, length=2)
        self.grid.horz_wall(x=x_offset + 9, y=8, length=4)
        self.grid.horz_wall(x=x_offset + 9, y=12, length=3)
        self.grid.vert_wall(x=x_offset + 1, y=1, length=12)
        self.grid.vert_wall(x=x_offset + 3, y=4, length=11)
        self.grid.vert_wall(x=x_offset + 7, y=5, length=9)
        self.grid.vert_wall(x=x_offset + 10, y=1, length=5)
        self.grid.vert_wall(x=x_offset + 9, y=8, length=5)
        self.grid.vert_wall(x=x_offset + 11, y=10, length=3)
        self.grid.vert_wall(x=x_offset + 12, y=2, length=3)
        self.put_obj(SlipperySouth(), x_offset + 2, 12)
        self.put_obj(SlipperyWest(), x_offset + 3, 13)
        self.put_obj(SlipperyNorth(), x_offset + 8, 8)
        self.put_obj(SlipperyWest(), x_offset + 9, 6)
        self.put_obj(SlipperyWest(), x_offset + 9, 7)


    def structure_7(self, x_offset, height, width): 
        self.grid.horz_wall(x=x_offset + 1, y=2, length=5)
        self.grid.horz_wall(x=x_offset + 1, y=6, length=5)
        self.grid.horz_wall(x=x_offset + 1, y=12, length=5)
        self.grid.horz_wall(x=x_offset + 3, y=4, length=5)
        self.grid.horz_wall(x=x_offset + 3, y=8, length=5)
        self.grid.horz_wall(x=x_offset + 9, y=8, length=3)
        self.grid.vert_wall(x=x_offset + 1, y=2, length=11)
        self.grid.vert_wall(x=x_offset + 3, y=8, length=3)
        self.grid.vert_wall(x=x_offset + 5, y=10, length=3)
        self.grid.vert_wall(x=x_offset + 7, y=2, length=12)
        self.grid.vert_wall(x=x_offset + 9, y=2, length=10)
        self.grid.vert_wall(x=x_offset + 11, y=3, length=4)
        self.grid.vert_wall(x=x_offset + 11, y=11, length=4)
        self.put_obj(SlipperyNorth(), x_offset, 2)
        self.put_obj(SlipperyWest(), x_offset + 1, 1)
        self.put_obj(SlipperyWest(), x_offset + 7, 1)
        self.put_obj(SlipperyNorth(), x_offset + 8, 2)
        self.put_obj(SlipperyNorth(), x_offset + 10, 3)
        self.put_obj(SlipperyWest(), x_offset + 11, 1)
        self.put_obj(SlipperyWest(), x_offset + 11, 2)
        # ? self.put_obj(SlipperyNorth(), 6, 1)


    def structure_8(self, x_offset, height, width): 
        self.grid.horz_wall(x=x_offset + 3, y=10, length=5)
        self.grid.horz_wall(x=x_offset + 7, y=11, length=5)
        self.grid.vert_wall(x=x_offset + 1, y=1, length=12)
        self.grid.vert_wall(x=x_offset + 3, y=2, length=10)
        self.grid.vert_wall(x=x_offset + 5, y=1, length=8)
        self.grid.vert_wall(x=x_offset + 7, y=2, length=10)
        self.grid.vert_wall(x=x_offset + 9, y=1, length=9)
        self.grid.vert_wall(x=x_offset + 11, y=2, length=11)
        self.put_obj(Wall(), x_offset + 3, 13)
        self.put_obj(Wall(), x_offset + 7, 13)
        self.put_obj(SlipperySouth(), x_offset + 2, 11)
        self.put_obj(SlipperyWest(), x_offset + 3, 12)
        # ? self.put_obj(SlipperyNorth(), x_offset + 12, 2)
    

    def step(self, action):
        # Get the position in front of the agent
        fwd_pos = self.front_pos
        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)
        obs, reward, terminated, truncated, info = super().step(action)

        #bfs reward
        #reward = self.bfs_reward[self.agent_pos[0] + self.grid.width * self.agent_pos[1]]
        #negative reward for stepping in lava, positive reward for reaching goal
        if action == self.actions.forward:
            if fwd_cell is not None and fwd_cell.type == "goal":
                reward += 100
            elif fwd_cell is not None and fwd_cell.type == "lava":
                reward -=100

        if self.render_mode == "human":
            print("step: {}, reward: {}, info: {}\t\t\tthat terminated: {}".format(self.total_timesteps, reward, info, terminated or truncated))
        return obs, reward, terminated, truncated, info
