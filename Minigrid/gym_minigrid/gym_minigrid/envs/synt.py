from gym_minigrid.minigrid import Goal, Grid, MiniGridEnv, MissionSpace, SlipperyNorth, Wall, HeatMapTile, Lava, Floor
from gym_minigrid.policyRepairEnv import PolicyRepairEnv
from gym_minigrid.minigrid import run_BFS_reward

DEFAULT_TILE_WIDTH  = 6
DEFAULT_TILE_HEIGHT = 5

class SyntExample(PolicyRepairEnv):

    def __init__(self, height=7, width=8, agent_start_pos=(1, 1), agent_start_dir=0, **kwargs):
        super().__init__(
            height=height,
            width=width,
            agent_start_pos=agent_start_pos,
            agent_start_dir=agent_start_dir,
            **kwargs
        )

    def _gen_grid(self, width, height):

        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        #self.PipeCorridor.place(self.grid, 3,0,width=7,orientation="V")
        #self.PipeJunction.place(self.grid, 2,4,orientation="R")

        self.put_obj(Goal(), width - 2, height - 2)
        self.grid.vert_wall(2,2,3,obj_type=Lava)
        self.grid.vert_wall(3,2,3,obj_type=Lava)


        #self.TileT.place(self.grid, 10,1,height=6, orientation="D")

        # setting agent, mission and bfs
        if not self.agent_pos or (self.agent_pos[0] == -1 and self.agent_pos[1] == -1 and self.agent_dir == -1):
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir

        self.mission = "get to the green goal square"

    def step(self, action):
        # Get the position in front of the agent
        fwd_pos = self.front_pos

        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)
        obs, reward, terminated, truncated, info = super().step(action)

        # bfs reward
        reward = 10

        # negative reward for stepping in lava, positive reward for reaching goal
        if action == self.actions.forward:
            if fwd_cell is not None and fwd_cell.type == "goal":
                reward += 100
            elif fwd_cell is not None and fwd_cell.type == "lava":
                reward -=100

        if self.render_mode == "human":
            print("step: {}, reward: {}, info: {}\t\t\tthat terminated: {}".format(self.total_timesteps, reward, info, terminated or truncated))

        return obs, reward, terminated, truncated, info

    # ---------------------------------------------------------------------------------- #

    class TileColumns:
        """
        ┌────────────┐
        │WGWGWGWGWGWG│O
        │            │
        │WGWGWGWGWGWG│
        │            │
        │WGWGWGWGGWWG│
        └────────────┘
        """

        @staticmethod
        def place(grid : Grid, x : int, y : int, width : int = None, height : int = None) -> tuple:
            width_  = width  if width  is not None else DEFAULT_TILE_WIDTH
            height_ = height if height is not None else DEFAULT_TILE_HEIGHT
            for i in range(0, height_ - height_ // 2):
                grid.horz_wall(x, y + i * 2, width_)
            return (x + width_, y)

    # ---------------------------------------------------------------------------------- #

    class PipeCorridor:
        """
        ┌────────────┐    ┌────────────┐
        │WG   ..   WG│O   │WGWGWGWGWGWG│O
        │WG   ..   WG│    │     ..     │
        │WG   ..   WG│    │     ..     │
        │WG   ..   WG│    │     ..     │
        │WG   ..   WG│    │WGWGWGWGWGWG│
        └────────────┘    └────────────┘
              V                 H                } orientation (vertical / horizontal)
        """

        @staticmethod
        def place(grid : Grid, x : int, y : int, width : int = None, height : int = None, orientation : str = None) -> tuple:
            height_ = height if height is not None else DEFAULT_TILE_HEIGHT
            width_ = width if width is not None else DEFAULT_TILE_WIDTH
            orientation_ = orientation if orientation is not None else 'V'
            assert(orientation_ in ['V', 'H'])
            if orientation_ == 'V':
                grid.vert_wall(x, y, height_)
                grid.vert_wall(x + width_ - 1, y, height_)
            else:
                grid.horz_wall(x, y, width_)
                grid.horz_wall(x, y + height_ - 1, width_)
            return (x + width_, y)

        # ---------------------------------------------------------------------------------- #

    class PipeEnd:
        """
        ┌────────────┐     ┌────────────┐     ┌────────────┐    ┌────────────┐
        │WGWGWGWGWGWG│O    │WGWGWGWGWGWG│O    │WG        WG│O   │WGWGWGWGWGWG│
        │WG          │     │          WG│     │WG        WG│    │WG        WG│
        │WG          │     │          WG│     │WG        WG│    │WG        WG│
        │WG          │     │          WG│     │WG        WG│    │WG        WG│
        │WGWGWGWGWGWG│     │WGWGWGWGWGWG│     │WGWGWGWGWGWG│    │WG        WG│
        └────────────┘     └────────────┘     └────────────┘    └────────────┘
               L                  R                  D                 U          }  orientation (left, right, down, up)
        """

        @staticmethod
        def place(grid : Grid, x : int, y : int, width : int = None, height : int = None, orientation : str = None) -> tuple:
            width_  = width  if width  is not None else DEFAULT_TILE_WIDTH
            height_ = height if height is not None else DEFAULT_TILE_HEIGHT
            orientation_ = orientation if orientation is not None else 'D'
            assert(orientation_ in ['L', 'R', 'D', 'U'])
            if orientation_ == 'L':
                grid.horz_wall(x, y, width_)
                grid.horz_wall(x, y + height_ - 1, width_)
                grid.vert_wall(x, y, height_)
            elif orientation_ == 'R':
                grid.horz_wall(x, y, width_)
                grid.horz_wall(x, y + height_ - 1, width_)
                grid.vert_wall(x, y + width_ - 1, height_)
            elif orientation_ == 'D':
                grid.horz_wall(x, y + height_ - 1, width_)
                grid.vert_wall(x, y, height_)
                grid.vert_wall(x + width_ - 1, y, height_)
            elif orientation_ == 'U':
                grid.horz_wall(x, y, width_)
                grid.vert_wall(x, y, height_)
                grid.vert_wall(x, y + width_ - 1, height_)
            return (x + width_, y)

    # ---------------------------------------------------------------------------------- #

    class PipeJunction:
        """
        ┌────────────┐          ┌────────────┐
        │      WG  WG│O         |WG  WG      │O
        │..WGWGWG  WG│          │WG  WGWGWG..│
        │          WG│          │WG          │
        │..WGWGWG  WG│          │WG  WGWGWG..│
        │      ..  ..│          │..  ..      │
        └────────────┘          └────────────┘
               R                       L               } orientation (side of crossing)
        """

        @staticmethod
        def place(grid : Grid, x : int, y : int, width : int = None, height : int = None, orientation : str = None) -> tuple:
            width_       = width  if width  is not None else DEFAULT_TILE_WIDTH
            height_      = height if height is not None else DEFAULT_TILE_HEIGHT
            orientation_ = orientation if orientation is not None else 'R'
            assert(width_ > 3 and height_ > 3 and orientation_ in ['L', 'R'])
            if orientation_ != 'L':
                grid.vert_wall(x + width_ - 1, y, height_)
                grid.vert_wall(x + width_ - 3, y, 2)
                grid.vert_wall(x + width_ - 3, y + 3, height_ - 3)
                grid.horz_wall(x, y + 1, width_ - 3)
                grid.horz_wall(x, y + 3, width_ - 3)
            else:
                grid.vert_wall(x, y, height_)
                grid.vert_wall(x + 2, y, 2)
                grid.vert_wall(x + 2, y + 3, height_ - 3)
                grid.horz_wall(x + 2, y + 1, width_ - 1)
                grid.horz_wall(x + 2, y + 3, width_ - 1)
            return (x + width_, y)

     # ---------------------------------------------------------------------------------- #

    class TileColumnsVertical:
        """
        ┌────────────┐     ┌────────────┐
        │WG  WG    WG│O    │WG    WG  WG│O
        │WG  WG    WG│     │WG    WG  WG│
        │WG  WG    WG│     │WG    WG  WG│
        │WG  WG    WG│     │WG    WG  WG│
        │WG  WG    WG│     │WG    WG  WG│
        └────────────┘     └────────────┘
              L                  R             } side (of the small corridor)
        """

        @staticmethod
        def place(grid : Grid, x : int, y : int, height : int = None, side : str = None) -> tuple:
            height_ = height if height is not None else DEFAULT_TILE_HEIGHT
            side_ = side if side is not None else 'L'
            assert(side_ in ['L', 'R'])
            for i in [0, 2 if side_ == 'L' else 3, 5]:
                grid.vert_wall(x + i, y, height_)

    # ---------------------------------------------------------------------------------- #

    class TileT:
        """
        ┌────────────┐   ┌────────────┐   ┌────────────┐   ┌────────────┐
        │WGWGWGWGWGWG│O  │     WG     │O  │WG          │O  │          WG│O
        │     WG     │   │     WG     │   │WG          │   │          WG│
        │     WG     │   │     WG     │   │WGWGWGWGWGWG│   │WGWGWGWGWGWG│
        │     WG     │   │     WG     │   │WG          │   │          WG│
        │     WG     │   │WGWGWGWGWGWG│   │WG          │   |          WG|
        └────────────┘   └────────────┘   └────────────┘   └────────────┘
               U                D                L                R           } orientation (up, down, left, right)
        """

        @staticmethod
        def place(grid : Grid, x : int, y : int, width : int = None, height : int = None, orientation : str = None) -> tuple:
            width_  = width  if width  is not None else DEFAULT_TILE_WIDTH
            height_ = height if height is not None else DEFAULT_TILE_HEIGHT
            orientation_ = orientation if orientation is not None else 'U'
            assert(orientation_ in ['U', 'D', 'L', 'R'])
            if orientation_ == 'U':
                grid.horz_wall(x, y, width_)
                grid.vert_wall(x + width_ // 2, y, height_)
            elif orientation_ == 'D':
                grid.horz_wall(x, y + height_ - 1, width_)
                grid.vert_wall(x + width_ // 2, y, height_)
            elif orientation_ == 'L':
                grid.horz_wall(x, y + (height_ - 1) // 2, width_)
                grid.vert_wall(x, y, height_)
            elif orientation_ == 'R':
                grid.horz_wall(x, y + (height_ - 1) // 2, width_)
                grid.vert_wall(x + width_ - 1, y, height_)
            return (x + width_, y)

    # ---------------------------------------------------------------------------------- #

    class TileI:
        """
        ┌────────────┐
        │WGWGWGWGWGWG│O
        │     WG     │
        │     WG     │
        │     WG     │
        │WGWGWGWGWGWG│
        └────────────┘
        """

        @staticmethod
        def place(grid : Grid, x : int, y : int, width : int = None, height : int = None) -> tuple:
            width_  = width  if width  is not None else DEFAULT_TILE_WIDTH
            height_ = height if height is not None else DEFAULT_TILE_HEIGHT
            grid.horz_wall(x, y, width_)
            grid.horz_wall(x, y + height_ - 1, width_)
            grid.vert_wall(round((x + width_) / 2), y, height_)
            return (x + width_, y)

    # ---------------------------------------------------------------------------------- #

    class TileTemplate:
        """
            <ASCII Representation>
            (special param explanation)
        """

        @staticmethod
        def place(grid : Grid, x : int, y : int, width : int = None, height : int = None) -> tuple:
            """
            Places the TileColumn at the given position and returns the new anchor.

            :param Grid grid: Object where wall should be placed within.
            :param int x: x - coordinate within grid where Tile should be placed.
            :param int y: y - coordinate within grid where Tile should be placed.
            :param int size: dimension of the tile (if it is scalable!)
            :return tuple new_anchor: The (x, y) coordinates where the next tile can be placed right next to it.
            """
            width_  = width  if width  is not None else DEFAULT_TILE_WIDTH
            height_ = height if height is not None else DEFAULT_TILE_HEIGHT
            pass
