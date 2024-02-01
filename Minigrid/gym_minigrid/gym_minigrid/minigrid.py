import hashlib
import math
from abc import abstractmethod
from enum import IntEnum
from typing import Any, Callable, Optional, Union, List

import gym
import numpy
import numpy as np
from gym import spaces
from gym.utils import seeding

from collections import deque
from copy import deepcopy
import colorsys


TASKREWARD = 20
PICKUPREWARD = TASKREWARD
DELIVERREWARD = TASKREWARD

# Size in pixels of a tile in the full-scale human view
from gym_minigrid.rendering import (
    downsample,
    fill_coords,
    highlight_img,
    point_in_circle,
    point_in_line,
    point_in_rect,
    point_in_triangle,
    rotate_fn,
)
from gym_minigrid.window import Window

#from gym_minigrid.Task import DoRandom, TaskManager, DoNothing, GoTo, PlaceObject, PickUpObject, Task

TILE_PIXELS = 32

# Map of color names to RGB values
COLORS = {
    "red": np.array([255, 0, 0]),
    "lightred": np.array([255, 165, 165]),
    "green": np.array([0, 255, 0]),
    "lightgreen": np.array([165, 255, 165]),
    "blue": np.array([0, 0, 255]),
    "lightblue": np.array([165, 165, 255]),
    "purple": np.array([112, 39, 195]),
    "lightpurple": np.array([202, 170, 238]),
    "yellow": np.array([255, 255, 0]),
    "grey": np.array([100, 100, 100]),
    "black": np.array([0, 0, 0]),
    "white": np.array([230, 230, 230]),
}

COLOR_NAMES = sorted(list(COLORS.keys()))

# Used to map colors to integers
COLOR_TO_IDX = {"red": 0, "green": 1, "blue": 2, "purple": 3, "yellow": 4, "grey": 5, "lightred": 6, "lightblue":7, "lightgreen":8, "lightpurple": 9, "black":10, "white": 11}

IDX_TO_COLOR = dict(zip(COLOR_TO_IDX.values(), COLOR_TO_IDX.keys()))

def color_bins(nr_bins):
    color_1 = np.array([222,237,182])
    color_2 = np.array([45,78,157])
    inter_values = [1./nr_bins * i for i in range(nr_bins+1)]
    color_values = []
    for value in inter_values:
        color_values.append((color_2 - color_1) * value + color_1)
    return color_values

def color_states(value, ranking, nr_bins):
    set_size = len(ranking)
    bin_size = math.floor(set_size/nr_bins)
    bin_bound_values = []
    for i in range(nr_bins):
        bin_bound_values.append(ranking[bin_size*i])
    bin_bound_values.append(1.)
    #if len(set(bin_bound_values))!= len(bin_bound_values):
    #   print("at least two bounds have the same value, you should probably reduce the bin size")

    color_values = color_bins(nr_bins)
    for i in range(nr_bins):
        if(value >= bin_bound_values[i] and value <=bin_bound_values[i+1]):
            if bin_bound_values[i]==bin_bound_values[i+1]:
                inter_value = 0
            else:
                inter_value = (value - bin_bound_values[i])/(bin_bound_values[i+1] - bin_bound_values[i])
            return (color_values[i+1] - color_values[i]) * inter_value + color_values[i]

def isSlippery(cell):
    if isinstance(cell, SlipperyNorth):
        return True
    elif isinstance(cell, SlipperySouth):
        return True
    elif isinstance(cell, SlipperyEast):
        return True
    elif isinstance(cell, SlipperyWest):
        return True
    else:
        return False

def isOneWay(cell):
    if isinstance(cell, OneWayNorth):
        return True
    elif isinstance(cell, OneWaySouth):
        return True
    elif isinstance(cell, OneWayEast):
        return True
    elif isinstance(cell, OneWayWest):
        return True
    else:
        return False

# Map of object type to integers
OBJECT_TO_IDX = {
    "unseen": 0,
    "empty": 1,
    "wall": 2,
    "floor": 3,
    "door": 4,
    "key": 5,
    "ball": 6,
    "box": 7,
    "goal": 8,
    "lava": 9,
    "agent": 10,
    "adversary": 11,
    "slipperynorth": 12,
    "slipperysouth": 13,
    "slipperyeast": 14,
    "slipperywest": 15,
    "onewaynorth": 20,
    "onewaysouth": 21,
    "onewayeast": 22,
    "onewaywest": 23,
    "heattile" : 16,
    "heattilereduced" : 17,
    "fixedtile" : 18,
    "rgbtile" : 19
}

IDX_TO_OBJECT = dict(zip(OBJECT_TO_IDX.values(), OBJECT_TO_IDX.keys()))

# Map of state names to integers
STATE_TO_IDX = {
    "open": 0,
    "closed": 1,
    "locked": 2,
}

# Map of agent direction indices to vectors
DIR_TO_VEC = [
    # Pointing right (positive X)
    np.array((1, 0)),
    # Down (positive Y)
    np.array((0, 1)),
    # Pointing left (negative X)
    np.array((-1, 0)),
    # Up (negative Y)
    np.array((0, -1)),
]


def check_if_no_duplicate(duplicate_list: list) -> bool:
    """Check if given list contains any duplicates"""
    return len(set(duplicate_list)) == len(duplicate_list)

def run_BFS_reward(grid, starting_position):
    max_distance = 0
    distances = [None] * grid.width * grid.height
    bfs_queue = deque([starting_position])
    traversed_cells = set()
    distances[starting_position[0] + grid.width * starting_position[1]] = 0
    while bfs_queue:
        current_cell = bfs_queue.pop()
        if current_cell in traversed_cells: continue
        traversed_cells.add(current_cell)
        current_distance = distances[current_cell[0] + grid.width * current_cell[1]]
        if current_distance > max_distance:
            max_distance = current_distance
        for neighbour in grid.get_neighbours(*current_cell):
            if neighbour in traversed_cells:
                continue
            bfs_queue.appendleft(neighbour)
            if distances[neighbour[0] + grid.width * neighbour[1]] is None:
                distances[neighbour[0] + grid.width * neighbour[1]] = current_distance + 1

    distances = [x if x else 0 for x in distances]
    #for i, x in enumerate(distances):
    #    if i % grid.width == 0:
    #        print("")
    #    if i is None:
    #        print("  ")
    #        continue
    #    else:
    #        print("{:0>4},".format(x), end="")
    #print("")
    return [-x/max_distance for x in distances]


class MissionSpace(spaces.Space[str]):
    r"""A space representing a mission for the Gym-Minigrid environments.
    The space allows generating random mission strings constructed with an input placeholder list.
    Example Usage::
        >>> observation_space = MissionSpace(mission_func=lambda color: f"Get the {color} ball.",
                                                ordered_placeholders=[["green", "blue"]])
        >>> observation_space.sample()
            "Get the green ball."
        >>> observation_space = MissionSpace(mission_func=lambda : "Get the ball.".,
                                                ordered_placeholders=None)
        >>> observation_space.sample()
            "Get the ball."
    """

    def __init__(
        self,
        mission_func: Callable[..., str],
        ordered_placeholders: Optional["list[list[str]]"] = None,
        seed: Optional[Union[int, seeding.RandomNumberGenerator]] = None,
    ):
        r"""Constructor of :class:`MissionSpace` space.

        Args:
            mission_func (lambda _placeholders(str): _mission(str)): Function that generates a mission string from random placeholders.
            ordered_placeholders (Optional["list[list[str]]"]): List of lists of placeholders ordered in placing order in the mission function mission_func.
            seed: seed: The seed for sampling from the space.
        """
        # Check that the ordered placeholders and mission function are well defined.
        if ordered_placeholders is not None:
            assert (
                len(ordered_placeholders) == mission_func.__code__.co_argcount
            ), f"The number of placeholders {len(ordered_placeholders)} is different from the number of parameters in the mission function {mission_func.__code__.co_argcount}."
            for placeholder_list in ordered_placeholders:
                assert check_if_no_duplicate(
                    placeholder_list
                ), "Make sure that the placeholders don't have any duplicate values."
        else:
            assert (
                mission_func.__code__.co_argcount == 0
            ), f"If the ordered placeholders are {ordered_placeholders}, the mission function shouldn't have any parameters."

        self.ordered_placeholders = ordered_placeholders
        self.mission_func = mission_func

        super().__init__(dtype=str, seed=seed)

        # Check that mission_func returns a string
        sampled_mission = self.sample()
        assert isinstance(
            sampled_mission, str
        ), f"mission_func must return type str not {type(sampled_mission)}"

    def sample(self) -> str:
        """Sample a random mission string."""
        if self.ordered_placeholders is not None:
            placeholders = []
            for rand_var_list in self.ordered_placeholders:
                idx = self.np_random.integers(0, len(rand_var_list))

                placeholders.append(rand_var_list[idx])

            return self.mission_func(*placeholders)
        else:
            return self.mission_func()

    def contains(self, x: Any) -> bool:
        """Return boolean specifying if x is a valid member of this space."""
        # Store a list of all the placeholders from self.ordered_placeholders that appear in x
        if self.ordered_placeholders is not None:
            check_placeholder_list = []
            for placeholder_list in self.ordered_placeholders:
                for placeholder in placeholder_list:
                    if placeholder in x:
                        check_placeholder_list.append(placeholder)

            # Remove duplicates from the list
            check_placeholder_list = list(set(check_placeholder_list))

            start_id_placeholder = []
            end_id_placeholder = []
            # Get the starting and ending id of the identified placeholders with possible duplicates
            new_check_placeholder_list = []
            for placeholder in check_placeholder_list:
                new_start_id_placeholder = [
                    i for i in range(len(x)) if x.startswith(placeholder, i)
                ]
                new_check_placeholder_list += [placeholder] * len(
                    new_start_id_placeholder
                )
                end_id_placeholder += [
                    start_id + len(placeholder) - 1
                    for start_id in new_start_id_placeholder
                ]
                start_id_placeholder += new_start_id_placeholder

            # Order by starting id the placeholders
            ordered_placeholder_list = sorted(
                zip(
                    start_id_placeholder, end_id_placeholder, new_check_placeholder_list
                )
            )

            # Check for repeated placeholders contained in each other
            remove_placeholder_id = []
            for i, placeholder_1 in enumerate(ordered_placeholder_list):
                starting_id = i + 1
                for j, placeholder_2 in enumerate(
                    ordered_placeholder_list[starting_id:]
                ):
                    # Check if place holder ids overlap and keep the longest
                    if max(placeholder_1[0], placeholder_2[0]) < min(
                        placeholder_1[1], placeholder_2[1]
                    ):
                        remove_placeholder = min(
                            placeholder_1[2], placeholder_2[2], key=len
                        )
                        if remove_placeholder == placeholder_1[2]:
                            remove_placeholder_id.append(i)
                        else:
                            remove_placeholder_id.append(i + j + 1)
            for id in remove_placeholder_id:
                del ordered_placeholder_list[id]

            final_placeholders = [
                placeholder[2] for placeholder in ordered_placeholder_list
            ]

            # Check that the identified final placeholders are in the same order as the original placeholders.
            for orered_placeholder, final_placeholder in zip(
                self.ordered_placeholders, final_placeholders
            ):
                if final_placeholder in orered_placeholder:
                    continue
                else:
                    return False
            try:
                mission_string_with_placeholders = self.mission_func(
                    *final_placeholders
                )
            except Exception as e:
                print(
                    f"{x} is not contained in MissionSpace due to the following exception: {e}"
                )
                return False

            return bool(mission_string_with_placeholders == x)

        else:
            return bool(self.mission_func() == x)

    def __repr__(self) -> str:
        """Gives a string representation of this space."""
        return f"MissionSpace({self.mission_func}, {self.ordered_placeholders})"

    def __eq__(self, other) -> bool:
        """Check whether ``other`` is equivalent to this instance."""
        if isinstance(other, MissionSpace):

            # Check that place holder lists are the same
            if self.ordered_placeholders is not None:
                # Check length
                if (len(self.order_placeholder) == len(other.order_placeholder)) and (
                    all(
                        set(i) == set(j)
                        for i, j in zip(self.order_placeholder, other.order_placeholder)
                    )
                ):
                    # Check mission string is the same with dummy space placeholders
                    test_placeholders = [""] * len(self.order_placeholder)
                    mission = self.mission_func(*test_placeholders)
                    other_mission = other.mission_func(*test_placeholders)
                    return mission == other_mission
            else:

                # Check that other is also None
                if other.ordered_placeholders is None:

                    # Check mission string is the same
                    mission = self.mission_func()
                    other_mission = other.mission_func()
                    return mission == other_mission

        # If none of the statements above return then False
        return False


class WorldObj:
    """
    Base class for grid world objects
    """

    def __init__(self, type, color):
        assert type in OBJECT_TO_IDX, type
        assert color in COLOR_TO_IDX, color
        self.type = type
        self.color = color
        self.contains = None

        # Initial position of the object
        self.init_pos = None

        # Current position of the object
        self.cur_pos = None

    def can_overlap(self):
        """Can the agent overlap with this?"""
        return False

    def can_pickup(self):
        """Can the agent pick this up?"""
        return False

    def can_contain(self):
        """Can this contain another object?"""
        return False

    def see_behind(self):
        """Can the agent see behind this object?"""
        return True

    def toggle(self, env, pos):
        """Method to trigger/toggle an action this object performs"""
        return False

    def encode(self):
        """Encode the a description of this object as a 3-tuple of integers"""
        return (OBJECT_TO_IDX[self.type], COLOR_TO_IDX[self.color], 0)

    @staticmethod
    def decode(type_idx, color_idx, state):
        """Create an object from a 3-tuple state description"""

        obj_type = IDX_TO_OBJECT[type_idx]
        color = IDX_TO_COLOR[color_idx]

        if obj_type == "empty" or obj_type == "unseen":
            return None

        # State, 0: open, 1: closed, 2: locked
        is_open = state == 0
        is_locked = state == 2

        if obj_type == "wall":
            v = Wall(color)
        elif obj_type == "floor":
            v = Floor(color)
        elif obj_type == "ball":
            v = Ball(color)
        elif obj_type == "key":
            v = Key(color)
        elif obj_type == "box":
            v = Box(color)
        elif obj_type == "door":
            v = Door(color, is_open, is_locked)
        elif obj_type == "goal":
            v = Goal()
        elif obj_type == "lava":
            v = Lava()
        elif obj_type == "slipperynorth":
            v = SlipperyNorth(color)
        elif obj_type == "slipperysouth":
            v = SlipperySouth(color)
        elif obj_type == "slipperywest":
            v = SlipperyWest(color)
        elif obj_type == "slipperyeast":
            v = SlipperyEast(color)
        elif obj_type == "onewaynorth":
            v = OneWayNorth(color)
        elif obj_type == "onewaysouth":
            v = OneWaySouth(color)
        elif obj_type == "onewayeast":
            v = OneWayEast(color)
        elif obj_type == "onewaywest":
            v = OneWayWest(color)
        elif obj_type == "heattile":
            v = HeatMapTile(color)
        elif obj_type == "fixedtile":
            v = FixedMapTile(color)
        elif obj_type == "heattilereduced":
            v = HeatMapTileReduced(color)
        elif obj_type == "rgbtile":
            v = RGBTile()
        else:
            assert False, "unknown object type in decode '%s'" % obj_type

        return v

    def render(self, r):
        """Draw this object with the given renderer"""
        raise NotImplementedError


class Goal(WorldObj):
    def __init__(self):
        super().__init__("goal", "green")

    def can_overlap(self):
        return True

    def render(self, img):
        fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.color])


class Floor(WorldObj):
    """
    Colored floor tile the agent can walk over
    """

    def __init__(self, color="blue"):
        super().__init__("floor", color)

    def can_overlap(self):
        return True

    def render(self, img):
        # Give the floor a pale color
        if self.color != "black":
            color = COLORS[self.color] / 2
        else:
            color = COLORS[self.color]
        fill_coords(img, point_in_rect(0.031, 1, 0.031, 1), color)

    def can_contain(self):
        return True


class Lava(WorldObj):
    def __init__(self):
        super().__init__("lava", "red")

    def can_overlap(self):
        return True

    def render(self, img):
        c = (255, 128, 0)

        # Background color
        fill_coords(img, point_in_rect(0, 1, 0, 1), c)

        # Little waves
        for i in range(3):
            ylo = 0.3 + 0.2 * i
            yhi = 0.4 + 0.2 * i
            fill_coords(img, point_in_line(0.1, ylo, 0.3, yhi, r=0.03), (0, 0, 0))
            fill_coords(img, point_in_line(0.3, yhi, 0.5, ylo, r=0.03), (0, 0, 0))
            fill_coords(img, point_in_line(0.5, ylo, 0.7, yhi, r=0.03), (0, 0, 0))
            fill_coords(img, point_in_line(0.7, yhi, 0.9, ylo, r=0.03), (0, 0, 0))


class Wall(WorldObj):
    def __init__(self, color="grey"):
        super().__init__("wall", color)

    def see_behind(self):
        return False

    def render(self, img):
        fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.color])


class Door(WorldObj):
    def __init__(self, color, is_open=False, is_locked=False):
        super().__init__("door", color)
        self.is_open = is_open
        self.is_locked = is_locked

    def can_overlap(self):
        """The agent can only walk over this cell when the door is open"""
        return self.is_open

    def see_behind(self):
        return self.is_open

    def toggle(self, env, pos):
        # If the player has the right key to open the door
        if self.is_locked:
            if isinstance(env.carrying, Key) and env.carrying.color == self.color:
                self.is_locked = False
                self.is_open = True
                return True
            return False

        self.is_open = not self.is_open
        return True

    def encode(self):
        """Encode the a description of this object as a 3-tuple of integers"""

        # State, 0: open, 1: closed, 2: locked
        if self.is_open:
            state = 0
        elif self.is_locked:
            state = 2
        # if door is closed and unlocked
        elif not self.is_open:
            state = 1
        else:
            raise ValueError(
                f"There is no possible state encoding for the state:\n -Door Open: {self.is_open}\n -Door Closed: {not self.is_open}\n -Door Locked: {self.is_locked}"
            )

        return (OBJECT_TO_IDX[self.type], COLOR_TO_IDX[self.color], state)

    def render(self, img):
        c = COLORS[self.color]

        if self.is_open:
            fill_coords(img, point_in_rect(0.88, 1.00, 0.00, 1.00), c)
            fill_coords(img, point_in_rect(0.92, 0.96, 0.04, 0.96), (0, 0, 0))
            return

        # Door frame and door
        if self.is_locked:
            fill_coords(img, point_in_rect(0.00, 1.00, 0.00, 1.00), c)
            fill_coords(img, point_in_rect(0.06, 0.94, 0.06, 0.94), 0.45 * np.array(c))

            # Draw key slot
            fill_coords(img, point_in_rect(0.52, 0.75, 0.50, 0.56), c)
        else:
            fill_coords(img, point_in_rect(0.00, 1.00, 0.00, 1.00), c)
            fill_coords(img, point_in_rect(0.04, 0.96, 0.04, 0.96), (0, 0, 0))
            fill_coords(img, point_in_rect(0.08, 0.92, 0.08, 0.92), c)
            fill_coords(img, point_in_rect(0.12, 0.88, 0.12, 0.88), (0, 0, 0))

            # Draw door handle
            fill_coords(img, point_in_circle(cx=0.75, cy=0.50, r=0.08), c)


class Key(WorldObj):
    def __init__(self, color="blue"):
        super().__init__("key", color)

    def can_pickup(self):
        return True

    def render(self, img):
        c = COLORS[self.color]

        # Vertical quad
        fill_coords(img, point_in_rect(0.50, 0.63, 0.31, 0.88), c)

        # Teeth
        fill_coords(img, point_in_rect(0.38, 0.50, 0.59, 0.66), c)
        fill_coords(img, point_in_rect(0.38, 0.50, 0.81, 0.88), c)

        # Ring
        fill_coords(img, point_in_circle(cx=0.56, cy=0.28, r=0.190), c)
        fill_coords(img, point_in_circle(cx=0.56, cy=0.28, r=0.064), (0, 0, 0))


class Ball(WorldObj):
    def __init__(self, color="blue"):
        super().__init__("ball", color)

    def can_pickup(self):
        return True

    def render(self, img):
        fill_coords(img, point_in_circle(0.5, 0.5, 0.31), COLORS[self.color])


class Box(WorldObj):
    def __init__(self, color, contains=None):
        super().__init__("box", color)
        self.contains = contains
        self.picked_up = False
        self.placed_at_destination = False

    def can_pickup(self):
        return not self.placed_at_destination

    def can_overlap(self):
        return True

    def render(self, img):
        c = COLORS[self.color]

        # Outline
        fill_coords(img, point_in_rect(0.12, 0.88, 0.12, 0.88), c)
        fill_coords(img, point_in_rect(0.18, 0.82, 0.18, 0.82), (0, 0, 0))

        # Horizontal slit
        fill_coords(img, point_in_rect(0.16, 0.84, 0.47, 0.53), c)

    def toggle(self, env, pos):
        # Replace the box by its contents
        return False # We dont want to destroy boxes, This is local change
        #env.grid.set(pos[0], pos[1], self.contains)
        #return True

class SlipperyNorth(WorldObj):
    def __init__(self, color: str = "blue"):
        super().__init__("slipperynorth", color)
        self.probabilities_forward = [0.0, 1./9, 2./9, 0.0, -50, -50, 0.0, 1./9, 2./9]
        self.probabilities_turn = [0.0, 0.0, 1./9, 0.0, -50, 1./9, 0.0, 0.0, 1./9]
        self.offset = (0,1)

    def can_overlap(self):
        return True

    def render(self, img):
        c = (100, 100, 200)
        fill_coords(img, point_in_rect(0, 1, 0, 1), c)
        for i in range(6):
            ylo = 0.1  + 0.15 * i
            yhi = 0.20 + 0.15 * i
            fill_coords(img, point_in_line(0.1, ylo, 0.3, yhi, r=0.03), (0, 0, 0))
            fill_coords(img, point_in_line(0.3, yhi, 0.5, ylo, r=0.03), (0, 0, 0))
            fill_coords(img, point_in_line(0.5, ylo, 0.7, yhi, r=0.03), (0, 0, 0))
            fill_coords(img, point_in_line(0.7, yhi, 0.9, ylo, r=0.03), (0, 0, 0))


class SlipperySouth(WorldObj):
    def __init__(self, color: str = "blue"):
        super().__init__("slipperysouth", color)
        self.probabilities_forward = [2./9, 1./9, 0.0, -50, -50, 0.0, 2./9, 1./9, 0.0]
        self.probabilities_turn = [1./9, 0.0, 0.0, 1./9, -50, 0.0, 1./9, 0.0, 0.0]
        self.offset = (0,-1)
    def can_overlap(self):
        return True

    def render(self, img):
        c = (100, 100, 200)
        fill_coords(img, point_in_rect(0, 1, 0, 1), c)
        for i in range(6):
            ylo = 0.1  + 0.15 * i
            yhi = 0.20 + 0.15 * i
            fill_coords(img, rotate_fn(point_in_line(0.1, ylo, 0.3, yhi, r=0.03), cx=0.5, cy=0.5, theta=math.pi), (0, 0, 0))
            fill_coords(img, rotate_fn(point_in_line(0.3, yhi, 0.5, ylo, r=0.03), cx=0.5, cy=0.5, theta=math.pi), (0, 0, 0))
            fill_coords(img, rotate_fn(point_in_line(0.5, ylo, 0.7, yhi, r=0.03), cx=0.5, cy=0.5, theta=math.pi), (0, 0, 0))
            fill_coords(img, rotate_fn(point_in_line(0.7, yhi, 0.9, ylo, r=0.03), cx=0.5, cy=0.5, theta=math.pi), (0, 0, 0))

class SlipperyEast(WorldObj):
    def __init__(self, color: str = "blue"):
        super().__init__("slipperyeast", color)
        self.probabilities_forward = [2./9, -50, 2./9, 1./9., -50, 1./9, 0.0, 0.0, 0.0]
        self.probabilities_turn = [1./9, 1./9, 1./9, 0.0, -50, 0.0, 0.0, 0.0, 0.0]
        self.offset = (-1,0)

    def can_overlap(self):
        return True

    def render(self, img):
        c = (100, 100, 200)
        fill_coords(img, point_in_rect(0, 1, 0, 1), c)
        for i in range(6):
            ylo = 0.1  + 0.15 * i
            yhi = 0.20 + 0.15 * i
            fill_coords(img, rotate_fn(point_in_line(0.1, ylo, 0.3, yhi, r=0.03), cx=0.5, cy=0.5, theta=0.5 * math.pi), (0, 0, 0))
            fill_coords(img, rotate_fn(point_in_line(0.3, yhi, 0.5, ylo, r=0.03), cx=0.5, cy=0.5, theta=0.5 * math.pi), (0, 0, 0))
            fill_coords(img, rotate_fn(point_in_line(0.5, ylo, 0.7, yhi, r=0.03), cx=0.5, cy=0.5, theta=0.5 * math.pi), (0, 0, 0))
            fill_coords(img, rotate_fn(point_in_line(0.7, yhi, 0.9, ylo, r=0.03), cx=0.5, cy=0.5, theta=0.5 * math.pi), (0, 0, 0))

class SlipperyWest(WorldObj):
    def __init__(self, color: str = "blue"):
        super().__init__("slipperywest", color)
        self.probabilities_forward = [0.0, 0.0, 0.0, 1./9., -50, 1./9, 2./9, -50, 2./9]
        self.probabilities_turn = [0.0, 0.0, 0.0, 0.0, -50, 0.0, 1./9, 1./9, 1./9]
        self.offset = (1,0)
    def can_overlap(self):
        return True

    def render(self, img):
        c = (100, 100, 200)
        fill_coords(img, point_in_rect(0, 1, 0, 1), c)
        for i in range(6):
            ylo = 0.1  + 0.15 * i
            yhi = 0.20 + 0.15 * i
            fill_coords(img, rotate_fn(point_in_line(0.1, ylo, 0.3, yhi, r=0.03), cx=0.5, cy=0.5, theta=-0.5 * math.pi), (0, 0, 0))
            fill_coords(img, rotate_fn(point_in_line(0.3, yhi, 0.5, ylo, r=0.03), cx=0.5, cy=0.5, theta=-0.5 * math.pi), (0, 0, 0))
            fill_coords(img, rotate_fn(point_in_line(0.5, ylo, 0.7, yhi, r=0.03), cx=0.5, cy=0.5, theta=-0.5 * math.pi), (0, 0, 0))
            fill_coords(img, rotate_fn(point_in_line(0.7, yhi, 0.9, ylo, r=0.03), cx=0.5, cy=0.5, theta=-0.5 * math.pi), (0, 0, 0))

class OneWayNorth(WorldObj):
    def __init__(self, color: str = "white"):
        super().__init__("onewaynorth", color)
        self.direction = 1
        self.direction_vector = (0,1)

    def can_overlap(self):
        return True

    def render(self, img):
        c = (230, 230, 230)
        fill_coords(img, point_in_rect(0, 1, 0, 1), c)
        ylo = 0.35
        yhi = 0.75
        fill_coords(img, point_in_line(0.1, ylo, 0.5, yhi, r=0.03), (0, 0, 0))
        fill_coords(img, point_in_line(0.5, yhi, 0.9, ylo, r=0.03), (0, 0, 0))

class OneWaySouth(WorldObj):
    def __init__(self, color: str = "white"):
        super().__init__("onewaysouth", color)
        self.direction = 3
        self.direction_vector = (0,-1)

    def can_overlap(self):
        return True

    def render(self, img):
        c = (230, 230, 230)
        fill_coords(img, point_in_rect(0, 1, 0, 1), c)
        ylo = 0.35
        yhi = 0.75
        fill_coords(img, rotate_fn(point_in_line(0.1, ylo, 0.5, yhi, r=0.03), cx=0.5, cy=0.5, theta=math.pi), (0, 0, 0))
        fill_coords(img, rotate_fn(point_in_line(0.5, yhi, 0.9, ylo, r=0.03), cx=0.5, cy=0.5, theta=math.pi), (0, 0, 0))

class OneWayEast(WorldObj):
    def __init__(self, color: str = "white"):
        super().__init__("onewayeast", color)
        self.direction = 2
        self.direction_vector = (-1,0)

    def can_overlap(self):
        return True

    def render(self, img):
        c = (230, 230, 230)
        fill_coords(img, point_in_rect(0, 1, 0, 1), c)
        ylo = 0.35
        yhi = 0.75
        fill_coords(img, rotate_fn(point_in_line(0.1, ylo, 0.5, yhi, r=0.03), cx=0.5, cy=0.5, theta=0.5 * math.pi), (0, 0, 0))
        fill_coords(img, rotate_fn(point_in_line(0.5, yhi, 0.9, ylo, r=0.03), cx=0.5, cy=0.5, theta=0.5 * math.pi), (0, 0, 0))

class OneWayWest(WorldObj):
    def __init__(self, color: str = "white"):
        super().__init__("onewaywest", color)
        self.direction = 0
        self.direction_vector = (1,0)

    def can_overlap(self):
        return True

    def render(self, img):
        c = (230, 230, 230)
        fill_coords(img, point_in_rect(0, 1, 0, 1), c)
        ylo = 0.35
        yhi = 0.75
        fill_coords(img, rotate_fn(point_in_line(0.1, ylo, 0.5, yhi, r=0.03), cx=0.5, cy=0.5, theta=-0.5 * math.pi), (0, 0, 0))
        fill_coords(img, rotate_fn(point_in_line(0.5, yhi, 0.9, ylo, r=0.03), cx=0.5, cy=0.5, theta=-0.5 * math.pi), (0, 0, 0))



tri_22 = point_in_triangle(
    (1.0, 0.0),
    (0.0, 0.0),
    (0.5, 0.5),
)
tri_33 = point_in_triangle(
    (0.0, 0.0),
    (0.0, 1.0),
    (0.5, 0.5),
)
tri_00 = point_in_triangle(
    (0.0, 1.0),
    (1.0, 1.0),
    (0.5, 0.5),
)
tri_11 = point_in_triangle(
    (1.0, 1.0),
    (1.0, 0.0),
    (0.5, 0.5),
)

def tri_mask(width, heigth):
    list22 = []
    list33 = []
    list00 = []
    list11 = []
    for x in range(width):
        for y in range(heigth):
            yf = (y + 0.5) / heigth
            xf = (x + 0.5) / width
            if tri_22(xf,yf):
                list22.append((x,y))
            elif tri_33(xf,yf):
                list33.append((x,y))
            elif tri_00(xf,yf):
                list00.append((x,y))
            elif tri_11(xf,yf):
                list11.append((x,y))
    return np.array(list22), np.array(list33), np.array(list00), np.array(list11)

list22, list33, list00, list11 = tri_mask(96, 96)

class HeatMapTile(WorldObj):
    def __init__(self, tile=dict(), ranking = [], nr_bins=5, color="blue"):
        super().__init__("heattile", color)
        self.tile_values = tile
        self.ranking = ranking
        self.nr_bins = nr_bins

    def can_overlap(self):
        return True

    def can_contain(self):
        return True

    def encode(self):
        """Encode the a description of this object as a 3-tuple of integers"""
        return (OBJECT_TO_IDX[self.type], COLOR_TO_IDX[self.color], 0)

    def render(self, img):
        img[tuple(list22.T)] = color_states(self.tile_values[2], self.ranking, self.nr_bins)
        img[tuple(list33.T)] = color_states(self.tile_values[3], self.ranking, self.nr_bins)
        img[tuple(list00.T)] = color_states(self.tile_values[0], self.ranking, self.nr_bins)
        img[tuple(list11.T)] = color_states(self.tile_values[1], self.ranking, self.nr_bins)

class FixedMapTile(WorldObj):
    def __init__(self, tile=dict(), color="blue"):
        super().__init__("fixedtile", "blue")
        self.tile_values = tile

    def encode(self):
        """Encode the a description of this object as a 3-tuple of integers"""
        return (OBJECT_TO_IDX[self.type], COLOR_TO_IDX[self.color], 0)

    def render(self, img):
        img[tuple(list22.T)] = self.tile_values[2]
        img[tuple(list33.T)] = self.tile_values[3]
        img[tuple(list00.T)] = self.tile_values[0]
        img[tuple(list11.T)] = self.tile_values[1]

class RGBTile(WorldObj):
    def __init__(self, color):
        super().__init__("rgbtile", "blue")
        self.color = color

    def can_overlap(self):
        return True

    def can_contain(self):
        return True

    def encode(self):
        """Encode the a description of this object as a 3-tuple of integers"""
        return (OBJECT_TO_IDX[self.type], COLOR_TO_IDX["blue"], 0)


    def render(self, img):
        fill_coords(img, point_in_rect(0, 1, 0, 1), self.color)


class HeatMapTileReduced(WorldObj):
    def __init__(self, ranking_value=0, ranking=[], nr_bins=5, color="blue"):
        super().__init__("heattilereduced", color)
        self.ranking = ranking
        self.nr_bins = nr_bins
        self.color = color_states(ranking_value, self.ranking, self.nr_bins)

    def can_overlap(self):
        return True

    def can_contain(self):
        return True

    def encode(self):
        """Encode the a description of this object as a 3-tuple of integers"""
        return (OBJECT_TO_IDX[self.type], COLOR_TO_IDX["blue"], 0)


    def render(self, img):
        fill_coords(img, point_in_rect(0, 1, 0, 1), self.color)


class Grid:
    """
    Represent a grid and operations on it
    """

    # Static cache of pre-renderer tiles
    tile_cache = {}

    def __init__(self, width, height):
        assert width >= 3
        assert height >= 3

        self.width = width
        self.height = height

        self.grid = [None] * width * height
        self.background = [None] * width * height

    def __contains__(self, key):
        if isinstance(key, WorldObj):
            for e in self.grid:
                if e is key:
                    return True
        elif isinstance(key, tuple):
            for e in self.grid:
                if e is None:
                    continue
                if (e.color, e.type) == key:
                    return True
                if key[0] is None and key[1] == e.type:
                    return True
        return False

    def __eq__(self, other):
        grid1 = self.encode()
        grid2 = other.encode()
        return np.array_equal(grid2, grid1)

    def __ne__(self, other):
        return not self == other

    def copy(self):
        return deepcopy(self)

    def set(self, i, j, v):
        assert i >= 0 and i < self.width
        assert j >= 0 and j < self.height
        self.grid[j * self.width + i] = v

    def get(self, i, j):
        assert i >= 0 and i < self.width
        assert j >= 0 and j < self.height
        return self.grid[j * self.width + i]

    def set_background(self, i, j, v):
        assert i >= 0 and i < self.width
        assert j >= 0 and j < self.height
        self.background[j * self.width + i] = v

    def get_background(self, i, j):
        assert i >= 0 and i < self.width
        assert j >= 0 and j < self.height
        return self.background[j * self.width + i]

    def horz_wall(self, x, y, length=None, obj_type=Wall):
        if length is None:
            length = self.width - x
        for i in range(0, length):
            self.set(x + i, y, obj_type())

    def vert_wall(self, x, y, length=None, obj_type=Wall):
        if length is None:
            length = self.height - y
        for j in range(0, length):
            self.set(x, y + j, obj_type())

    def wall_rect(self, x, y, w, h, obj_type=Wall):
        self.horz_wall(x, y, w, obj_type=obj_type)
        self.horz_wall(x, y + h - 1, w, obj_type=obj_type)
        self.vert_wall(x, y, h, obj_type=obj_type)
        self.vert_wall(x + w - 1, y, h, obj_type=obj_type)

    def rotate_left(self):
        """
        Rotate the grid to the left (counter-clockwise)
        """

        grid = Grid(self.height, self.width)

        for i in range(self.width):
            for j in range(self.height):
                v = self.get(i, j)
                grid.set(j, grid.height - 1 - i, v)

        return grid

    def slice(self, topX, topY, width, height):
        """
        Get a subset of the grid
        """

        grid = Grid(width, height)

        for j in range(0, height):
            for i in range(0, width):
                x = topX + i
                y = topY + j

                if x >= 0 and x < self.width and y >= 0 and y < self.height:
                    v = self.get(x, y)
                else:
                    v = Wall()

                grid.set(i, j, v)

        return grid

    @classmethod
    def render_tile(
        cls, obj, background, agent_dir=None, highlight=False, tile_size=TILE_PIXELS, subdivs=3, carrying=None, cache=True
    ):
        """
        Render a tile and cache the result
        """
        # Hash map lookup key for the cache
        # this prevents re-rendering tiles
        key = (agent_dir, highlight, tile_size)
        if obj:
            key = key + obj.encode()
        if background:
            key = key + background.encode()
        if carrying:
            key = key + carrying.encode()

        if key in cls.tile_cache and cache:
            return cls.tile_cache[key]

        img = np.zeros(
            shape=(tile_size * subdivs, tile_size * subdivs, 3), dtype=np.uint8
        )

        # Draw the grid lines (top and left edges)
        fill_coords(img, point_in_rect(0, 0.031, 0, 1), (100, 100, 100))
        fill_coords(img, point_in_rect(0, 1, 0, 0.031), (100, 100, 100))

        if background is not None:
            background.render(img)

        if obj is not None:
            obj.render(img)

        # Overlay the agent on top
        if agent_dir is not None:
            tri_fn = point_in_triangle(
                (0.12, 0.19),
                (0.87, 0.50),
                (0.12, 0.81),
            )

            # Rotate the agent based on its direction
            tri_fn = rotate_fn(tri_fn, cx=0.5, cy=0.5, theta=0.5 * math.pi * agent_dir)
            fill_coords(img, tri_fn, (255, 0, 0))

            if carrying:
                width = 0.15
                tri_fn = point_in_triangle(
                    (0.12+width, 0.19+width),
                    (0.87-width, 0.50),
                    (0.12+width, 0.81-width),
                )

                # Rotate the agent based on its direction
                tri_fn = rotate_fn(tri_fn, cx=0.5, cy=0.5, theta=0.5 * math.pi * agent_dir)
                fill_coords(img, tri_fn, (255, 255, 255))

        # Highlight the cell if needed
        if highlight:
            highlight_img(img)

        # Downsample the image to perform supersampling/anti-aliasing
        img = downsample(img, subdivs)

        # Cache the rendered tile
        cls.tile_cache[key] = img

        return img

    def render(self, tile_size, agent_pos, agent_dir=None, highlight_mask=None, carrying=None, cache=True):
        """
        Render this grid at a given scale
        :param r: target renderer object
        :param tile_size: tile size in pixels
        """
        if highlight_mask is None:
            highlight_mask = np.zeros(shape=(self.width, self.height), dtype=bool)

        # Compute the total grid size
        width_px = self.width * tile_size
        height_px = self.height * tile_size

        img = np.zeros(shape=(height_px, width_px, 3), dtype=np.uint8)

        # Render the grid
        for j in range(0, self.height):
            for i in range(0, self.width):
                cell = self.get(i, j)
                background = self.get_background(i,j)
                agent_here = np.array_equal(agent_pos, (i, j))
                tile_img = Grid.render_tile(
                    cell,
                    background,
                    agent_dir=agent_dir if agent_here else None,
                    highlight=highlight_mask[i, j],
                    tile_size=tile_size,
                    carrying=carrying if agent_here else None,
                    cache=cache
                )

                ymin = j * tile_size
                ymax = (j + 1) * tile_size
                xmin = i * tile_size
                xmax = (i + 1) * tile_size
                img[ymin:ymax, xmin:xmax, :] = tile_img

        return img

    def encode(self, vis_mask=None):
        """
        Produce a compact numpy encoding of the grid
        """

        if vis_mask is None:
            vis_mask = np.ones((self.width, self.height), dtype=bool)

        array = np.zeros((self.width, self.height, 3), dtype="uint8")

        for i in range(self.width):
            for j in range(self.height):
                if vis_mask[i, j]:
                    v = self.get(i, j)

                    if v is None:
                        array[i, j, 0] = OBJECT_TO_IDX["empty"]
                        array[i, j, 1] = 0
                        array[i, j, 2] = 0

                    else:
                        array[i, j, :] = v.encode()

        return array

    @staticmethod
    def decode(array):
        """
        Decode an array grid encoding back into a grid
        """

        width, height, channels = array.shape
        assert channels == 3

        vis_mask = np.ones(shape=(width, height), dtype=bool)

        grid = Grid(width, height)
        for i in range(width):
            for j in range(height):
                type_idx, color_idx, state = array[i, j]
                v = WorldObj.decode(type_idx, color_idx, state)
                grid.set(i, j, v)
                vis_mask[i, j] = type_idx != OBJECT_TO_IDX["unseen"]

        return grid, vis_mask

    def process_vis(self, agent_pos):
        mask = np.zeros(shape=(self.width, self.height), dtype=bool)

        mask[agent_pos[0], agent_pos[1]] = True

        for j in reversed(range(0, self.height)):
            for i in range(0, self.width - 1):
                if not mask[i, j]:
                    continue

                cell = self.get(i, j)
                if cell and not cell.see_behind():
                    continue

                mask[i + 1, j] = True
                if j > 0:
                    mask[i + 1, j - 1] = True
                    mask[i, j - 1] = True

            for i in reversed(range(1, self.width)):
                if not mask[i, j]:
                    continue

                cell = self.get(i, j)
                if cell and not cell.see_behind():
                    continue

                mask[i - 1, j] = True
                if j > 0:
                    mask[i - 1, j - 1] = True
                    mask[i, j - 1] = True

        for j in range(0, self.height):
            for i in range(0, self.width):
                if not mask[i, j]:
                    self.set(i, j, None)

        return mask

    def get_neighbours(self, i, j):
        neighbours = list()
        potential_neighbours = [(i-1,j), (i,j+1), (i+1,j), (i,j-1)]
        for n in potential_neighbours:
            cell = self.get(*n)
            if cell is None or (cell.can_overlap() and not isinstance(cell, Lava)):
                neighbours.append(n)
        return neighbours


class MiniGridEnv(gym.Env):
    """
    2D grid world game environment
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 10,
    }

    # Enumeration of possible actions
    class Actions(IntEnum):
        # Turn left, turn right, move forward
        left = 0
        right = 1
        forward = 2
        # Pick up an object
        pickup = 3
        # Drop an object
        drop = 4
        # Toggle/activate an object
        toggle = 5
        # Done completing task
        done = 6

    class MovementOnlyActions(IntEnum):
        # Turn left, turn right, move forward
        left = 0
        right = 1
        forward = 2
        # Pick up an object

    def __init__(
        self,
        mission_space: MissionSpace,
        grid_size: int = None,
        width: int = None,
        height: int = None,
        max_steps: int = 100,
        see_through_walls: bool = False,
        agent_view_size: int = 7,
        render_mode: Optional[str] = None,
        highlight: bool = True,
        tile_size: int = TILE_PIXELS,
        agent_pov: bool = False,
    ):
        # Initialize mission
        self.mission = mission_space.sample()
        self.adversaries = list()
        self.background_boxes = dict()
        self.bfs_reward = list()

        self.violation_deque = deque(maxlen=12)
        # Can't set both grid_size and width/height
        if grid_size:
            assert width is None and height is None
            width = grid_size
            height = grid_size

        # Action enumeration for this environment
        self.actions = MiniGridEnv.Actions
        #self.actions = MiniGridEnv.MovementOnlyActions

        # Actions are discrete integer values
        self.action_space = spaces.Discrete(len(self.actions))

        # Number of cells (width and height) in the agent view
        assert agent_view_size % 2 == 1
        assert agent_view_size >= 3
        self.agent_view_size = agent_view_size

        # Observations are dictionaries containing an
        # encoding of the grid and a textual 'mission' string
        image_observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.agent_view_size, self.agent_view_size, 3),
            dtype="uint8",
        )
        self.observation_space = spaces.Dict(
            {
                "image": image_observation_space,
                "direction": spaces.Discrete(4),
                "mission": mission_space,
            }
        )

        # Range of possible rewards
        self.reward_range = (0, 1)

        self.window: Window = None

        # Environment configuration
        self.width = width
        self.height = height
        self.max_steps = max_steps
        self.see_through_walls = see_through_walls


        # Current position and direction of the agent
        self.agent_pos: np.ndarray = None
        self.agent_dir: int = None

        # Current grid and mission and carryinh
        self.grid = Grid(width, height)
        self.colorful_grid = Grid(width, height)
        self.carrying = None

        # Rendering attributes
        self.render_mode = render_mode
        self.highlight = highlight
        self.tile_size = tile_size
        self.agent_pov = agent_pov

        # safety violations
        self.safety_violations = []
        self.safety_violations_timesteps = []
        self.total_timesteps = 0
        self.safety_violations_this_episode = None
        self.episode_count = 0

    def reset(self, *, state=None, seed=None, options=None):
        super().reset(seed=seed)
        # Reinitialize episode-specific variables
        if state:
            self.agent_pos = (state.pos_x, state.pos_y)
            self.agent_dir = state.dir
        else:
            self.agent_pos = (-1, -1)
            self.agent_dir = -1

        # Generate a new random grid at the start of each episode
        self._gen_grid(self.width, self.height)

        # These fields should be defined by _gen_grid
        assert (
            self.agent_pos >= (0, 0)
            if isinstance(self.agent_pos, tuple)
            else all(self.agent_pos >= 0) and self.agent_dir >= 0
        )

        # Check that the agent doesn't overlap with an object
        start_cell = self.grid.get(*self.agent_pos)
        assert start_cell is None or start_cell.can_overlap()

        # Item picked up, being carried, initially nothing
        self.carrying = None

        # Step count since episode start
        self.step_count = 0

        if self.render_mode == "human":
            self.render()

        # Return first observation
        obs = self.gen_obs()

        if self.safety_violations_this_episode is not None:
            self.safety_violations.append(self.safety_violations_this_episode)
            self.safety_violations_timesteps.append(self.total_timesteps)
        self.safety_violations_this_episode = 0
        self.episode_count += 1
        #input("Episode End, Hit Enter.")
        self.violation_deque.clear()
        return obs, {}

    def hash(self, size=16):
        """Compute a hash that uniquely identifies the current state of the environment.
        :param size: Size of the hashing
        """
        sample_hash = hashlib.sha256()

        to_encode = [self.grid.encode().tolist(), self.agent_pos, self.agent_dir]
        for item in to_encode:
            sample_hash.update(str(item).encode("utf8"))

        return sample_hash.hexdigest()[:size]

    @property
    def steps_remaining(self):
        return self.max_steps - self.step_count

    def printGrid(self, init=False):
        """
        Produce a pretty string of the environment's grid along with the agent.
        A grid cell is represented by 2-character string, the first one for
        the object and the second one for the color.
        """
        if init:
            self._gen_grid(self.grid.width, self.grid.height) # todo need to add this for minigrid2prism
        print("Dimensions: {} x {}".format(self.grid.height, self.grid.width))
        self._gen_grid(self.grid.width, self.grid.height)

        # Map of object types to short string
        OBJECT_TO_STR = {
            "wall": "W",
            "floor": "F",
            "door": "D",
            "key": "K",
            "ball": "A",
            "box": "B",
            "goal": "G",
            "lava": "V",
            "adversary": "Z",
            "slipperynorth": "n",
            "slipperysouth": "s",
            "slipperyeast": "e",
            "slipperywest": "w",
            "onewaynorth": "u",
            "onewaysouth": "v",
            "onewayeast": "x",
            "onewaywest": "y"

        }

        # Map agent's direction to short string
        AGENT_DIR_TO_STR = {0: ">", 1: "V", 2: "<", 3: "^"}

        str = ""
        bfs_rewards = list()
        background_str = ""
        for j in range(self.grid.height):
            for i in range(self.grid.width):
                b = self.grid.get_background(i, j)
                c = self.grid.get(i, j)
                if init:
                    if c and c.type == "wall":
                        background_str += OBJECT_TO_STR[c.type] + c.color[0].upper()
                    elif c and c.type == "slipperynorth":
                        background_str += OBJECT_TO_STR[c.type] + c.color[0].upper()
                    elif c and c.type == "slipperysouth":
                        background_str += OBJECT_TO_STR[c.type] + c.color[0].upper()
                    elif c and c.type == "slipperyeast":
                        background_str += OBJECT_TO_STR[c.type] + c.color[0].upper()
                    elif c and c.type == "slipperywest":
                        background_str += OBJECT_TO_STR[c.type] + c.color[0].upper()
                    elif b is None:
                        background_str += "  "
                    else:
                        if b.type != "floor":
                            type_str = OBJECT_TO_STR[b.type]
                        else:
                            type_str = " "

                        background_str += type_str + b.color.replace("light","")[0].upper()
                    if self.bfs_reward:
                        bfs_rewards.append(f"{i};{j};{self.bfs_reward[i + self.grid.width * j]}")


                if self.agent_pos is not None and i == self.agent_pos[0] and j == self.agent_pos[1]:
                    #str += 2 * AGENT_DIR_TO_STR[self.agent_dir]
                    if init:
                        str += "XR"
                    else:
                        str += 2 * AGENT_DIR_TO_STR[self.agent_dir]
                    continue


                if c is None or (isinstance(c, Floor) and c.color == "black"):
                    #print("{}, {}".format(i,j,), end="")
                    str += "  "
                    continue

                #print("{}, {}: {}{}".format(i,j,OBJECT_TO_STR[c.type], c.color[0]), end="")

                if c.type == "door":
                    if c.is_open:
                        str += "__"
                    elif c.is_locked:
                        str += "L" + c.color[0].upper()
                    else:
                        str += "D" + c.color[0].upper()
                    continue

                if not init and c.type == "adversary":
                    str += AGENT_DIR_TO_STR[c.adversary_dir] + c.color[0].upper()
                    continue

                str += OBJECT_TO_STR[c.type] + c.color[0].upper()

            if j < self.grid.height - 1:
                str += "\n"
                if init:
                    background_str += "\n"
            #print("")
        if init and self.bfs_reward:
            return str + "\n" + "-" * self.grid.width * 2 + "\n" + background_str + "\n" + "-" * self.grid.width * 2 + "\n" + ";".join(bfs_rewards)
        else:
            return str + "\n" + "-" * self.grid.width * 2 + "\n" + background_str

    @abstractmethod
    def _gen_grid(self, width, height):
        pass

    def _reward(self):
        """
        Compute the reward to be given upon success
        """

        return 0# - 0.9 * (self.step_count / self.max_steps)

    def _rand_int(self, low, high):
        """
        Generate random integer in [low,high[
        """

        return self.np_random.integers(low, high)

    def _rand_float(self, low, high):
        """
        Generate random float in [low,high[
        """

        return self.np_random.uniform(low, high)

    def _rand_bool(self):
        """
        Generate random boolean value
        """

        return self.np_random.integers(0, 2) == 0

    def _rand_elem(self, iterable):
        """
        Pick a random element in a list
        """

        lst = list(iterable)
        idx = self._rand_int(0, len(lst))
        return lst[idx]

    def _rand_subset(self, iterable, num_elems):
        """
        Sample a random subset of distinct elements of a list
        """

        lst = list(iterable)
        assert num_elems <= len(lst)

        out = []

        while len(out) < num_elems:
            elem = self._rand_elem(lst)
            lst.remove(elem)
            out.append(elem)

        return out

    def _rand_color(self):
        """
        Generate a random color name (string)
        """

        return self._rand_elem(COLOR_NAMES)

    def _rand_pos(self, xLow, xHigh, yLow, yHigh):
        """
        Generate a random (x,y) position tuple
        """

        return (
            self.np_random.integers(xLow, xHigh),
            self.np_random.integers(yLow, yHigh),
        )

    def place_obj(self, obj, top=None, size=None, reject_fn=None, max_tries=math.inf):
        """
        Place an object at an empty position in the grid

        :param top: top-left position of the rectangle where to place
        :param size: size of the rectangle where to place
        :param reject_fn: function to filter out potential positions
        """

        if top is None:
            top = (0, 0)
        else:
            top = (max(top[0], 0), max(top[1], 0))

        if size is None:
            size = (self.grid.width, self.grid.height)

        num_tries = 0

        while True:
            # This is to handle with rare cases where rejection sampling
            # gets stuck in an infinite loop
            if num_tries > max_tries:
                raise RecursionError("rejection sampling failed in place_obj")

            num_tries += 1

            pos = np.array(
                (
                    self._rand_int(top[0], min(top[0] + size[0], self.grid.width)),
                    self._rand_int(top[1], min(top[1] + size[1], self.grid.height)),
                )
            )

            pos = tuple(pos)

            # Don't place an object on top of a wall
            if isinstance(self.grid.get(*pos), Wall):
                continue

            if isOneWay(self.grid.get(*pos)):
                continue

            # Don't place the object on top of another object
            if not (isinstance(self.grid.get(*pos), Floor) or self.grid.get(*pos) is None):
                continue

            # Don't place the object where the agent is
            if np.array_equal(pos, self.agent_pos):
                continue

            # Check if there is a filtering criterion
            if reject_fn and reject_fn(self, pos):
                continue

            break

        self.grid.set(pos[0], pos[1], obj)

        if obj is not None:
            obj.init_pos = pos
            obj.cur_pos = pos

        return pos

    def put_obj(self, obj, i, j):
        """
        Put an object at a specific position in the grid
        """

        self.grid.set(i, j, obj)
        obj.init_pos = (i, j)
        obj.cur_pos = (i, j)

    def place_agent(self, top=None, size=None, rand_dir=True, max_tries=math.inf):
        """
        Set the agent's starting point at an empty position in the grid
        """

        self.agent_pos = (-1, -1)
        pos = self.place_obj(None, top, size, max_tries=max_tries)
        self.agent_pos = pos

        if rand_dir:
            self.agent_dir = self._rand_int(0, 4)
            if self.cannotTurn() and isinstance(self.grid.get(*self.front_pos), Wall):
                self.agent_dir = (self.agent_dir + 1) % 4

        return pos

    def add_slippery_tile(
        self,
        i: int,
        j: int,
        type: str
    ):
        """
        Adds a slippery tile to the grid
        """
        if type=="slipperynorth":
            slippery_tile = SlipperyNorth()
        elif type=="slipperysouth":
            slippery_tile = SlipperySouth()
        elif type=="slipperyeast":
            slippery_tile = SlipperyEast()
        elif type=="slipperywest":
            slippery_tile = SlipperyWest()
        else:
            slippery_tile = SlipperyNorth()

        self.grid.set(i, j, slippery_tile)
        return (i, j)

    #def add_adversary(
    #    self,
    #    i: int,
    #    j: int,
    #    color: str,
    #    direction: int = 0,
    #    tasks: List[Task] = [DoRandom()]
    #):
    #    """
    #    Adds a slippery tile to the grid
    #    """

    #    slippery_tile = Slippery()
    #    adv = Adversary(direction,color, tasks=tasks)
    #    self.put_obj(adv,i,j)
    #    self.adversaries[color] = adv
    #    return (i, j)


    @property
    def dir_vec(self):
        """
        Get the direction vector for the agent, pointing in the direction
        of forward movement.
        """

        assert self.agent_dir >= 0 and self.agent_dir < 4
        return DIR_TO_VEC[self.agent_dir]

    @property
    def right_vec(self):
        """
        Get the vector pointing to the right of the agent.
        """

        dx, dy = self.dir_vec
        return np.array((-dy, dx))

    @property
    def front_pos(self):
        """
        Get the position of the cell that is right in front of the agent
        """

        return self.agent_pos + self.dir_vec

    def get_view_coords(self, i, j):
        """
        Translate and rotate absolute grid coordinates (i, j) into the
        agent's partially observable view (sub-grid). Note that the resulting
        coordinates may be negative or outside of the agent's view size.
        """

        ax, ay = self.agent_pos
        dx, dy = self.dir_vec
        rx, ry = self.right_vec

        # Compute the absolute coordinates of the top-left view corner
        sz = self.agent_view_size
        hs = self.agent_view_size // 2
        tx = ax + (dx * (sz - 1)) - (rx * hs)
        ty = ay + (dy * (sz - 1)) - (ry * hs)

        lx = i - tx
        ly = j - ty

        # Project the coordinates of the object relative to the top-left
        # corner onto the agent's own coordinate system
        vx = rx * lx + ry * ly
        vy = -(dx * lx + dy * ly)

        return vx, vy

    def get_view_exts(self, agent_view_size=None):
        """
        Get the extents of the square set of tiles visible to the agent
        Note: the bottom extent indices are not included in the set
        if agent_view_size is None, use self.agent_view_size
        """

        agent_view_size = agent_view_size or self.agent_view_size

        # Facing right
        if self.agent_dir == 0:
            topX = self.agent_pos[0]
            topY = self.agent_pos[1] - agent_view_size // 2
        # Facing down
        elif self.agent_dir == 1:
            topX = self.agent_pos[0] - agent_view_size // 2
            topY = self.agent_pos[1]
        # Facing left
        elif self.agent_dir == 2:
            topX = self.agent_pos[0] - agent_view_size + 1
            topY = self.agent_pos[1] - agent_view_size // 2
        # Facing up
        elif self.agent_dir == 3:
            topX = self.agent_pos[0] - agent_view_size // 2
            topY = self.agent_pos[1] - agent_view_size + 1
        else:
            assert False, "invalid agent direction"

        botX = topX + agent_view_size
        botY = topY + agent_view_size

        return (topX, topY, botX, botY)

    def relative_coords(self, x, y):
        """
        Check if a grid position belongs to the agent's field of view, and returns the corresponding coordinates
        """

        vx, vy = self.get_view_coords(x, y)

        if vx < 0 or vy < 0 or vx >= self.agent_view_size or vy >= self.agent_view_size:
            return None

        return vx, vy

    def in_view(self, x, y):
        """
        check if a grid position is visible to the agent
        """

        return self.relative_coords(x, y) is not None

    def agent_sees(self, x, y):
        """
        Check if a non-empty grid position is visible to the agent
        """

        coordinates = self.relative_coords(x, y)
        if coordinates is None:
            return False
        vx, vy = coordinates

        obs = self.gen_obs()
        obs_grid, _ = Grid.decode(obs["image"])
        obs_cell = obs_grid.get(vx, vy)
        world_cell = self.grid.get(x, y)

        assert world_cell is not None

        return obs_cell is not None and obs_cell.type == world_cell.type

    def get_neighbours_prob_forward(self, agent_pos, probabilities, offset):
        neighbours = [tuple((x,y)) for x in range(agent_pos[0]-1, agent_pos[0]+2) for y in range(agent_pos[1]-1,agent_pos[1]+2)]
        probabilities_dict = dict(zip(neighbours, probabilities))

        for pos in probabilities_dict:
            cell = self.grid.get(*pos)
            if cell is not None and not cell.can_overlap():
                probabilities_dict[pos] = 0.0

        sum_prob = 0
        for pos in probabilities_dict:
            if (probabilities_dict[pos]!=-50):
                sum_prob += probabilities_dict[pos]

        if probabilities_dict[tuple((agent_pos[0] + offset[0], agent_pos[1]+offset[1]))] == 0:
            probabilities_dict[agent_pos] = 1-sum_prob
        else:
            probabilities_dict[tuple((agent_pos[0] + offset[0], agent_pos[1]+offset[1]))] = 1-sum_prob
            probabilities_dict[agent_pos] = 0.0

        return list(probabilities_dict.keys()), list(probabilities_dict.values())

    def get_neighbours_prob_turn(self, agent_pos, probabilities):
        neighbours = [tuple((x,y)) for x in range(agent_pos[0]-1, agent_pos[0]+2) for y in range(agent_pos[1]-1,agent_pos[1]+2)]
        non_blocked_neighbours = []
        i = 0
        non_blocked_prob = []
        for pos in neighbours:
            cell = self.grid.get(*pos)
            if (cell is None or cell.can_overlap()):
                non_blocked_neighbours.append(pos)
                non_blocked_prob.append(probabilities[i])
            i += 1

        sum_prob = 0
        for prob in non_blocked_prob:
            if (prob!=-50):
                sum_prob += prob
        non_blocked_prob = [x if x!=-50 else 1-sum_prob for x in non_blocked_prob]
        return non_blocked_neighbours, non_blocked_prob

    def cannotTurn(self):
        agent_pos = self.agent_pos
        return (( isinstance(self.grid.get(agent_pos[0], agent_pos[1]-1), Wall)
            and   isinstance(self.grid.get(agent_pos[0], agent_pos[1]+1), Wall) )
            or (  isinstance(self.grid.get(agent_pos[0]-1, agent_pos[1]), Wall)
            and   isinstance(self.grid.get(agent_pos[0]+1, agent_pos[1]), Wall) ))

    def step(self, action):
        self.step_count += 1
        self.total_timesteps += 1

        reward = 0
        terminated = False
        truncated = False
        # Get the position in front of the agent
        fwd_pos = self.front_pos

        # Get the contents of the cell in front of the agent
        try:
            fwd_cell = self.grid.get(*fwd_pos)
        except:
            print(f"Tried to get fwd_cell at {self.agent_pos}")
            print(f"Cell at self.agent_pos = {self.grid.get(*self.agent_pos)}")
            print(f"Time step: {self.step_count}")
            assert(False)
        current_cell = self.grid.get(*self.agent_pos)

        ran_into_lava = False
        reached_goal = False
        need_position_update = False

        noise = getattr(self, 'noise', 0.0)

        if action == self.actions.forward and isSlippery(current_cell):
            possible_fwd_pos, prob = self.get_neighbours_prob_forward(self.agent_pos, current_cell.probabilities_forward, current_cell.offset)
            fwd_pos_index = np.random.choice(len(possible_fwd_pos), 1, p=prob)
            fwd_pos = possible_fwd_pos[fwd_pos_index[0]]
            fwd_cell = self.grid.get(*fwd_pos)
            need_position_update = True

        elif action == self.actions.forward and isOneWay(current_cell):
            need_position_update = True
            fwd_cell = self.grid.get(*fwd_pos)

        # Rotate left
        elif action == self.actions.left:
            if self.cannotTurn():
                reward = -100
            else:
                self.agent_dir -= 1
                if self.agent_dir < 0:
                    self.agent_dir += 4
                if isSlippery(current_cell):
                    possible_fwd_pos, prob = self.get_neighbours_prob_turn(self.agent_pos, current_cell.probabilities_turn)
                    fwd_pos_index = np.random.choice(len(possible_fwd_pos), 1, p=prob)
                    fwd_pos = possible_fwd_pos[fwd_pos_index[0]]
                    fwd_cell = self.grid.get(*fwd_pos)
                    need_position_update = True

        # Rotate right
        elif action == self.actions.right:
            if self.cannotTurn():
                reward = -100
            else:
                self.agent_dir = (self.agent_dir + 1) % 4
                if isSlippery(current_cell):
                    possible_fwd_pos, prob = self.get_neighbours_prob_turn(self.agent_pos, current_cell.probabilities_turn)
                    fwd_pos_index = np.random.choice(len(possible_fwd_pos), 1, p=prob)
                    fwd_pos = possible_fwd_pos[fwd_pos_index[0]]
                    fwd_cell = self.grid.get(*fwd_pos)
                    need_position_update = True

        # forward
        elif action == self.actions.forward:
            if fwd_cell is None or fwd_cell.can_overlap():
                #self.agent_pos = tuple(fwd_pos)
                fwd_cell = self.grid.get(*fwd_pos)
                need_position_update = True

        # Pick up an object
        elif action == self.actions.pickup:
            if fwd_cell and fwd_cell.can_pickup():
                if self.carrying is None:
                    if type(fwd_cell) == Box and fwd_cell.color == "red":
                        if not fwd_cell.picked_up:
                            self.reward += PICKUPREWARD
                            fwd_cell.picked_up = True

                        self.carrying = fwd_cell
                        self.carrying.cur_pos = np.array([-1, -1])
                        self.grid.set(fwd_pos[0], fwd_pos[1], None)
        # Drop an object
        elif action == self.actions.drop:
            if not fwd_cell and self.carrying:
                if type(self.carrying == Box) and self.carrying.color == "red" and self.grid.get_background(*fwd_pos) and self.grid.get_background(*fwd_pos).color == "red":
                    self.reward += DELIVERREWARD
                    terminated = True
                    self.carrying.placed_at_destination = True
                    self.grid.set(fwd_pos[0], fwd_pos[1], self.carrying)
                    self.carrying.cur_pos = fwd_pos
                    self.carrying = None

        # Toggle/activate an object
        elif action == self.actions.toggle:
            if fwd_cell:
                fwd_cell.toggle(self, fwd_pos)

        # Done action (not used by default)
        elif action == self.actions.done:
            pass

        else:
            raise ValueError(f"Unknown action: {action}")
        if fwd_cell is not None and isOneWay(fwd_cell) and abs(fwd_cell.direction - self.agent_dir) == 2:
            fwd_pos = self.agent_pos
        elif fwd_cell is not None and fwd_cell.type == "goal":
            terminated = True
            reached_goal = True
            reward = 100
        elif fwd_cell is not None and fwd_cell.type == "lava":
            terminated = True
            ran_into_lava = True
            reward = -100
        if need_position_update and (fwd_cell is None or fwd_cell.can_overlap()):
            self.agent_pos = tuple(fwd_pos)

        if self.step_count >= self.max_steps:
            truncated = True

        if self.render_mode == "human":
            self.render()

        obs = self.gen_obs()
        return obs, reward, terminated, truncated, {"pos": self.agent_pos, "dir": self.agent_dir, "ran_into_lava": ran_into_lava, "reached_goal": reached_goal, "is_success": reached_goal}

    def gen_obs_grid(self, agent_view_size=None):
        """
        Generate the sub-grid observed by the agent.
        This method also outputs a visibility mask telling us which grid
        cells the agent can actually see.
        if agent_view_size is None, self.agent_view_size is used
        """

        topX, topY, botX, botY = self.get_view_exts(agent_view_size)

        agent_view_size = agent_view_size or self.agent_view_size

        grid = self.grid.slice(topX, topY, agent_view_size, agent_view_size)

        for i in range(self.agent_dir + 1):
            grid = grid.rotate_left()

        # Process occluders and visibility
        # Note that this incurs some performance cost
        if not self.see_through_walls:
            vis_mask = grid.process_vis(
                agent_pos=(agent_view_size // 2, agent_view_size - 1)
            )
        else:
            vis_mask = np.ones(shape=(grid.width, grid.height), dtype=bool)

        # Make it so the agent sees what it's carrying
        # We do this by placing the carried object at the agent's position
        # in the agent's partially observable view
        agent_pos = grid.width // 2, grid.height - 1
        if self.carrying:
            grid.set(*agent_pos, self.carrying)
        else:
            grid.set(*agent_pos, None)

        return grid, vis_mask

    def gen_obs(self):
        """
        Generate the agent's view (partially observable, low-resolution encoding)
        """

        grid, vis_mask = self.gen_obs_grid()

        # Encode the partially observable view into a numpy array
        image = grid.encode(vis_mask)

        # Observations are dictionaries containing:
        # - an image (partially observable view of the environment)
        # - the agent's direction/orientation (acting as a compass)
        # - a textual mission string (instructions for the agent)
        obs = {"image": image, "direction": self.agent_dir, "mission": self.mission}

        return obs

    def get_pov_render(self, tile_size):
        """
        Render an agent's POV observation for visualization
        """
        grid, vis_mask = self.gen_obs_grid()

        # Render the whole grid
        img = grid.render(
            tile_size,
            agent_pos=(self.agent_view_size // 2, self.agent_view_size - 1),
            agent_dir=3,
            highlight_mask=vis_mask,
            carrying=self.carrying,
        )

        return img

    def get_full_render(self, highlight, tile_size):
        """
        Render a non-paratial observation for visualization
        """
        # Compute which cells are visible to the agent
        _, vis_mask = self.gen_obs_grid()

        # Compute the world coordinates of the bottom-left corner
        # of the agent's view area
        f_vec = self.dir_vec
        r_vec = self.right_vec
        top_left = (
            self.agent_pos
            + f_vec * (self.agent_view_size - 1)
            - r_vec * (self.agent_view_size // 2)
        )

        # Mask of which cells to highlight
        highlight_mask = np.zeros(shape=(self.width, self.height), dtype=bool)

        # For each cell in the visibility mask
        for vis_j in range(0, self.agent_view_size):
            for vis_i in range(0, self.agent_view_size):
                # If this cell is not visible, don't highlight it
                if not vis_mask[vis_i, vis_j]:
                    continue

                # Compute the world coordinates of this cell
                abs_i, abs_j = top_left - (f_vec * vis_j) + (r_vec * vis_i)

                if abs_i < 0 or abs_i >= self.width:
                    continue
                if abs_j < 0 or abs_j >= self.height:
                    continue

                # Mark this cell to be highlighted
                highlight_mask[abs_i, abs_j] = True

        # Render the whole grid
        img = self.grid.render(
            tile_size,
            self.agent_pos,
            self.agent_dir,
            highlight_mask=highlight_mask if highlight else None,
            carrying=self.carrying,
        )

        return img

    def get_frame(
        self,
        highlight: bool = True,
        tile_size: int = TILE_PIXELS,
        agent_pov: bool = False,
    ):
        """Returns an RGB image corresponding to the whole environment or the agent's point of view.

        Args:

            highlight (bool): If true, the agent's field of view or point of view is highlighted with a lighter gray color.
            tile_size (int): How many pixels will form a tile from the NxM grid.
            agent_pov (bool): If true, the rendered frame will only contain the point of view of the agent.

        Returns:

            frame (np.ndarray): A frame of type numpy.ndarray with shape (x, y, 3) representing RGB values for the x-by-y pixel image.

        """

        if agent_pov:
            return self.get_pov_render(tile_size)
        else:
            return self.get_full_render(highlight, tile_size)

    def render(self, mode=""):
        img = self.get_frame(self.highlight, self.tile_size, self.agent_pov)
        if mode == "human":
            if self.window is None:
                self.window = Window("gym_minigrid")
                self.window.show(block=False)
            self.window.set_caption(self.mission)
            self.window.show_img(img)
        elif mode == "rgb_array":
            return img

    def close(self):
        if self.window:
            self.window.close()
