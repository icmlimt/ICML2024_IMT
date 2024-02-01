from warnings import warn

from gym.envs.registration import register

from gym_minigrid import minigrid, roomgrid, wrappers

from gym_minigrid.policyRepairEnv import *

def register_minigrid_envs():

    # gym_minigrid package deprecated in favor of minigrid
    warn(
        "The package name gym_minigrid has been deprecated in favor of minigrid. Please uninstall gym_minigrid and install minigrid with `pip install minigrid`. Future releases will be maintained under the new package name minigrid.",
        DeprecationWarning,
        stacklevel=2,
    )

    # BlockedUnlockPickup
    # ----------------------------------------

    register(
        id="MiniGrid-BlockedUnlockPickup-v0",
        entry_point="gym_minigrid.envs:BlockedUnlockPickupEnv",
    )

    # LavaCrossing
    # ----------------------------------------
    register(
        id="MiniGrid-LavaCrossingS9N1-v0",
        entry_point="gym_minigrid.envs:CrossingEnv",
        kwargs={"size": 9, "num_crossings": 1},
    )

    register(
        id="MiniGrid-LavaCrossingS9N2-v0",
        entry_point="gym_minigrid.envs:CrossingEnv",
        kwargs={"size": 9, "num_crossings": 2},
    )

    register(
        id="MiniGrid-LavaCrossingS9N3-v0",
        entry_point="gym_minigrid.envs:CrossingEnv",
        kwargs={"size": 9, "num_crossings": 3},
    )

    register(
        id="MiniGrid-LavaCrossingS11N5-v0",
        entry_point="gym_minigrid.envs:CrossingEnv",
        kwargs={"size": 11, "num_crossings": 5},
    )

    # SimpleCrossing
    # ----------------------------------------

    register(
        id="MiniGrid-SimpleCrossingS9N1-v0",
        entry_point="gym_minigrid.envs:CrossingEnv",
        kwargs={"size": 9, "num_crossings": 1, "obstacle_type": minigrid.Wall},
    )

    register(
        id="MiniGrid-SimpleCrossingS9N2-v0",
        entry_point="gym_minigrid.envs:CrossingEnv",
        kwargs={"size": 9, "num_crossings": 2, "obstacle_type": minigrid.Wall},
    )

    register(
        id="MiniGrid-SimpleCrossingS9N3-v0",
        entry_point="gym_minigrid.envs:CrossingEnv",
        kwargs={"size": 9, "num_crossings": 3, "obstacle_type": minigrid.Wall},
    )

    register(
        id="MiniGrid-SimpleCrossingS11N5-v0",
        entry_point="gym_minigrid.envs:CrossingEnv",
        kwargs={"size": 11, "num_crossings": 5, "obstacle_type": minigrid.Wall},
    )

    # DistShift
    # ----------------------------------------

    register(
        id="MiniGrid-DistShift1-v0",
        entry_point="gym_minigrid.envs:DistShiftEnv",
        kwargs={"strip2_row": 2},
    )

    register(
        id="MiniGrid-DistShift2-v0",
        entry_point="gym_minigrid.envs:DistShiftEnv",
        kwargs={"strip2_row": 5},
    )

    # DoorKey
    # ----------------------------------------

    register(
        id="MiniGrid-DoorKey-5x5-v0",
        entry_point="gym_minigrid.envs:DoorKeyEnv",
        kwargs={"size": 5},
    )

    register(
        id="MiniGrid-DoorKey-6x6-v0",
        entry_point="gym_minigrid.envs:DoorKeyEnv",
        kwargs={"size": 5},
    )

    register(
        id="MiniGrid-DoorKey-8x8-v0",
        entry_point="gym_minigrid.envs:DoorKeyEnv",
        kwargs={"size": 8},
    )

    register(
        id="MiniGrid-DoorKey-16x16-v0",
        entry_point="gym_minigrid.envs:DoorKeyEnv",
        kwargs={"size": 16},
    )

    # Dynamic-Obstacles
    # ----------------------------------------

    register(
        id="MiniGrid-Dynamic-Obstacles-5x5-v0",
        entry_point="gym_minigrid.envs:DynamicObstaclesEnv",
        kwargs={"size": 5, "n_obstacles": 2},
    )

    register(
        id="MiniGrid-Dynamic-Obstacles-Random-5x5-v0",
        entry_point="gym_minigrid.envs:DynamicObstaclesEnv",
        kwargs={"size": 5, "agent_start_pos": None, "n_obstacles": 2},
    )

    register(
        id="MiniGrid-Dynamic-Obstacles-6x6-v0",
        entry_point="gym_minigrid.envs:DynamicObstaclesEnv",
        kwargs={"size": 6, "n_obstacles": 3},
    )

    register(
        id="MiniGrid-Dynamic-Obstacles-Random-6x6-v0",
        entry_point="gym_minigrid.envs:DynamicObstaclesEnv",
        kwargs={"size": 6, "agent_start_pos": None, "n_obstacles": 3},
    )

    register(
        id="MiniGrid-Dynamic-Obstacles-8x8-v0",
        entry_point="gym_minigrid.envs:DynamicObstaclesEnv",
    )

    register(
        id="MiniGrid-Dynamic-Obstacles-16x16-v0",
        entry_point="gym_minigrid.envs:DynamicObstaclesEnv",
        kwargs={"size": 16, "n_obstacles": 8},
    )

    # Empty
    # ----------------------------------------

    register(
        id="MiniGrid-Empty-5x5-v0",
        entry_point="gym_minigrid.envs:EmptyEnv",
        kwargs={"size": 5},
    )

    register(
        id="MiniGrid-Empty-Random-5x5-v0",
        entry_point="gym_minigrid.envs:EmptyEnv",
        kwargs={"size": 5, "agent_start_pos": None},
    )

    register(
        id="MiniGrid-Empty-6x6-v0",
        entry_point="gym_minigrid.envs:EmptyEnv",
        kwargs={"size": 6},
    )

    register(
        id="MiniGrid-Empty-Random-6x6-v0",
        entry_point="gym_minigrid.envs:EmptyEnv",
        kwargs={"size": 6, "agent_start_pos": None},
    )

    register(
        id="MiniGrid-Empty-8x8-v0",
        entry_point="gym_minigrid.envs:EmptyEnv",
    )

    register(
        id="MiniGrid-Empty-11x11-v0",
        entry_point="gym_minigrid.envs:EmptyEnv",
        kwargs={"size": 11},
    )

    register(
        id="MiniGrid-Empty-16x16-v0",
        entry_point="gym_minigrid.envs:EmptyEnv",
        kwargs={"size": 16},
    )

    # Fetch
    # ----------------------------------------

    register(
        id="MiniGrid-Fetch-5x5-N2-v0",
        entry_point="gym_minigrid.envs:FetchEnv",
        kwargs={"size": 5, "numObjs": 2},
    )

    register(
        id="MiniGrid-Fetch-6x6-N2-v0",
        entry_point="gym_minigrid.envs:FetchEnv",
        kwargs={"size": 6, "numObjs": 2},
    )

    register(id="MiniGrid-Fetch-8x8-N3-v0", entry_point="gym_minigrid.envs:FetchEnv")

    # FourRooms
    # ----------------------------------------

    register(
        id="MiniGrid-FourRooms-v0",
        entry_point="gym_minigrid.envs:FourRoomsEnv",
    )

    # GoToDoor
    # ----------------------------------------

    register(
        id="MiniGrid-GoToDoor-5x5-v0",
        entry_point="gym_minigrid.envs:GoToDoorEnv",
    )

    register(
        id="MiniGrid-GoToDoor-6x6-v0",
        entry_point="gym_minigrid.envs:GoToDoorEnv",
        kwargs={"size": 6},
    )

    register(
        id="MiniGrid-GoToDoor-8x8-v0",
        entry_point="gym_minigrid.envs:GoToDoorEnv",
        kwargs={"size": 8},
    )

    # GoToObject
    # ----------------------------------------

    register(
        id="MiniGrid-GoToObject-6x6-N2-v0",
        entry_point="gym_minigrid.envs:GoToObjectEnv",
    )

    register(
        id="MiniGrid-GoToObject-8x8-N2-v0",
        entry_point="gym_minigrid.envs:GoToObjectEnv",
        kwargs={"size": 8, "numObjs": 2},
    )

    # KeyCorridor
    # ----------------------------------------

    register(
        id="MiniGrid-KeyCorridorS3R1-v0",
        entry_point="gym_minigrid.envs:KeyCorridorEnv",
        kwargs={"room_size": 3, "num_rows": 1},
    )

    register(
        id="MiniGrid-KeyCorridorS3R2-v0",
        entry_point="gym_minigrid.envs:KeyCorridorEnv",
        kwargs={"room_size": 3, "num_rows": 2},
    )

    register(
        id="MiniGrid-KeyCorridorS3R3-v0",
        entry_point="gym_minigrid.envs:KeyCorridorEnv",
        kwargs={"room_size": 3, "num_rows": 3},
    )

    register(
        id="MiniGrid-KeyCorridorS4R3-v0",
        entry_point="gym_minigrid.envs:KeyCorridorEnv",
        kwargs={"room_size": 4, "num_rows": 3},
    )

    register(
        id="MiniGrid-KeyCorridorS5R3-v0",
        entry_point="gym_minigrid.envs:KeyCorridorEnv",
        kwargs={"room_size": 5, "num_rows": 3},
    )

    register(
        id="MiniGrid-KeyCorridorS6R3-v0",
        entry_point="gym_minigrid.envs:KeyCorridorEnv",
        kwargs={"room_size": 6, "num_rows": 3},
    )

    # LavaGap
    # ----------------------------------------

    register(
        id="MiniGrid-LavaGapS5-v0",
        entry_point="gym_minigrid.envs:LavaGapEnv",
        kwargs={"size": 5},
    )

    register(
        id="MiniGrid-LavaGapS6-v0",
        entry_point="gym_minigrid.envs:LavaGapEnv",
        kwargs={"size": 6},
    )

    register(
        id="MiniGrid-LavaGapS7-v0",
        entry_point="gym_minigrid.envs:LavaGapEnv",
        kwargs={"size": 7},
    )

    # LockedRoom
    # ----------------------------------------

    register(
        id="MiniGrid-LockedRoom-v0",
        entry_point="gym_minigrid.envs:LockedRoomEnv",
    )

    # Memory
    # ----------------------------------------

    register(
        id="MiniGrid-MemoryS17Random-v0",
        entry_point="gym_minigrid.envs:MemoryEnv",
        kwargs={"size": 17, "random_length": True},
    )

    register(
        id="MiniGrid-MemoryS13Random-v0",
        entry_point="gym_minigrid.envs:MemoryEnv",
        kwargs={"size": 13, "random_length": True},
    )

    register(
        id="MiniGrid-MemoryS13-v0",
        entry_point="gym_minigrid.envs:MemoryEnv",
        kwargs={"size": 13},
    )

    register(
        id="MiniGrid-MemoryS11-v0",
        entry_point="gym_minigrid.envs:MemoryEnv",
        kwargs={"size": 11},
    )

    register(
        id="MiniGrid-MemoryS9-v0",
        entry_point="gym_minigrid.envs:MemoryEnv",
        kwargs={"size": 9},
    )

    register(
        id="MiniGrid-MemoryS7-v0",
        entry_point="gym_minigrid.envs:MemoryEnv",
        kwargs={"size": 7},
    )

    # MultiRoom
    # ----------------------------------------

    register(
        id="MiniGrid-MultiRoom-N2-S4-v0",
        entry_point="gym_minigrid.envs:MultiRoomEnv",
        kwargs={"minNumRooms": 2, "maxNumRooms": 2, "maxRoomSize": 4},
    )

    register(
        id="MiniGrid-MultiRoom-N4-S5-v0",
        entry_point="gym_minigrid.envs:MultiRoomEnv",
        kwargs={"minNumRooms": 6, "maxNumRooms": 6, "maxRoomSize": 5},
    )

    register(
        id="MiniGrid-MultiRoom-N6-v0",
        entry_point="gym_minigrid.envs:MultiRoomEnv",
        kwargs={"minNumRooms": 6, "maxNumRooms": 6},
    )

    # ObstructedMaze
    # ----------------------------------------

    register(
        id="MiniGrid-ObstructedMaze-1Dl-v0",
        entry_point="gym_minigrid.envs:ObstructedMaze_1Dlhb",
        kwargs={"key_in_box": False, "blocked": False},
    )

    register(
        id="MiniGrid-ObstructedMaze-1Dlh-v0",
        entry_point="gym_minigrid.envs:ObstructedMaze_1Dlhb",
        kwargs={"key_in_box": True, "blocked": False},
    )

    register(
        id="MiniGrid-ObstructedMaze-1Dlhb-v0",
        entry_point="gym_minigrid.envs:ObstructedMaze_1Dlhb",
    )

    register(
        id="MiniGrid-ObstructedMaze-2Dl-v0",
        entry_point="gym_minigrid.envs:ObstructedMaze_Full",
        kwargs={
            "agent_room": (2, 1),
            "key_in_box": False,
            "blocked": False,
            "num_quarters": 1,
            "num_rooms_visited": 4,
        },
    )

    register(
        id="MiniGrid-ObstructedMaze-2Dlh-v0",
        entry_point="gym_minigrid.envs:ObstructedMaze_Full",
        kwargs={
            "agent_room": (2, 1),
            "key_in_box": True,
            "blocked": False,
            "num_quarters": 1,
            "num_rooms_visited": 4,
        },
    )

    register(
        id="MiniGrid-ObstructedMaze-2Dlhb-v0",
        entry_point="gym_minigrid.envs:ObstructedMaze_Full",
        kwargs={
            "agent_room": (2, 1),
            "key_in_box": True,
            "blocked": True,
            "num_quarters": 1,
            "num_rooms_visited": 4,
        },
    )

    register(
        id="MiniGrid-ObstructedMaze-1Q-v0",
        entry_point="gym_minigrid.envs:ObstructedMaze_Full",
        kwargs={
            "agent_room": (1, 1),
            "key_in_box": True,
            "blocked": True,
            "num_quarters": 1,
            "num_rooms_visited": 5,
        },
    )

    register(
        id="MiniGrid-ObstructedMaze-2Q-v0",
        entry_point="gym_minigrid.envs:ObstructedMaze_Full",
        kwargs={
            "agent_room": (2, 1),
            "key_in_box": True,
            "blocked": True,
            "num_quarters": 2,
            "num_rooms_visited": 11,
        },
    )

    register(
        id="MiniGrid-ObstructedMaze-Full-v0",
        entry_point="gym_minigrid.envs:ObstructedMaze_Full",
    )

    # Playground
    # ----------------------------------------

    register(
        id="MiniGrid-Playground-v0",
        entry_point="gym_minigrid.envs:PlaygroundEnv",
    )

    # PutNear
    # ----------------------------------------

    register(
        id="MiniGrid-PutNear-6x6-N2-v0",
        entry_point="gym_minigrid.envs:PutNearEnv",
    )

    register(
        id="MiniGrid-PutNear-8x8-N3-v0",
        entry_point="gym_minigrid.envs:PutNearEnv",
        kwargs={"size": 8, "numObjs": 3},
    )

    # RedBlueDoors
    # ----------------------------------------

    register(
        id="MiniGrid-RedBlueDoors-6x6-v0",
        entry_point="gym_minigrid.envs:RedBlueDoorEnv",
        kwargs={"size": 6},
    )

    register(
        id="MiniGrid-RedBlueDoors-8x8-v0",
        entry_point="gym_minigrid.envs:RedBlueDoorEnv",
    )

    # Unlock
    # ----------------------------------------

    register(id="MiniGrid-Unlock-v0", entry_point="gym_minigrid.envs:UnlockEnv")

    # UnlockPickup
    # ----------------------------------------

    register(
        id="MiniGrid-UnlockPickup-v0",
        entry_point="gym_minigrid.envs:UnlockPickupEnv",
    )



    # GuidedRLExperiments
    # ----------------------------------------

    #register(
    #    id="MiniGrid-GuidedRLExperiments-TwoRooms-v0",
    #    entry_point="gym_minigrid.envs:TwoRooms",
    #)

    ## AgExperiments
    ## ----------------------------------------

    #register(
    #    id="MiniGrid-AgExperiments-DeliveryStations-v0",
    #    entry_point="gym_minigrid.envs:DeliveryStations",
    #)

    #register(
    #    id="MiniGrid-AgExperiments-DeliveryStationsWithRows-v0",
    #    entry_point="gym_minigrid.envs:DeliveryStationsWithRows",
    #)

    #register(
    #    id="MiniGrid-WareHouse",
    #    entry_point="gym_minigrid.envs:WareHouse",
    #)
    #register(
    #    id="MiniGrid-AgExperiments-DeliveryStationsSmall-v0",
    #    entry_point="gym_minigrid.envs:DeliveryStationsSmall",
    #)
    #register(
    #    id="MiniGrid-AgExperiments-DeliveryStationsSmallNoAg-v0",
    #    entry_point="gym_minigrid.envs:DeliveryStationsSmallNoAg",
    #)

    register(
        id="Testing-v0",
        entry_point="gym_minigrid.envs:Testing",
    )

    register(
        id="LavaObstaclesMini-v0",
        entry_point="gym_minigrid.envs:LavaObstaclesMini",
    )

    register(
        id="LavaSymmetricMini-v0",
        entry_point="gym_minigrid.envs:LavaSymmetricMini",
    )

    register(
        id="SlipperyMini-v0",
        entry_point="gym_minigrid.envs:SlipperyMini",
    )

    # Testing Playground

    register(
        id="Abagarion",
        entry_point="gym_minigrid.envs:Abagarion",
    )

    register(
        id="ColumnsMini-v0",
        entry_point="gym_minigrid.envs:ColumnsMini",
    )

    register(
        id="ExperimentSlippery-v0",
        entry_point="gym_minigrid.envs:ExperimentSlippery",
    )

    #cliff walking

    register(
        id="MyCliffWalking-v0",
        entry_point="gym_minigrid.envs:CliffWalking",
        kwargs={"width": 12, "height": 8, "nr_cliffs":1},
    )

    register(
        id="MyCliffWalking-S9-v0",
        entry_point="gym_minigrid.envs:CliffWalking",
        kwargs={"width": 9, "nr_cliffs":1},
    )

    register(
        id="MyCliffWalking-S17-v0",
        entry_point="gym_minigrid.envs:CliffWalking",
        kwargs={"width": 17, "nr_cliffs":2},
    )

    register(
        id="MyCliffWalking-S25-v0",
        entry_point="gym_minigrid.envs:CliffWalking",
        kwargs={"width": 25, "nr_cliffs":3},
    )
    register(
        id="MyCliffWalking-S33-v0",
        entry_point="gym_minigrid.envs:CliffWalking",
        kwargs={"width": 33, "nr_cliffs":4},
    )

    register(
        id="MyTaxi-v0",
        entry_point="gym_minigrid.envs:Taxi",
        kwargs={"size": 15, "nr_structures": 1},
    )

    register(
        id="MyTaxi-S29-v0",
        entry_point="gym_minigrid.envs:Taxi",
        kwargs={"size": 29, "nr_structures": 2},
    )

    register(
        id="MyTaxi-S43-v0",
        entry_point="gym_minigrid.envs:Taxi",
        kwargs={"size": 43, "nr_structures": 3},
    )

    register(
        id="MyTaxi-S57-v0",
        entry_point="gym_minigrid.envs:Taxi",
        kwargs={"size": 57, "nr_structures": 4},
    )

    register(
        id="MyTaxi-S71-v0",
        entry_point="gym_minigrid.envs:Taxi",
        kwargs={"size": 71, "nr_structures": 5},
    )

    register(
        id="MyTaxi-S85-v0",
        entry_point="gym_minigrid.envs:Taxi",
        kwargs={"size": 85, "nr_structures": 6},
    )

    register(
        id="MyTaxi-S99-v0",
        entry_point="gym_minigrid.envs:Taxi",
        kwargs={"size": 99, "nr_structures": 7},
    )

    register(
        id="MyTaxi-S113-v0",
        entry_point="gym_minigrid.envs:Taxi",
        kwargs={"size": 113, "nr_structures": 8},
    )

    register(
        id="MyTaxi-S127-v0",
        entry_point="gym_minigrid.envs:Taxi",
        kwargs={"size": 127, "nr_structures": 9},
    )

    register(
        id="MyTaxi-S141-v0",
        entry_point="gym_minigrid.envs:Taxi",
        kwargs={"size": 141, "nr_structures": 10},
    )

    register(
        id="MyTaxi-S155-v0",
        entry_point="gym_minigrid.envs:Taxi",
        kwargs={"size": 155, "nr_structures": 11},
    )

    register(
        id="RiverDeadend-v0",
        entry_point="gym_minigrid.envs:RiverDeadend"
    )

    register(
        id="CityDeadend-v0",
        entry_point="gym_minigrid.envs:CityDeadend"
    )

    register(
        id="Cookie-v0",
        entry_point="gym_minigrid.envs:Cookie"
    )

    register(
        id="Barcelona-v0",
        entry_point="gym_minigrid.envs:Barcelona"
    )

    #register(
    #    id="UAV-v0",
    #    entry_point="gym_minigrid.envs:UAV"
    #)
