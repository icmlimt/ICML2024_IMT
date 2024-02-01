#pragma once

#include <iostream>
#include <fstream>
#include <map>
#include <utility>

#include "MinigridGrammar.h"
#include "PrismModulesPrinter.h"

struct GridOptions {
  std::vector<AgentName> agentsToBeConsidered;
  std::vector<AgentName> agentsWithView;
  std::vector<AgentName> agentsWithProbabilisticBehaviour;
  std::vector<float>     probabilitiesForActions;
  bool                   enforceOneWays;
};

class Grid {
  public:
    Grid(cells gridCells, cells background, const GridOptions &gridOptions, const std::map<coordinates, float> &stateRewards = {});

    cells getGridCells();

    bool isBlocked(coordinates p);
    bool isWall(coordinates p);
    bool isLockedDoor(coordinates p);
    bool isKey(coordinates p);
    bool isBox(coordinates p);
    void printToPrism(std::ostream &os, const prism::ModelType& modelType);

    std::array<bool, 8> getWalkableDirOf8Neighborhood(cell c);

    friend std::ostream& operator<<(std::ostream& os, const Grid &grid);

  private:
    GridOptions gridOptions;

    cells allGridCells;
    cells background;
    coordinates maxBoundaries;

    cell agent;
    cells adversaries;
    AgentNameAndPositionMap agentNameAndPositionMap;

    cells walls;
    cells floor;
    cells slipperyNorth;
    cells slipperyEast;
    cells slipperySouth;
    cells slipperyWest;
    cells oneWayNorth;
    cells oneWayEast;
    cells oneWaySouth;
    cells oneWayWest;
    cells lockedDoors;
    cells boxes;
    cells lava;

    cells goals;
    cells keys;

    std::map<Color, cells> backgroundTiles;

    std::map<coordinates, float> stateRewards;
};
