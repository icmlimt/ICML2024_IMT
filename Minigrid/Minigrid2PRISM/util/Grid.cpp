#include "Grid.h"

#include <algorithm>

Grid::Grid(cells gridCells, cells background, const GridOptions &gridOptions, const std::map<coordinates, float> &stateRewards)
  : allGridCells(gridCells), background(background), gridOptions(gridOptions), stateRewards(stateRewards)
{
  cell max = allGridCells.at(allGridCells.size() - 1);
  maxBoundaries = std::make_pair(max.row - 1, max.column - 1);
  std::copy_if(gridCells.begin(), gridCells.end(), std::back_inserter(walls), [](cell c) {
      return c.type == Type::Wall;
  });
  std::copy_if(gridCells.begin(), gridCells.end(), std::back_inserter(lava), [](cell c) {
      return c.type == Type::Lava;
  });
  std::copy_if(gridCells.begin(), gridCells.end(), std::back_inserter(floor), [](cell c) {
      return c.type == Type::Floor;
  });
  std::copy_if(background.begin(), background.end(), std::back_inserter(slipperyNorth), [](cell c) {
      return c.type == Type::SlipperyNorth;
  });
  std::copy_if(background.begin(), background.end(), std::back_inserter(slipperyEast), [](cell c) {
      return c.type == Type::SlipperyEast;
  });
  std::copy_if(background.begin(), background.end(), std::back_inserter(slipperySouth), [](cell c) {
      return c.type == Type::SlipperySouth;
  });
  std::copy_if(background.begin(), background.end(), std::back_inserter(slipperyWest), [](cell c) {
      return c.type == Type::SlipperyWest;
  });
  std::copy_if(gridCells.begin(), gridCells.end(), std::back_inserter(oneWayNorth), [](cell c) {
      return c.type == Type::OneWayNorth;
  });
  std::copy_if(gridCells.begin(), gridCells.end(), std::back_inserter(oneWayEast), [](cell c) {
      return c.type == Type::OneWayEast;
  });
  std::copy_if(gridCells.begin(), gridCells.end(), std::back_inserter(oneWaySouth), [](cell c) {
      return c.type == Type::OneWaySouth;
  });
  std::copy_if(gridCells.begin(), gridCells.end(), std::back_inserter(oneWayWest), [](cell c) {
      return c.type == Type::OneWayWest;
  });
  std::copy_if(gridCells.begin(), gridCells.end(), std::back_inserter(lockedDoors), [](cell c) {
      return c.type == Type::LockedDoor;
  });
  std::copy_if(gridCells.begin(), gridCells.end(), std::back_inserter(goals), [](cell c) {
      return c.type == Type::Goal;
  });
  std::copy_if(gridCells.begin(), gridCells.end(), std::back_inserter(keys), [](cell c) {
      return c.type == Type::Key;
  });
  std::copy_if(gridCells.begin(), gridCells.end(), std::back_inserter(boxes), [](cell c) {
      return c.type == Type::Box;
  });
  agent = *std::find_if(gridCells.begin(), gridCells.end(), [](cell c) {
      return c.type == Type::Agent;
  });
  std::copy_if(gridCells.begin(), gridCells.end(), std::back_inserter(adversaries), [](cell c) {
      return c.type == Type::Adversary;
  });
  agentNameAndPositionMap.insert({ "Agent", agent.getCoordinates() });
  for(auto const& adversary : adversaries) {
    std::string color = adversary.getColor();
    color.at(0) = std::toupper(color.at(0));
    try {
      if(gridOptions.agentsToBeConsidered.size() != 0 && std::find(gridOptions.agentsToBeConsidered.begin(), gridOptions.agentsToBeConsidered.end(), color) == gridOptions.agentsToBeConsidered.end()) continue;
      auto success = agentNameAndPositionMap.insert({ color, adversary.getCoordinates() });
      if(!success.second) {
        throw std::logic_error("Agent with " + color + " already present\n");
      }
    } catch(const std::logic_error& e) {
      std::cerr << "Expected agents colors to be different. Agent with color : '" << color << "' already present." << std::endl;
      throw;
    }
  }
  for(auto const& color : allColors) {
    cells cellsOfColor;
    std::copy_if(background.begin(), background.end(), std::back_inserter(cellsOfColor), [&color](cell c) {
        return c.type == Type::Floor && c.color == color;
    });
    if(cellsOfColor.size() > 0) {
      backgroundTiles.emplace(color, cellsOfColor);
    }
  }
}

std::ostream& operator<<(std::ostream& os, const Grid& grid) {
  int lastRow = 1;
  for(auto const& cell : grid.allGridCells) {
    if(lastRow != cell.row)
      os << std::endl;
    os << static_cast<char>(cell.type) << static_cast<char>(cell.color);
    lastRow = cell.row;
  }
  return os;
}

cells Grid::getGridCells() {
  return allGridCells;
}

bool Grid::isBlocked(coordinates p) {
  return isWall(p) || isLockedDoor(p) || isKey(p);
}

bool Grid::isWall(coordinates p) {
  return std::find_if(walls.begin(), walls.end(),
      [p](cell cell) {
        return cell.row == p.first && cell.column == p.second;
      }) != walls.end();
}

bool Grid::isLockedDoor(coordinates p) {
  return std::find_if(lockedDoors.begin(), lockedDoors.end(),
      [p](cell cell) {
        return cell.row == p.first && cell.column == p.second;
      }) != lockedDoors.end();
}

bool Grid::isKey(coordinates p) {
  return std::find_if(keys.begin(), keys.end(),
      [p](cell cell) {
        return cell.row == p.first && cell.column == p.second;
      }) != keys.end();
}

bool Grid::isBox(coordinates p) {
  return std::find_if(boxes.begin(), boxes.end(),
      [p](cell cell) {
        return cell.row == p.first && cell.column == p.second;
      }) != boxes.end();
}

void Grid::printToPrism(std::ostream& os, const prism::ModelType& modelType) {
  cells northRestriction;
  cells eastRestriction;
  cells southRestriction;
  cells westRestriction;
  cells walkable = floor;
  walkable.insert(walkable.end(), goals.begin(), goals.end());
  walkable.insert(walkable.end(), boxes.begin(), boxes.end());
  walkable.push_back(agent);
  walkable.insert(walkable.end(), adversaries.begin(), adversaries.end());
  walkable.insert(walkable.end(), lava.begin(), lava.end());
  walkable.insert(walkable.end(), oneWayNorth.begin(), oneWayNorth.end());
  walkable.insert(walkable.end(), oneWaySouth.begin(), oneWaySouth.end());
  walkable.insert(walkable.end(), oneWayEast.begin(), oneWayEast.end());
  walkable.insert(walkable.end(), oneWayWest.begin(), oneWayWest.end());
  //for(auto const w : walkable) {
  //  std::cout << w << "\t";
  //}
  //std::cout << std::endl;
  for(auto const& c : walkable) {
    if(isBlocked(c.getNorth())) northRestriction.push_back(c);
    if(isBlocked(c.getEast()))   eastRestriction.push_back(c);
    if(isBlocked(c.getSouth())) southRestriction.push_back(c);
    if(isBlocked(c.getWest()))   westRestriction.push_back(c);
  }
  //std::cout << std::endl;
  if(gridOptions.enforceOneWays) {
    for(auto const& c : walkable) {
      if(c.getNorth(allGridCells).type == Type::OneWayNorth) { /*std::cout << c << "  one way north\n"; */ northRestriction.push_back(c); }
      if(c.getEast(allGridCells).type == Type::OneWayEast)   { /*std::cout << c << "  one way east\n";  */ eastRestriction.push_back(c); }
      if(c.getSouth(allGridCells).type == Type::OneWaySouth) { /*std::cout << c << "  one way south\n"; */ southRestriction.push_back(c); }
      if(c.getWest(allGridCells).type == Type::OneWayWest)   { /*std::cout << c << "  one way west\n";  */ westRestriction.push_back(c); }
    }
  }

  prism::PrismModulesPrinter printer(modelType, agentNameAndPositionMap.size(), gridOptions.enforceOneWays);
  printer.printModel(os, modelType);
  if(modelType == prism::ModelType::SMG) {
    printer.printGlobalMoveVariable(os, agentNameAndPositionMap.size());
  }
  for(auto const &backgroundTilesOfColor : backgroundTiles) {
    for(auto agentNameAndPosition = agentNameAndPositionMap.begin(); agentNameAndPosition != agentNameAndPositionMap.end(); ++agentNameAndPosition) {
      printer.printBackgroundLabels(os, agentNameAndPosition->first, backgroundTilesOfColor);
    }
  }
  cells noTurnFloorNorthSouth;
  cells noTurnFloorEastWest;
  if(gridOptions.enforceOneWays) {
    for(auto const& c : walkable) {
      cell east =  c.getEast(allGridCells);
      cell south = c.getSouth(allGridCells);
      cell west =  c.getWest(allGridCells);
      cell north = c.getNorth(allGridCells);
      if(east.type == Type::Wall && west.type == Type::Wall) {
        noTurnFloorNorthSouth.push_back(c);
      } else if (north.type == Type::Wall && south.type == Type::Wall) {
        noTurnFloorEastWest.push_back(c);
      }
    }
  }

  for(auto agentNameAndPosition = agentNameAndPositionMap.begin(); agentNameAndPosition != agentNameAndPositionMap.end(); ++agentNameAndPosition) {
    printer.printFormulas(os, agentNameAndPosition->first, northRestriction, eastRestriction, southRestriction, westRestriction, { slipperyNorth, slipperyEast, slipperySouth, slipperyWest }, lava, walls, noTurnFloorNorthSouth, noTurnFloorEastWest, slipperyNorth, slipperyEast, slipperySouth, slipperyWest, oneWayNorth, oneWayEast, oneWaySouth, oneWayWest);
    printer.printGoalLabel(os, agentNameAndPosition->first, goals);
    printer.printKeysLabels(os, agentNameAndPosition->first, keys);
  }

  std::vector<AgentName> agentNames;
  std::transform(agentNameAndPositionMap.begin(),
                 agentNameAndPositionMap.end(),
                 std::back_inserter(agentNames),
                 [](const std::map<AgentNameAndPosition::first_type,AgentNameAndPosition::second_type>::value_type &pair){return pair.first;});
  if(modelType == prism::ModelType::SMG) {
    printer.printCrashLabel(os, agentNames);
  }
  size_t agentIndex  = 0;
  for(auto agentNameAndPosition = agentNameAndPositionMap.begin(); agentNameAndPosition != agentNameAndPositionMap.end(); ++agentNameAndPosition, agentIndex++) {
    AgentName agentName = agentNameAndPosition->first;
    bool agentWithView = std::find(gridOptions.agentsWithView.begin(), gridOptions.agentsWithView.end(), agentName) != gridOptions.agentsWithView.end();
    bool agentWithProbabilisticBehaviour = std::find(gridOptions.agentsWithProbabilisticBehaviour.begin(), gridOptions.agentsWithProbabilisticBehaviour.end(), agentName) != gridOptions.agentsWithProbabilisticBehaviour.end();
    std::set<std::string> slipperyActions;
    printer.printInitStruct(os, agentName);
    if(agentWithProbabilisticBehaviour) printer.printModule(os, agentName, agentIndex, maxBoundaries, agentNameAndPosition->second, keys, backgroundTiles, agentWithView, gridOptions.probabilitiesForActions);
    else                                printer.printModule(os, agentName, agentIndex, maxBoundaries, agentNameAndPosition->second, keys, backgroundTiles, agentWithView);
    for(auto const& c : slipperyNorth) {
      printer.printSlipperyMove(os, agentName, agentIndex, c.getCoordinates(), slipperyActions, getWalkableDirOf8Neighborhood(c), prism::PrismModulesPrinter::SlipperyType::North);
      if(!gridOptions.enforceOneWays) printer.printSlipperyTurn(os, agentName, agentIndex, c.getCoordinates(), slipperyActions, getWalkableDirOf8Neighborhood(c), prism::PrismModulesPrinter::SlipperyType::North);
    }
    for(auto const& c : slipperyEast) {
      printer.printSlipperyMove(os, agentName, agentIndex, c.getCoordinates(), slipperyActions, getWalkableDirOf8Neighborhood(c), prism::PrismModulesPrinter::SlipperyType::East);
      if(!gridOptions.enforceOneWays) printer.printSlipperyTurn(os, agentName, agentIndex, c.getCoordinates(), slipperyActions, getWalkableDirOf8Neighborhood(c), prism::PrismModulesPrinter::SlipperyType::East);
    }
    for(auto const& c : slipperySouth) {
      printer.printSlipperyMove(os, agentName, agentIndex, c.getCoordinates(), slipperyActions, getWalkableDirOf8Neighborhood(c), prism::PrismModulesPrinter::SlipperyType::South);
      if(!gridOptions.enforceOneWays) printer.printSlipperyTurn(os, agentName, agentIndex, c.getCoordinates(), slipperyActions, getWalkableDirOf8Neighborhood(c), prism::PrismModulesPrinter::SlipperyType::South);
    }
    for(auto const& c : slipperyWest) {
      printer.printSlipperyMove(os, agentName, agentIndex, c.getCoordinates(), slipperyActions, getWalkableDirOf8Neighborhood(c), prism::PrismModulesPrinter::SlipperyType::West);
      if(!gridOptions.enforceOneWays) printer.printSlipperyTurn(os, agentName, agentIndex, c.getCoordinates(), slipperyActions, getWalkableDirOf8Neighborhood(c), prism::PrismModulesPrinter::SlipperyType::West);
    }

    printer.printEndmodule(os);
    if(modelType == prism::ModelType::SMG) {
      if(agentWithProbabilisticBehaviour) printer.printPlayerStruct(os, agentNameAndPosition->first, agentWithView, gridOptions.probabilitiesForActions, slipperyActions);
      else                                printer.printPlayerStruct(os, agentNameAndPosition->first, agentWithView, {}, slipperyActions);
    }
    //if(!stateRewards.empty()) {
    printer.printRewards(os, agentName, stateRewards, lava, goals, backgroundTiles);
    //}
  }
}


std::array<bool, 8> Grid::getWalkableDirOf8Neighborhood(cell c) /* const */ {
  return (std::array<bool, 8>)
         {
           !isBlocked(c.getNorth()),
           !isBlocked(c.getNorth(allGridCells).getEast()),
           !isBlocked(c.getEast()),
           !isBlocked(c.getSouth(allGridCells).getEast()),
           !isBlocked(c.getSouth()),
           !isBlocked(c.getSouth(allGridCells).getWest()),
           !isBlocked(c.getWest()),
           !isBlocked(c.getNorth(allGridCells).getWest())
         };
}
