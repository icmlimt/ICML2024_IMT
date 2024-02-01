#pragma once

#include <iostream>
#include <functional>
#include "MinigridGrammar.h"
#include "PrismPrinter.h"

namespace prism {
  class PrismModulesPrinter {
    public:
      PrismModulesPrinter(const ModelType &modelType, const size_t &numberOfPlayer, const bool enforceOneWays = false);

      std::ostream& printRestrictionFormula(std::ostream& os, const AgentName &agentName, const std::string &direction, const cells &cells);
      std::ostream& printIsOnSlipperyFormula(std::ostream& os, const AgentName &agentName, const std::vector<std::reference_wrapper<cells>> &slipperyCollection, const cells &slipperyNorth, const cells &slipperyEast, const cells &slipperySouth, const cells &slipperyWest);
      std::ostream& printIsOnWindyFormula(std::ostream& os, const AgentName &agentName, const cells &windyNorth, const cells &windyEast, const cells &windySouth, const cells &windyWest);
      std::ostream& printGoalLabel(std::ostream& os, const AgentName&agentName, const cells &goals);
      std::ostream& printCrashLabel(std::ostream &os, const std::vector<AgentName> agentNames);
      std::ostream& printAvoidanceLabel(std::ostream &os, const std::vector<AgentName> agentNames, const int &distance);
      std::ostream& printKeysLabels(std::ostream& os, const AgentName&agentName, const cells &keys);
      std::ostream& printBackgroundLabels(std::ostream &os, const AgentName &agentName, const std::pair<Color, cells> &backgroundTiles);
      std::ostream& printIsInLavaFormula(std::ostream& os, const AgentName &agentName, const cells &lava);
      std::ostream& printIsFixedFormulas(std::ostream& os, const AgentName &agentName);
      std::ostream& printTurningNotAllowedFormulas(std::ostream& os, const AgentName &agentName, const cells &noTurnFloorNorthSouth, const cells &noTurnFloorEastWest);
      std::ostream& printOneWayFormulas(std::ostream& os, const AgentName &agentName, const cells &oneWayNorth, const cells &oneWayEast, const cells &oneWaySouth, const cells &oneWayWest);
      std::ostream& printWallFormula(std::ostream& os, const AgentName &agentName, const cells &walls);
      std::ostream& printFormulas(std::ostream& os,
                                  const AgentName&agentName,
                                  const cells &restrictionNorth,
                                  const cells &restrictionEast,
                                  const cells &restrictionSouth,
                                  const cells &restrictionWest,
                                  const std::vector<std::reference_wrapper<cells>> &slipperyCollection,
                                  const cells &lava,
                                  const cells &walls,
                                  const cells &noTurnFloorNorthSouth,
                                  const cells &noTurnFloorEastWest,
                                  const cells &slipperyNorth,
                                  const cells &slipperyEast,
                                  const cells &slipperySouth,
                                  const cells &slipperyWest,
                                  const cells &oneWayNorth,
                                  const cells &oneWayEast,
                                  const cells &oneWaySouth,
                                  const cells &oneWayWest);

       /*
        * Representation for Slippery Tile.
        *  -) North: Slips from North to South
        *  -) East: Slips from East to West
        *  -) South: Slips from South to North
        *  -) West: Slips from West to East
        */
      enum class SlipperyType { North, East, South, West };

      /*
       * Prints Slippery on move action.
       *
       * @param neighborhood: Information of wall-blocks in 8-neighborhood { n, nw, e, se, s, sw, w, nw }. If entry is false, then corresponding neighboorhood position is a wall.
       * @param orientation: Information of slippery type (either north, south, east, west).
       */
      std::ostream& printSlipperyMove(std::ostream &os, const AgentName &agentName, const size_t &agentIndex, const coordinates &c, std::set<std::string> &slipperyActions, const std::array<bool, 8>& neighborhood, SlipperyType orientation);

      /*
       * Prints Slippery on turn action.
       *
       * @param neighborhood: Information of wall-blocks in 8-neighborhood { n, nw, e, se, s, sw, w, nw }. If entry is false, then corresponding neighboorhood position is a wall.
       * @param orientation: Information of slippery type (either north, south, east, west).
       */
      std::ostream& printSlipperyTurn(std::ostream &os, const AgentName &agentName, const size_t &agentIndex, const coordinates &c, std::set<std::string> &slipperyActions, const std::array<bool, 8>& neighborhood, SlipperyType orientation);

      std::ostream& printGenericWindyMove(std::ostream &os, const AgentName &agentName, const size_t &agentIndex, const float &noise);
      std::ostream& printWindyMove(std::ostream &os, const AgentName &agentName, const size_t &agentIndex, const coordinates &c, const std::array<bool, 8>& neighborhood, SlipperyType orientation);
      std::string windyMoveHelper(const AgentName &agentName, const size_t &agentIndex, const size_t &num_updates, std::array<float, 9> prob_updates, const std::array<std::string, 9> &positionTransition, const coordinates &c, const std::array<bool, 8> &neighborhood, const size_t &viewDirection, const std::string &actionName);

      std::ostream& printModel(std::ostream &os, const ModelType &modelType);
      std::ostream& printBooleansForKeys(std::ostream &os, const AgentName &agentName, const cells &keys);
      std::ostream& printActionsForKeys(std::ostream &os, const AgentName &agentName, const cells &keys);
      std::ostream& printBooleansForBackground(std::ostream &os, const AgentName &agentName, const std::map<Color, cells> &backgroundTiles);
      std::ostream& printActionsForBackground(std::ostream &os, const AgentName &agentName, const std::map<Color, cells> &backgroundTiles);
      std::ostream& printInitStruct(std::ostream &os, const AgentName &agentName);
      std::ostream& printModule(std::ostream &os, const AgentName &agentName, const size_t &agentIndex, const coordinates &boundaries, const coordinates& initialPosition, const cells &keys, const std::map<Color, cells> &backgroundTiles, const bool agentWithView, const std::vector<float> &probabilities = {});
      std::ostream& printMovementActions(std::ostream &os, const AgentName &agentName, const size_t &agentIndex, const bool agentWithView, const float &probability = 1.0);
      std::ostream& printDoneActions(std::ostream &os, const AgentName &agentName, const size_t &agentIndex);
      std::ostream& printEndmodule(std::ostream &os);
      std::ostream& printPlayerStruct(std::ostream &os, const AgentName &agentName, const bool agentWithView, const std::vector<float> &probabilities = {}, const std::set<std::string> &slipperyActions = {});
      std::ostream& printGlobalMoveVariable(std::ostream &os, const size_t &numberOfPlayer);

      std::ostream& printRewards(std::ostream &os, const AgentName &agentName, const std::map<coordinates, float> &stateRewards, const cells &lava, const cells &goals, const std::map<Color, cells> &backgroundTiles);

      std::string moveGuard(const size_t &agentIndex);
      std::string moveUpdate(const size_t &agentIndex);

      std::string viewVariable(const AgentName &agentName, const size_t &agentDirection, const bool agentWithView);

      bool isGame() const;
    private:

      ModelType const& modelType;
      const size_t numberOfPlayer;
      bool enforceOneWays;
  };
}
