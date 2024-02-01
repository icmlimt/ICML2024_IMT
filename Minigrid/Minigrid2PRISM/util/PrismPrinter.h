#pragma once

#include <string>

#include "cell.h"

typedef std::string AgentName;
typedef std::pair<std::string, coordinates> AgentNameAndPosition;
typedef std::map<AgentNameAndPosition::first_type, AgentNameAndPosition::second_type> AgentNameAndPositionMap;

namespace prism {
  enum class ModelType {
    MDP, SMG
  };
}
