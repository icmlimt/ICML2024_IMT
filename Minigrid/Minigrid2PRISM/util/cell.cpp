#include "cell.h"

#include <stdexcept>

std::ostream &operator<<(std::ostream &os, const cell &c) {
  os << static_cast<char>(c.type) << static_cast<char>(c.color);
  os <<  " at (" << c.row << "," << c.column << ")";
  return os;
}

cell cell::getNorth(const std::vector<cell> &grid) const {
  auto north = std::find_if(grid.begin(), grid.end(), [this](const cell &c) {
        return this->row - 1 == c.row && this->column == c.column;
      });
  if(north == grid.end()) {
    throw std::logic_error{ "Cannot get cell north of (" + std::to_string(row) + "," + std::to_string(column) + ")"};
    std::exit(EXIT_FAILURE);
  }
  return *north;
}

cell cell::getEast(const std::vector<cell> &grid) const {
  auto east = std::find_if(grid.begin(), grid.end(), [this](const cell &c) {
        return this->row == c.row && this->column + 1 == c.column;
      });
  if(east == grid.end()) {
    throw std::logic_error{ "Cannot get cell east of (" + std::to_string(row) + "," + std::to_string(column) + ")"};
    std::exit(EXIT_FAILURE);
  }
  return *east;
}

cell cell::getSouth(const std::vector<cell> &grid) const {
  auto south = std::find_if(grid.begin(), grid.end(), [this](const cell &c) {
        return this->row + 1 == c.row && this->column == c.column;
      });
  if(south == grid.end()) {
    throw std::logic_error{ "Cannot get cell south of (" + std::to_string(row) + "," + std::to_string(column) + ")"};
    std::exit(EXIT_FAILURE);
  }
  return *south;
}

cell cell::getWest(const std::vector<cell> &grid) const {
  auto west = std::find_if(grid.begin(), grid.end(), [this](const cell &c) {
        return this->row == c.row && this->column - 1 == c.column;
      });
  if(west == grid.end()) {
    throw std::logic_error{ "Cannot get cell west of (" + std::to_string(row) + "," + std::to_string(column) + ")"};
    std::exit(EXIT_FAILURE);
  }
  return *west;
}

coordinates cell::getCoordinates() const {
  return std::make_pair(row, column);
}

std::string cell::getColor() const {
  switch(color) {
    case Color::Red:    return "red";
    case Color::Green:  return "green";
    case Color::Blue:   return "blue";
    case Color::Purple: return "purple";
    case Color::Yellow: return "yellow";
    case Color::None:   return "transparent";
    default: return "";
    //case Color::Grey   = 'G',
  }
}

std::string getColor(Color color) {
  switch(color) {
    case Color::Red:    return "red";
    case Color::Green:  return "green";
    case Color::Blue:   return "blue";
    case Color::Purple: return "purple";
    case Color::Yellow: return "yellow";
    case Color::None:   return "transparent";
    default: return "";
    //case Color::Grey   = 'G',
  }
}
