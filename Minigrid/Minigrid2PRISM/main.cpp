#include "util/OptionParser.h"
#include "util/MinigridGrammar.h"
#include "util/Grid.h"

#include <iostream>
#include <fstream>
#include <filesystem>
#include <sstream>

std::vector<std::string> parseCommaSeparatedString(std::string const& str) {
  std::vector<std::string> result;
  std::stringstream stream(str);
  while(stream.good()) {
    std::string substr;
    getline(stream, substr, ',');
    substr.at(0) = std::toupper(substr.at(0));
    result.push_back(substr);
  }
  return result;
}

struct printer {
    typedef boost::spirit::utf8_string string;

    void element(string const& tag, string const& value, int depth) const {
        for (int i = 0; i < (depth*4); ++i) std::cout << ' ';

        std::cout << "tag: " << tag;
        if (value != "") std::cout << ", value: " << value;
        std::cout << std::endl;
    }
};

void print_info(boost::spirit::info const& what) {
  using boost::spirit::basic_info_walker;

  printer pr;
  basic_info_walker<printer> walker(pr, what.tag, 0);
  boost::apply_visitor(walker, what.value);
}

int main(int argc, char* argv[]) {
  popl::OptionParser optionParser("Allowed options");

  auto helpOption = optionParser.add<popl::Switch>("h", "help", "Print this help message.");
  auto inputFilename = optionParser.add<popl::Value<std::string>>("i", "input-file", "Filename of the input file.");
  auto outputFilename = optionParser.add<popl::Value<std::string>>("o", "output-file", "Filename for the output file.");

  auto agentsToBeConsidered = optionParser.add<popl::Value<std::string>, popl::Attribute::optional>("a", "agents", "Which parsed agents should be considered in the output. WIP.");
  auto viewForAgents = optionParser.add<popl::Value<std::string>, popl::Attribute::optional>("v", "view", "Agents for which the 'view'('direction') variable should be included. WIP.");

  auto probabilisticBehaviourAgents = optionParser.add<popl::Value<std::string>, popl::Attribute::optional>("p", "prob-beh", "Agents for which we want to include probabilistic actions");
  auto probabilities = optionParser.add<popl::Value<std::string>, popl::Attribute::optional>("q", "probs", "The probabilities for which probabilistic actions should be added. WIP");

  auto enforceOneWays = optionParser.add<popl::Switch>("f", "force-oneways", "Enforce encoding of oneways. This entails that slippery tiles do not allow turning and have prob 1 to shift the agent by one and makes turning impossible if in a one tile wide section of the grid.");

  try {
    optionParser.parse(argc, argv);

    if(helpOption->count() > 0) {
      std::cout << optionParser << std::endl;
      return EXIT_SUCCESS;
    }
  } catch (const popl::invalid_option &e) {
    return io::printPoplException(e);
  } catch (const std::exception &e) {
		std::cerr << "Exception: " << e.what() << "\n";
		return EXIT_FAILURE;
	}

  GridOptions gridOptions = { {}, {} };
  if(agentsToBeConsidered->is_set()) {
    gridOptions.agentsToBeConsidered = parseCommaSeparatedString(agentsToBeConsidered->value(0));
  }
  if(viewForAgents->is_set()) {
    gridOptions.agentsWithView = parseCommaSeparatedString(viewForAgents->value(0));
  }
  if(enforceOneWays->is_set()) {
    gridOptions.enforceOneWays = true;
  }
  if(probabilisticBehaviourAgents->is_set()) {
    gridOptions.agentsWithProbabilisticBehaviour = parseCommaSeparatedString(probabilisticBehaviourAgents->value(0));
    for(auto const& a : gridOptions.agentsWithProbabilisticBehaviour) {
      std::cout << a << std::endl;
    }
    if(probabilities->is_set()) {
      std::vector<std::string> parsedStrings = parseCommaSeparatedString(probabilities->value(0));

      std::transform(parsedStrings.begin(), parsedStrings.end(), std::back_inserter(gridOptions.probabilitiesForActions), [](const std::string& string) {
        return std::stof(string);
      });
      for(auto const& a : gridOptions.probabilitiesForActions) {
        std::cout << a << std::endl;
      }
    } else {
      throw std::logic_error{ "When adding agents with probabilistic behaviour, you also need to specify a list of probabilities via --probs." };
    }
  }


  std::fstream file {outputFilename->value(0), file.trunc | file.out};
  std::fstream infile {inputFilename->value(0), infile.in};
  std::string line, content, background, rewards;
  std::cout << "\n";
  bool parsingBackground = false;
  bool parsingStateRewards = false;
  while (std::getline(infile, line) && !line.empty()) {
    if(line.at(0) == '-' && line.at(line.size() - 1) == '-' && parsingBackground) {
      parsingStateRewards = true;
      parsingBackground = false;
      continue;
    } else if(line.at(0) == '-' && line.at(line.size() - 1) == '-') {
      parsingBackground = true;
      continue;
    }
    if(!parsingBackground && !parsingStateRewards) {
      std::cout << "Reading   :\t" << line << "\n";
      content += line + "\n";
    } else if (parsingBackground) {
      std::cout << "Background:\t" << line << "\n";
      background += line + "\n";
    } else if(parsingStateRewards) {
      rewards += line + "\n";
    }
  }
  std::cout << "\n";


  pos_iterator_t contentFirst(content.begin());
  pos_iterator_t contentIter = contentFirst;
  pos_iterator_t contentLast(content.end());
  MinigridParser<pos_iterator_t> contentParser(contentFirst);
  pos_iterator_t backgroundFirst(background.begin());
  pos_iterator_t backgroundIter = backgroundFirst;
  pos_iterator_t backgroundLast(background.end());
  MinigridParser<pos_iterator_t> backgroundParser(backgroundFirst);

  cells contentCells;
  cells backgroundCells;
  std::map<coordinates, float> stateRewards;
  try {
    bool ok = phrase_parse(contentIter, contentLast, contentParser, qi::space, contentCells);
    // TODO if(background is not empty) {
    ok     &= phrase_parse(backgroundIter, backgroundLast, backgroundParser, qi::space, backgroundCells);
    // TODO }

    boost::escaped_list_separator<char> seps('\\', ';', '\n');
    Tokenizer csvParser(rewards, seps);
    for(auto iter = csvParser.begin(); iter != csvParser.end(); ++iter) {
      int x = std::stoi(*iter);
      int y = std::stoi(*(++iter));
      float reward = std::stof(*(++iter));
      stateRewards[std::make_pair(x,y)] = reward;
    }
    if(ok) {
      Grid grid(contentCells, backgroundCells, gridOptions, stateRewards);
      //grid.printToPrism(std::cout, prism::ModelType::MDP);
      grid.printToPrism(file, prism::ModelType::MDP);
    }
  } catch(qi::expectation_failure<pos_iterator_t> const& e) {
    std::cout << "expected: "; print_info(e.what_);
    std::cout << "got: \"" << std::string(e.first, e.last) << '"' << std::endl;
    std::cout << "Expectation failure: " << e.what() << " at '" << std::string(e.first,e.last) << "'\n";
  } catch(const std::exception& e) {
    std::cerr << "Exception '" << typeid(e).name() << "' caught:" << std::endl;
    std::cerr << "\t" << e.what() << std::endl;
    std::exit(EXIT_FAILURE);
  }

  return 0;
}
