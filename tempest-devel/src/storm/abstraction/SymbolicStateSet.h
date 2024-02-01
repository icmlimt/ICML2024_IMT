#pragma once

#include "storm/abstraction/StateSet.h"

#include "storm/storage/dd/DdType.h"
#include "storm/storage/dd/Bdd.h"

namespace storm {
    namespace abstraction {
        
        template<storm::dd::DdType Type>
        class SymbolicStateSet : public StateSet {
        public:
            SymbolicStateSet(storm::dd::Bdd<Type> const& states);
            
            virtual bool isSymbolic() const override;
            
            storm::dd::Bdd<Type> const& getStates() const;
            
        private:
            storm::dd::Bdd<Type> states;
        };
        
    }
}
