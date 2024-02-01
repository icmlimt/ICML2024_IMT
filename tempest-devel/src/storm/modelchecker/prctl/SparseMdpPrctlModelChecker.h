#ifndef STORM_MODELCHECKER_SPARSEMDPPRCTLMODELCHECKER_H_
#define STORM_MODELCHECKER_SPARSEMDPPRCTLMODELCHECKER_H_

#include "storm/modelchecker/propositional/SparsePropositionalModelChecker.h"
#include "storm/models/sparse/Mdp.h"
#include "storm/solver/MinMaxLinearEquationSolver.h"


namespace storm {
    
    class Environment;
    
    namespace modelchecker {
        template<class SparseMdpModelType>
        class SparseMdpPrctlModelChecker : public SparsePropositionalModelChecker<SparseMdpModelType> {
        public:
            typedef typename SparseMdpModelType::ValueType ValueType;
            typedef typename SparseMdpModelType::RewardModelType RewardModelType;
            
            explicit SparseMdpPrctlModelChecker(SparseMdpModelType const& model);
            
            /*!
             * Returns false, if this task can certainly not be handled by this model checker (independent of the concrete model).
             * @param requiresSingleInitialState if not nullptr, this flag is set to true iff checking this formula requires a model with a single initial state
             */
            static bool canHandleStatic(CheckTask<storm::logic::Formula, ValueType> const& checkTask, bool* requiresSingleInitialState = nullptr);
            
            // The implemented methods of the AbstractModelChecker interface.
            virtual bool canHandle(CheckTask<storm::logic::Formula, ValueType> const& checkTask) const override;
            virtual std::unique_ptr<CheckResult> computeBoundedUntilProbabilities(Environment const& env, CheckTask<storm::logic::BoundedUntilFormula, ValueType> const& checkTask) override;
            virtual std::unique_ptr<CheckResult> computeNextProbabilities(Environment const& env, CheckTask<storm::logic::NextFormula, ValueType> const& checkTask) override;
            virtual std::unique_ptr<CheckResult> computeUntilProbabilities(Environment const& env, CheckTask<storm::logic::UntilFormula, ValueType> const& checkTask) override;
            virtual std::unique_ptr<CheckResult> computeGloballyProbabilities(Environment const& env, CheckTask<storm::logic::GloballyFormula, ValueType> const& checkTask) override;
            virtual std::unique_ptr<CheckResult> computeConditionalProbabilities(Environment const& env, CheckTask<storm::logic::ConditionalFormula, ValueType> const& checkTask) override;
            virtual std::unique_ptr<CheckResult> computeCumulativeRewards(Environment const& env, storm::logic::RewardMeasureType rewardMeasureType, CheckTask<storm::logic::CumulativeRewardFormula, ValueType> const& checkTask) override;
            virtual std::unique_ptr<CheckResult> computeInstantaneousRewards(Environment const& env, storm::logic::RewardMeasureType rewardMeasureType, CheckTask<storm::logic::InstantaneousRewardFormula, ValueType> const& checkTask) override;
            virtual std::unique_ptr<CheckResult> computeTotalRewards(Environment const& env, storm::logic::RewardMeasureType rewardMeasureType, CheckTask<storm::logic::TotalRewardFormula, ValueType> const& checkTask) override;
            virtual std::unique_ptr<CheckResult> computeReachabilityRewards(Environment const& env, storm::logic::RewardMeasureType rewardMeasureType, CheckTask<storm::logic::EventuallyFormula, ValueType> const& checkTask) override;
            virtual std::unique_ptr<CheckResult> computeReachabilityTimes(Environment const& env, storm::logic::RewardMeasureType rewardMeasureType, CheckTask<storm::logic::EventuallyFormula, ValueType> const& checkTask) override;
            virtual std::unique_ptr<CheckResult> computeLongRunAverageProbabilities(Environment const& env, CheckTask<storm::logic::StateFormula, ValueType> const& checkTask) override;
            virtual std::unique_ptr<CheckResult> computeLongRunAverageRewards(Environment const& env, storm::logic::RewardMeasureType rewardMeasureType, CheckTask<storm::logic::LongRunAverageRewardFormula, ValueType> const& checkTask) override;
            virtual std::unique_ptr<CheckResult> computeLTLProbabilities(Environment const& env, CheckTask<storm::logic::PathFormula, ValueType> const& checkTask) override;
            virtual std::unique_ptr<CheckResult> computeHOAPathProbabilities(Environment const& env, CheckTask<storm::logic::HOAPathFormula, ValueType> const& checkTask) override;
            virtual std::unique_ptr<CheckResult> checkMultiObjectiveFormula(Environment const& env, CheckTask<storm::logic::MultiObjectiveFormula, ValueType> const& checkTask) override;
            virtual std::unique_ptr<CheckResult> checkQuantileFormula(Environment const& env, CheckTask<storm::logic::QuantileFormula, ValueType> const& checkTask) override;

            // <state_action_ranking>

            /*!
             * Function to remove the MDP's action. 
             */
            std::unique_ptr<storm::models::sparse::Mdp<ValueType>> restrictMdp(
                const storm::storage::BitVector& actionsToRemove);

            /*!
             * Collection of all state-action values, where the index corresponds to the state id.
             */
            std::vector<ValueType> stateValueMapping;

            /*!
             * Foreach state: Information which action gives the minimum reward.
             * Is created during state-action ranking.
             */
            storm::storage::BitVector worstStateActions;

            /*!
             * Bitmask for worstStateActions where the corresponding stateValue is above a given threshold.
             * Is created during state-action ranking.
             */
            storm::storage::BitVector worstStateActionsBitmask;

            /*!
             * Function to set stateValueMapping - member.
             * @param choiceValues the collection of all existing (state-action)-values.
             */
            void computeStateActionRanking(const std::vector<ValueType>& choiceValues);

            void printResultsPerState(const std::vector<ValueType>& result, const storm::OptimizationDirection dir, const std::string prefix = "");
            // </state_action_ranking>
        };
    } // namespace modelchecker
} // namespace storm

#endif /* STORM_MODELCHECKER_SPARSEMDPPRCTLMODELCHECKER_H_ */
