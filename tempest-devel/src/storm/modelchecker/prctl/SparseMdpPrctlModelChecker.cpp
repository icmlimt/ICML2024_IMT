#include "storm/modelchecker/prctl/SparseMdpPrctlModelChecker.h"

#include "storm/utility/constants.h"
#include "storm/utility/macros.h"
#include "storm/utility/vector.h"
#include "storm/utility/graph.h"
#include "storm/utility/FilteredRewardModel.h"

#include "storm/modelchecker/results/ExplicitQualitativeCheckResult.h"
#include "storm/modelchecker/results/ExplicitQuantitativeCheckResult.h"
#include "storm/modelchecker/results/ExplicitParetoCurveCheckResult.h"

#include "storm/logic/FragmentSpecification.h"

#include "storm/models/sparse/StandardRewardModel.h"

#include "storm/modelchecker/prctl/helper/SparseMdpPrctlHelper.h"
#include "storm/modelchecker/helper/infinitehorizon/SparseNondeterministicInfiniteHorizonHelper.h"
#include "storm/modelchecker/helper/finitehorizon/SparseNondeterministicStepBoundedHorizonHelper.h"
#include "storm/modelchecker/helper/ltl/SparseLTLHelper.h"
#include "storm/modelchecker/helper/utility/SetInformationFromCheckTask.h"

#include "storm/modelchecker/prctl/helper/rewardbounded/QuantileHelper.h"
#include "storm/modelchecker/multiobjective/multiObjectiveModelChecking.h"

#include "storm/solver/SolveGoal.h"
#include "storm/storage/BitVector.h"

#include "storm/shields/ShieldHandling.h"

#include "storm/settings/modules/GeneralSettings.h"
#include "storm/exceptions/InvalidStateException.h"
#include "storm/exceptions/InvalidPropertyException.h"
#include "storm/storage/expressions/Expressions.h"

#include "storm/exceptions/InvalidPropertyException.h"

namespace storm {
    namespace modelchecker {
        template<typename SparseMdpModelType>
        SparseMdpPrctlModelChecker<SparseMdpModelType>::SparseMdpPrctlModelChecker(SparseMdpModelType const& model) : SparsePropositionalModelChecker<SparseMdpModelType>(model) {
            // Intentionally left empty.
        }

        template<typename SparseMdpModelType>
        bool SparseMdpPrctlModelChecker<SparseMdpModelType>::canHandleStatic(CheckTask<storm::logic::Formula, ValueType> const& checkTask, bool* requiresSingleInitialState) {
            storm::logic::Formula const& formula = checkTask.getFormula();
            if (formula.isInFragment(storm::logic::prctlstar().setLongRunAverageRewardFormulasAllowed(true).setLongRunAverageProbabilitiesAllowed(true).setConditionalProbabilityFormulasAllowed(true).setOnlyEventuallyFormuluasInConditionalFormulasAllowed(true).setTotalRewardFormulasAllowed(true).setRewardBoundedUntilFormulasAllowed(true).setRewardBoundedCumulativeRewardFormulasAllowed(true).setMultiDimensionalBoundedUntilFormulasAllowed(true).setMultiDimensionalCumulativeRewardFormulasAllowed(true).setTimeOperatorsAllowed(true).setReachbilityTimeFormulasAllowed(true).setRewardAccumulationAllowed(true))) {
                return true;
            } else if (checkTask.isOnlyInitialStatesRelevantSet()) {
                auto multiObjectiveFragment = storm::logic::multiObjective().setCumulativeRewardFormulasAllowed(true).setTimeBoundedCumulativeRewardFormulasAllowed(true).setStepBoundedCumulativeRewardFormulasAllowed(true).setRewardBoundedCumulativeRewardFormulasAllowed(true).setTimeBoundedUntilFormulasAllowed(true).setStepBoundedUntilFormulasAllowed(true).setRewardBoundedUntilFormulasAllowed(true).setMultiDimensionalBoundedUntilFormulasAllowed(true).setMultiDimensionalCumulativeRewardFormulasAllowed(true).setRewardAccumulationAllowed(true);
                if (formula.isInFragment(multiObjectiveFragment) || formula.isInFragment(storm::logic::quantiles())) {
                    if (requiresSingleInitialState) {
                        *requiresSingleInitialState = true;
                    }
                    return true;
                }
            }
            return false;
        }

        template<typename SparseMdpModelType>
        bool SparseMdpPrctlModelChecker<SparseMdpModelType>::canHandle(CheckTask<storm::logic::Formula, ValueType> const& checkTask) const {
            bool requiresSingleInitialState = false;
            if (canHandleStatic(checkTask, &requiresSingleInitialState)) {
                return !requiresSingleInitialState || this->getModel().getInitialStates().getNumberOfSetBits() == 1;
            } else {
                return false;
            }
        }

        template<typename SparseMdpModelType>
        std::unique_ptr<CheckResult> SparseMdpPrctlModelChecker<SparseMdpModelType>::computeBoundedUntilProbabilities(Environment const& env, CheckTask<storm::logic::BoundedUntilFormula, ValueType> const& checkTask) {
            storm::logic::BoundedUntilFormula const& pathFormula = checkTask.getFormula();
            STORM_LOG_THROW(checkTask.isOptimizationDirectionSet(), storm::exceptions::InvalidPropertyException, "Formula needs to specify whether minimal or maximal values are to be computed on nondeterministic model.");
            if (pathFormula.isMultiDimensional() || pathFormula.getTimeBoundReference().isRewardBound()) {
                STORM_LOG_THROW(checkTask.isOnlyInitialStatesRelevantSet(), storm::exceptions::InvalidOperationException, "Checking non-trivial bounded until probabilities can only be computed for the initial states of the model.");
                STORM_LOG_WARN_COND(!checkTask.isQualitativeSet(), "Checking non-trivial bounded until formulas is not optimized w.r.t. qualitative queries");
                storm::logic::OperatorInformation opInfo(checkTask.getOptimizationDirection());
                if (checkTask.isBoundSet()) {
                    opInfo.bound = checkTask.getBound();
                }
                auto formula = std::make_shared<storm::logic::ProbabilityOperatorFormula>(checkTask.getFormula().asSharedPointer(), opInfo);
                helper::rewardbounded::MultiDimensionalRewardUnfolding<ValueType, true> rewardUnfolding(this->getModel(), formula);
                auto numericResult = storm::modelchecker::helper::SparseMdpPrctlHelper<ValueType>::computeRewardBoundedValues(env, checkTask.getOptimizationDirection(), rewardUnfolding, this->getModel().getInitialStates());
                return std::unique_ptr<CheckResult>(new ExplicitQuantitativeCheckResult<ValueType>(std::move(numericResult)));
            } else {
                STORM_LOG_THROW(pathFormula.hasUpperBound(), storm::exceptions::InvalidPropertyException, "Formula needs to have (a single) upper step bound.");
                STORM_LOG_THROW(pathFormula.hasIntegerLowerBound(), storm::exceptions::InvalidPropertyException, "Formula lower step bound must be discrete/integral.");
                STORM_LOG_THROW(pathFormula.hasIntegerUpperBound(), storm::exceptions::InvalidPropertyException, "Formula needs to have discrete upper time bound.");
                std::unique_ptr<CheckResult> leftResultPointer = this->check(env, pathFormula.getLeftSubformula());
                std::unique_ptr<CheckResult> rightResultPointer = this->check(env, pathFormula.getRightSubformula());
                ExplicitQualitativeCheckResult const& leftResult = leftResultPointer->asExplicitQualitativeCheckResult();
                ExplicitQualitativeCheckResult const& rightResult = rightResultPointer->asExplicitQualitativeCheckResult();
                storm::modelchecker::helper::SparseNondeterministicStepBoundedHorizonHelper<ValueType> helper;
                std::vector<ValueType> numericResult;

                //This works only with empty vectors, no nullptr
                storm::storage::BitVector resultMaybeStates;
                std::vector<ValueType> choiceValues;

                numericResult = helper.compute(env, storm::solver::SolveGoal<ValueType>(this->getModel(), checkTask), this->getModel().getTransitionMatrix(), this->getModel().getBackwardTransitions(), leftResult.getTruthValuesVector(), rightResult.getTruthValuesVector(), pathFormula.getNonStrictLowerBound<uint64_t>(), pathFormula.getNonStrictUpperBound<uint64_t>(), resultMaybeStates, choiceValues, checkTask.getHint());

                this->printResultsPerState(numericResult, checkTask.getOptimizationDirection(), "prob_");

                if(checkTask.isShieldingTask()) {
                    tempest::shields::createShield<ValueType>(std::make_shared<storm::models::sparse::Mdp<ValueType>>(this->getModel()), std::move(choiceValues), checkTask.getShieldingExpression(), checkTask.getOptimizationDirection(), std::move(resultMaybeStates), storm::storage::BitVector(resultMaybeStates.size(), true));
                }
                return std::unique_ptr<CheckResult>(new ExplicitQuantitativeCheckResult<ValueType>(std::move(numericResult)));
            }
        }

        template<typename SparseMdpModelType>
        std::unique_ptr<CheckResult> SparseMdpPrctlModelChecker<SparseMdpModelType>::computeNextProbabilities(Environment const& env, CheckTask<storm::logic::NextFormula, ValueType> const& checkTask) {
            storm::logic::NextFormula const& pathFormula = checkTask.getFormula();
            STORM_LOG_THROW(checkTask.isOptimizationDirectionSet(), storm::exceptions::InvalidPropertyException, "Formula needs to specify whether minimal or maximal values are to be computed on nondeterministic model.");
            std::unique_ptr<CheckResult> subResultPointer = this->check(env, pathFormula.getSubformula());
            ExplicitQualitativeCheckResult const& subResult = subResultPointer->asExplicitQualitativeCheckResult();
            auto ret = storm::modelchecker::helper::SparseMdpPrctlHelper<ValueType>::computeNextProbabilities(env, storm::solver::SolveGoal<ValueType>(this->getModel(), checkTask), checkTask.getOptimizationDirection(), this->getModel().getTransitionMatrix(), subResult.getTruthValuesVector());
            std::unique_ptr<CheckResult> result(new ExplicitQuantitativeCheckResult<ValueType>(std::move(ret.values)));
            if(checkTask.isShieldingTask()) {
                tempest::shields::createShield<ValueType>(std::make_shared<storm::models::sparse::Mdp<ValueType>>(this->getModel()), std::move(ret.choiceValues), checkTask.getShieldingExpression(), checkTask.getOptimizationDirection(), std::move(ret.maybeStates), storm::storage::BitVector(ret.maybeStates.size(), true));
            } else if (checkTask.isProduceSchedulersSet() && ret.scheduler) {
                result->asExplicitQuantitativeCheckResult<ValueType>().setScheduler(std::move(ret.scheduler));
            }
            return result;
        }

        template<typename SparseMdpModelType>
        std::unique_ptr<CheckResult> SparseMdpPrctlModelChecker<SparseMdpModelType>::computeUntilProbabilities(Environment const& env, CheckTask<storm::logic::UntilFormula, ValueType> const& checkTask) {
            storm::logic::UntilFormula const& pathFormula = checkTask.getFormula();
            STORM_LOG_THROW(checkTask.isOptimizationDirectionSet(), storm::exceptions::InvalidPropertyException, "Formula needs to specify whether minimal or maximal values are to be computed on nondeterministic model.");
            std::unique_ptr<CheckResult> leftResultPointer = this->check(env, pathFormula.getLeftSubformula());
            std::unique_ptr<CheckResult> rightResultPointer = this->check(env, pathFormula.getRightSubformula());
            ExplicitQualitativeCheckResult const& leftResult = leftResultPointer->asExplicitQualitativeCheckResult();
            ExplicitQualitativeCheckResult const& rightResult = rightResultPointer->asExplicitQualitativeCheckResult();
            auto ret = storm::modelchecker::helper::SparseMdpPrctlHelper<ValueType>::computeUntilProbabilities(env, storm::solver::SolveGoal<ValueType>(this->getModel(), checkTask), this->getModel().getTransitionMatrix(), this->getModel().getBackwardTransitions(), leftResult.getTruthValuesVector(), rightResult.getTruthValuesVector(), checkTask.isQualitativeSet(), checkTask.isProduceSchedulersSet(), checkTask.getHint());
            std::unique_ptr<CheckResult> result(new ExplicitQuantitativeCheckResult<ValueType>(std::move(ret.values)));
            if(checkTask.isShieldingTask()) {
                tempest::shields::createShield<ValueType>(std::make_shared<storm::models::sparse::Mdp<ValueType>>(this->getModel()), std::move(ret.choiceValues), checkTask.getShieldingExpression(), checkTask.getOptimizationDirection(), std::move(ret.maybeStates), storm::storage::BitVector(ret.maybeStates.size(), true));
            } else if (checkTask.isProduceSchedulersSet() && ret.scheduler) {
                result->asExplicitQuantitativeCheckResult<ValueType>().setScheduler(std::move(ret.scheduler));
            }
            return result;
        }

        template<typename SparseMdpModelType>
        std::unique_ptr<CheckResult> SparseMdpPrctlModelChecker<SparseMdpModelType>::computeGloballyProbabilities(Environment const& env, CheckTask<storm::logic::GloballyFormula, ValueType> const& checkTask) {
            storm::logic::GloballyFormula const& pathFormula = checkTask.getFormula();
            STORM_LOG_THROW(checkTask.isOptimizationDirectionSet(), storm::exceptions::InvalidPropertyException, "Formula needs to specify whether minimal or maximal values are to be computed on nondeterministic model.");
            std::unique_ptr<CheckResult> subResultPointer = this->check(env, pathFormula.getSubformula());
            ExplicitQualitativeCheckResult const& subResult = subResultPointer->asExplicitQualitativeCheckResult();
            auto ret = storm::modelchecker::helper::SparseMdpPrctlHelper<ValueType>::computeGloballyProbabilities(env, storm::solver::SolveGoal<ValueType>(this->getModel(), checkTask), this->getModel().getTransitionMatrix(), this->getModel().getBackwardTransitions(), subResult.getTruthValuesVector(), checkTask.isQualitativeSet(), checkTask.isProduceSchedulersSet());
            std::unique_ptr<CheckResult> result(new ExplicitQuantitativeCheckResult<ValueType>(std::move(ret.values)));
            if(checkTask.isShieldingTask()) {
                tempest::shields::createShield<ValueType>(std::make_shared<storm::models::sparse::Mdp<ValueType>>(this->getModel()), std::move(ret.choiceValues), checkTask.getShieldingExpression(), checkTask.getOptimizationDirection(),subResult.getTruthValuesVector(), storm::storage::BitVector(ret.maybeStates.size(), true));
            } else if (checkTask.isProduceSchedulersSet() && ret.scheduler) {
                result->asExplicitQuantitativeCheckResult<ValueType>().setScheduler(std::move(ret.scheduler));
            }
            return result;
        }

        template<typename SparseMdpModelType>
        std::unique_ptr<CheckResult> SparseMdpPrctlModelChecker<SparseMdpModelType>::computeHOAPathProbabilities(Environment const& env, CheckTask<storm::logic::HOAPathFormula, ValueType> const& checkTask) {
            storm::logic::HOAPathFormula const& pathFormula = checkTask.getFormula();

            storm::modelchecker::helper::SparseLTLHelper<ValueType, true> helper(this->getModel().getTransitionMatrix());
            storm::modelchecker::helper::setInformationFromCheckTaskNondeterministic(helper, checkTask, this->getModel());

            auto formulaChecker = [&] (storm::logic::Formula const& formula) { return this->check(env, formula)->asExplicitQualitativeCheckResult().getTruthValuesVector(); };
            auto apSets = helper.computeApSets(pathFormula.getAPMapping(), formulaChecker);
            std::vector<ValueType> numericResult = helper.computeDAProductProbabilities(env, *pathFormula.readAutomaton(), apSets);

            std::unique_ptr<CheckResult> result(new ExplicitQuantitativeCheckResult<ValueType>(std::move(numericResult)));
            if (checkTask.isProduceSchedulersSet()) {
                result->asExplicitQuantitativeCheckResult<ValueType>().setScheduler(std::make_unique<storm::storage::Scheduler<ValueType>>(helper.extractScheduler(this->getModel())));
            }

            return result;
        }

        template<typename SparseMdpModelType>
        std::unique_ptr<CheckResult> SparseMdpPrctlModelChecker<SparseMdpModelType>::computeLTLProbabilities(Environment const& env, CheckTask<storm::logic::PathFormula, ValueType> const& checkTask) {
            storm::logic::PathFormula const& pathFormula = checkTask.getFormula();

            STORM_LOG_THROW(checkTask.isOptimizationDirectionSet(), storm::exceptions::InvalidPropertyException, "Formula needs to specify whether minimal or maximal values are to be computed on nondeterministic model.");

            storm::modelchecker::helper::SparseLTLHelper<ValueType, true> helper(this->getModel().getTransitionMatrix());
            storm::modelchecker::helper::setInformationFromCheckTaskNondeterministic(helper, checkTask, this->getModel());

            auto formulaChecker = [&] (storm::logic::Formula const& formula) { return this->check(env, formula)->asExplicitQualitativeCheckResult().getTruthValuesVector(); };
            std::vector<ValueType> numericResult = helper.computeLTLProbabilities(env, pathFormula, formulaChecker);

            std::unique_ptr<CheckResult> result(new ExplicitQuantitativeCheckResult<ValueType>(std::move(numericResult)));
            if (checkTask.isProduceSchedulersSet()) {
                result->asExplicitQuantitativeCheckResult<ValueType>().setScheduler(std::make_unique<storm::storage::Scheduler<ValueType>>(helper.extractScheduler(this->getModel())));
            }

            return result;
        }

        template<typename SparseMdpModelType>
        std::unique_ptr<CheckResult> SparseMdpPrctlModelChecker<SparseMdpModelType>::computeConditionalProbabilities(Environment const& env, CheckTask<storm::logic::ConditionalFormula, ValueType> const& checkTask) {
            storm::logic::ConditionalFormula const& conditionalFormula = checkTask.getFormula();
            STORM_LOG_THROW(checkTask.isOptimizationDirectionSet(), storm::exceptions::InvalidPropertyException, "Formula needs to specify whether minimal or maximal values are to be computed on nondeterministic model.");
            STORM_LOG_THROW(this->getModel().getInitialStates().getNumberOfSetBits() == 1, storm::exceptions::InvalidPropertyException, "Cannot compute conditional probabilities on MDPs with more than one initial state.");
            STORM_LOG_THROW(conditionalFormula.getSubformula().isEventuallyFormula(), storm::exceptions::InvalidPropertyException, "Illegal conditional probability formula.");
            STORM_LOG_THROW(conditionalFormula.getConditionFormula().isEventuallyFormula(), storm::exceptions::InvalidPropertyException, "Illegal conditional probability formula.");

            std::unique_ptr<CheckResult> leftResultPointer = this->check(env, conditionalFormula.getSubformula().asEventuallyFormula().getSubformula());
            std::unique_ptr<CheckResult> rightResultPointer = this->check(env, conditionalFormula.getConditionFormula().asEventuallyFormula().getSubformula());
            ExplicitQualitativeCheckResult const& leftResult = leftResultPointer->asExplicitQualitativeCheckResult();
            ExplicitQualitativeCheckResult const& rightResult = rightResultPointer->asExplicitQualitativeCheckResult();

            return storm::modelchecker::helper::SparseMdpPrctlHelper<ValueType>::computeConditionalProbabilities(env, storm::solver::SolveGoal<ValueType>(this->getModel(), checkTask), this->getModel().getTransitionMatrix(), this->getModel().getBackwardTransitions(), leftResult.getTruthValuesVector(), rightResult.getTruthValuesVector());
        }

        template<typename SparseMdpModelType>
        std::unique_ptr<CheckResult> SparseMdpPrctlModelChecker<SparseMdpModelType>::computeCumulativeRewards(Environment const& env, storm::logic::RewardMeasureType, CheckTask<storm::logic::CumulativeRewardFormula, ValueType> const& checkTask) {
            storm::logic::CumulativeRewardFormula const& rewardPathFormula = checkTask.getFormula();
            STORM_LOG_THROW(checkTask.isOptimizationDirectionSet(), storm::exceptions::InvalidPropertyException, "Formula needs to specify whether minimal or maximal values are to be computed on nondeterministic model.");
            if (rewardPathFormula.isMultiDimensional() || rewardPathFormula.getTimeBoundReference().isRewardBound()) {
                STORM_LOG_THROW(checkTask.isOnlyInitialStatesRelevantSet(), storm::exceptions::InvalidOperationException, "Checking reward bounded cumulative reward formulas can only be done for the initial states of the model.");
                STORM_LOG_THROW(!checkTask.getFormula().hasRewardAccumulation(), storm::exceptions::InvalidOperationException, "Checking reward bounded cumulative reward formulas is not supported if reward accumulations are given.");
                STORM_LOG_WARN_COND(!checkTask.isQualitativeSet(), "Checking reward bounded until formulas is not optimized w.r.t. qualitative queries");
                storm::logic::OperatorInformation opInfo(checkTask.getOptimizationDirection());
                if (checkTask.isBoundSet()) {
                    opInfo.bound = checkTask.getBound();
                }
                auto formula = std::make_shared<storm::logic::RewardOperatorFormula>(checkTask.getFormula().asSharedPointer(), checkTask.getRewardModel(), opInfo);
                helper::rewardbounded::MultiDimensionalRewardUnfolding<ValueType, true> rewardUnfolding(this->getModel(), formula);
                auto numericResult = storm::modelchecker::helper::SparseMdpPrctlHelper<ValueType>::computeRewardBoundedValues(env, checkTask.getOptimizationDirection(), rewardUnfolding, this->getModel().getInitialStates());
                return std::unique_ptr<CheckResult>(new ExplicitQuantitativeCheckResult<ValueType>(std::move(numericResult)));
            } else {
                STORM_LOG_THROW(rewardPathFormula.hasIntegerBound(), storm::exceptions::InvalidPropertyException, "Formula needs to have a discrete time bound.");
                auto rewardModel = storm::utility::createFilteredRewardModel(this->getModel(), checkTask);
                auto ret = storm::modelchecker::helper::SparseMdpPrctlHelper<ValueType>::computeCumulativeRewards(env, storm::solver::SolveGoal<ValueType>(this->getModel(), checkTask), this->getModel().getTransitionMatrix(), rewardModel.get(), rewardPathFormula.getNonStrictBound<uint64_t>());
                this->computeStateActionRanking(ret.choiceValues);
                this->printResultsPerState(ret.values, checkTask.getOptimizationDirection());
                return std::unique_ptr<CheckResult>(new ExplicitQuantitativeCheckResult<ValueType>(std::move(ret.values)));
            }
        }

        template<typename SparseMdpModelType>
        std::unique_ptr<CheckResult> SparseMdpPrctlModelChecker<SparseMdpModelType>::computeInstantaneousRewards(Environment const& env, storm::logic::RewardMeasureType, CheckTask<storm::logic::InstantaneousRewardFormula, ValueType> const& checkTask) {
            storm::logic::InstantaneousRewardFormula const& rewardPathFormula = checkTask.getFormula();
            STORM_LOG_THROW(checkTask.isOptimizationDirectionSet(), storm::exceptions::InvalidPropertyException, "Formula needs to specify whether minimal or maximal values are to be computed on nondeterministic model.");
            STORM_LOG_THROW(rewardPathFormula.hasIntegerBound(), storm::exceptions::InvalidPropertyException, "Formula needs to have a discrete time bound.");
            auto ret = storm::modelchecker::helper::SparseMdpPrctlHelper<ValueType>::computeInstantaneousRewards(env, storm::solver::SolveGoal<ValueType>(this->getModel(), checkTask), this->getModel().getTransitionMatrix(), checkTask.isRewardModelSet() ? this->getModel().getRewardModel(checkTask.getRewardModel()) : this->getModel().getRewardModel(""), rewardPathFormula.getBound<uint64_t>());
            this->computeStateActionRanking(ret.choiceValues);
            return std::unique_ptr<CheckResult>(new ExplicitQuantitativeCheckResult<ValueType>(std::move(ret.values)));
        }

        template<typename SparseMdpModelType>
        std::unique_ptr<CheckResult> SparseMdpPrctlModelChecker<SparseMdpModelType>::computeReachabilityRewards(Environment const& env, storm::logic::RewardMeasureType, CheckTask<storm::logic::EventuallyFormula, ValueType> const& checkTask) {
            storm::logic::EventuallyFormula const& eventuallyFormula = checkTask.getFormula();
            STORM_LOG_THROW(checkTask.isOptimizationDirectionSet(), storm::exceptions::InvalidPropertyException, "Formula needs to specify whether minimal or maximal values are to be computed on nondeterministic model.");
            std::unique_ptr<CheckResult> subResultPointer = this->check(env, eventuallyFormula.getSubformula());
            ExplicitQualitativeCheckResult const& subResult = subResultPointer->asExplicitQualitativeCheckResult();
            auto rewardModel = storm::utility::createFilteredRewardModel(this->getModel(), checkTask);
            auto ret = storm::modelchecker::helper::SparseMdpPrctlHelper<ValueType>::computeReachabilityRewards(env, storm::solver::SolveGoal<ValueType>(this->getModel(), checkTask), this->getModel().getTransitionMatrix(), this->getModel().getBackwardTransitions(), rewardModel.get(), subResult.getTruthValuesVector(), checkTask.isQualitativeSet(), checkTask.isProduceSchedulersSet(), checkTask.getHint());
            std::unique_ptr<CheckResult> result(new ExplicitQuantitativeCheckResult<ValueType>(std::move(ret.values)));
            if (checkTask.isProduceSchedulersSet() && ret.scheduler) {
                result->asExplicitQuantitativeCheckResult<ValueType>().setScheduler(std::move(ret.scheduler));
            }
            return result;
        }

        template<typename SparseMdpModelType>
        std::unique_ptr<CheckResult> SparseMdpPrctlModelChecker<SparseMdpModelType>::computeReachabilityTimes(Environment const& env, storm::logic::RewardMeasureType, CheckTask<storm::logic::EventuallyFormula, ValueType> const& checkTask) {
            storm::logic::EventuallyFormula const& eventuallyFormula = checkTask.getFormula();
            STORM_LOG_THROW(checkTask.isOptimizationDirectionSet(), storm::exceptions::InvalidPropertyException, "Formula needs to specify whether minimal or maximal values are to be computed on nondeterministic model.");
            std::unique_ptr<CheckResult> subResultPointer = this->check(env, eventuallyFormula.getSubformula());
            ExplicitQualitativeCheckResult const& subResult = subResultPointer->asExplicitQualitativeCheckResult();
            auto ret = storm::modelchecker::helper::SparseMdpPrctlHelper<ValueType>::computeReachabilityTimes(env, storm::solver::SolveGoal<ValueType>(this->getModel(), checkTask), this->getModel().getTransitionMatrix(), this->getModel().getBackwardTransitions(), subResult.getTruthValuesVector(), checkTask.isQualitativeSet(), checkTask.isProduceSchedulersSet(), checkTask.getHint());
            std::unique_ptr<CheckResult> result(new ExplicitQuantitativeCheckResult<ValueType>(std::move(ret.values)));
            if (checkTask.isProduceSchedulersSet() && ret.scheduler) {
                result->asExplicitQuantitativeCheckResult<ValueType>().setScheduler(std::move(ret.scheduler));
            }
            return result;
        }

        template<typename SparseMdpModelType>
        std::unique_ptr<CheckResult> SparseMdpPrctlModelChecker<SparseMdpModelType>::computeTotalRewards(Environment const& env, storm::logic::RewardMeasureType, CheckTask<storm::logic::TotalRewardFormula, ValueType> const& checkTask) {
            STORM_LOG_THROW(checkTask.isOptimizationDirectionSet(), storm::exceptions::InvalidPropertyException, "Formula needs to specify whether minimal or maximal values are to be computed on nondeterministic model.");
            auto rewardModel = storm::utility::createFilteredRewardModel(this->getModel(), checkTask);
            auto ret = storm::modelchecker::helper::SparseMdpPrctlHelper<ValueType>::computeTotalRewards(env, storm::solver::SolveGoal<ValueType>(this->getModel(), checkTask), this->getModel().getTransitionMatrix(), this->getModel().getBackwardTransitions(), rewardModel.get(), checkTask.isQualitativeSet(), checkTask.isProduceSchedulersSet(), checkTask.getHint());
            std::unique_ptr<CheckResult> result(new ExplicitQuantitativeCheckResult<ValueType>(std::move(ret.values)));
            if (checkTask.isProduceSchedulersSet() && ret.scheduler) {
                result->asExplicitQuantitativeCheckResult<ValueType>().setScheduler(std::move(ret.scheduler));
            }
            return result;
        }

		template<typename SparseMdpModelType>
		std::unique_ptr<CheckResult> SparseMdpPrctlModelChecker<SparseMdpModelType>::computeLongRunAverageProbabilities(Environment const& env, CheckTask<storm::logic::StateFormula, ValueType> const& checkTask) {
		    storm::logic::StateFormula const& stateFormula = checkTask.getFormula();
			STORM_LOG_THROW(checkTask.isOptimizationDirectionSet(), storm::exceptions::InvalidPropertyException, "Formula needs to specify whether minimal or maximal values are to be computed on nondeterministic model.");
			std::unique_ptr<CheckResult> subResultPointer = this->check(env, stateFormula);
			ExplicitQualitativeCheckResult const& subResult = subResultPointer->asExplicitQualitativeCheckResult();

			storm::modelchecker::helper::SparseNondeterministicInfiniteHorizonHelper<ValueType> helper(this->getModel().getTransitionMatrix());
            storm::modelchecker::helper::setInformationFromCheckTaskNondeterministic(helper, checkTask, this->getModel());
			auto values = helper.computeLongRunAverageProbabilities(env, subResult.getTruthValuesVector());

            std::unique_ptr<CheckResult> result(new ExplicitQuantitativeCheckResult<ValueType>(std::move(values)));
            if(checkTask.isShieldingTask()) {
                storm::storage::BitVector allStatesBv = storm::storage::BitVector(this->getModel().getTransitionMatrix().getRowGroupCount(), true);
                tempest::shields::createQuantitativeShield<ValueType>(std::make_shared<storm::models::sparse::Mdp<ValueType>>(this->getModel()), helper.getChoiceValues(), checkTask.getShieldingExpression(), checkTask.getOptimizationDirection(), allStatesBv, allStatesBv);
            } else if (checkTask.isProduceSchedulersSet()) {
                result->asExplicitQuantitativeCheckResult<ValueType>().setScheduler(std::make_unique<storm::storage::Scheduler<ValueType>>(helper.extractScheduler()));
            }
            return result;
		}

        template<typename SparseMdpModelType>
        std::unique_ptr<CheckResult> SparseMdpPrctlModelChecker<SparseMdpModelType>::computeLongRunAverageRewards(Environment const& env, storm::logic::RewardMeasureType rewardMeasureType, CheckTask<storm::logic::LongRunAverageRewardFormula, ValueType> const& checkTask) {
            STORM_LOG_THROW(checkTask.isOptimizationDirectionSet(), storm::exceptions::InvalidPropertyException, "Formula needs to specify whether minimal or maximal values are to be computed on nondeterministic model.");
            auto rewardModel = storm::utility::createFilteredRewardModel(this->getModel(), checkTask);
            storm::modelchecker::helper::SparseNondeterministicInfiniteHorizonHelper<ValueType> helper(this->getModel().getTransitionMatrix());
            storm::modelchecker::helper::setInformationFromCheckTaskNondeterministic(helper, checkTask, this->getModel());
			auto values = helper.computeLongRunAverageRewards(env, rewardModel.get());
            std::unique_ptr<CheckResult> result(new ExplicitQuantitativeCheckResult<ValueType>(std::move(values)));
            if (checkTask.isProduceSchedulersSet()) {
                result->asExplicitQuantitativeCheckResult<ValueType>().setScheduler(std::make_unique<storm::storage::Scheduler<ValueType>>(helper.extractScheduler()));
            }
            return result;
        }

        template<typename SparseMdpModelType>
        std::unique_ptr<CheckResult> SparseMdpPrctlModelChecker<SparseMdpModelType>::checkMultiObjectiveFormula(Environment const& env, CheckTask<storm::logic::MultiObjectiveFormula, ValueType> const& checkTask) {
            return multiobjective::performMultiObjectiveModelChecking(env, this->getModel(), checkTask.getFormula());
        }

        template<typename SparseMdpModelType>
        std::unique_ptr<CheckResult> SparseMdpPrctlModelChecker<SparseMdpModelType>::checkQuantileFormula(Environment const& env, CheckTask<storm::logic::QuantileFormula, ValueType> const& checkTask) {
            STORM_LOG_THROW(checkTask.isOnlyInitialStatesRelevantSet(), storm::exceptions::InvalidOperationException, "Computing quantiles is only supported for the initial states of a model.");
            STORM_LOG_THROW(this->getModel().getInitialStates().getNumberOfSetBits() == 1, storm::exceptions::InvalidOperationException, "Quantiles not supported on models with multiple initial states.");
            uint64_t initialState = *this->getModel().getInitialStates().begin();

            helper::rewardbounded::QuantileHelper<SparseMdpModelType> qHelper(this->getModel(), checkTask.getFormula());
            auto res = qHelper.computeQuantile(env);

            if (res.size() == 1 && res.front().size() == 1) {
                return std::unique_ptr<CheckResult>(new ExplicitQuantitativeCheckResult<ValueType>(initialState, std::move(res.front().front())));
            } else {
                return std::unique_ptr<CheckResult>(new ExplicitParetoCurveCheckResult<ValueType>(initialState, std::move(res)));
            }
        }

        // <state_action_ranking>

        template<typename SparseMdpModelType>
        std::unique_ptr<storm::models::sparse::Mdp<typename SparseMdpModelType::ValueType>> SparseMdpPrctlModelChecker<SparseMdpModelType>::restrictMdp(const storm::storage::BitVector& actionsToRemove)
        {
            auto actionsToKeep = ~actionsToRemove;

            storm::storage::SparseMatrix<ValueType> trimmedMatrix = this->getModel().getTransitionMatrix().restrictRows(actionsToKeep, false);
            STORM_LOG_DEBUG("Trimmed Transition Matrix: \n" << trimmedMatrix);

            // adapt choice labelings, rewards, etc.

            auto choiceLabeling = this->getModel().hasChoiceLabeling() ? this->getModel().getOptionalChoiceLabeling() : boost::none;
            if (choiceLabeling) choiceLabeling.get().removeChoices(actionsToRemove);

            auto choiceOrigins = this->getModel().hasChoiceOrigins() ? this->getModel().getOptionalChoiceOrigins() : boost::none;
            if (choiceOrigins) choiceOrigins.get()->removeRows(actionsToRemove);

            auto rewardModels =  this->getModel().getRewardModels();
            for (auto &reward : rewardModels) reward.second = reward.second.restrictActions(actionsToKeep);

            // create final trimmed model

            storm::storage::sparse::ModelComponents<ValueType> trimmedModelComponents = storm::storage::sparse::ModelComponents<ValueType>(trimmedMatrix, this->getModel().getStateLabeling(), rewardModels, false, boost::none, boost::none, choiceLabeling, choiceOrigins);
            if (this->getModel().hasStateValuations()) trimmedModelComponents.stateValuations = this->getModel().getStateValuations();

            auto trimmedModel = std::make_unique<storm::models::sparse::Mdp<ValueType>>(trimmedModelComponents);

            // TODO NEXT: CONTINUE

            // auto rhs = checkTask.getFormula().getrightProperty();
            // STORM_LOG_DEBUG("rhs: " << (*rhs.getRawFormula()));
            // auto rightCheckTask = CheckTask<storm::logic::Formula, ValueType>(*rhs.getRawFormula());
            // rightCheckTask.setProduceSchedulers(checkTask.isProduceSchedulersSet());
            // rightCheckTask.updateOperatorInformation();
            // std::unique_ptr<CheckResult> tmpr = storm::api::verifyWithSparseEngine<ValueType>(env, trimmedModel, rightCheckTask);
            // auto rightresult = tmpr->asExplicitQuantitativeCheckResult<ValueType>();
            //  if(rightCheckTask.isProduceSchedulersSet()) {
            //     rightresult.getScheduler().printToStream(std::cout, trimmedModel, false, false); //Todo find better solution
            //     rightresult.removeScheduler();
            // }

            // uint64_t initialState = *trimmedModel->getInitialStates().begin();
            // return std::unique_ptr<CheckResult>(new ExplicitQuantitativeCheckResult<ValueType>(std::move(rightresult)));

            return trimmedModel;
        }

        template<typename SparseMdpModelType>
        void SparseMdpPrctlModelChecker<SparseMdpModelType>::printResultsPerState(const std::vector<ValueType>& result, const storm::OptimizationDirection dir, const std::string prefix) {

            std::ofstream outData;
            std::string direction = dir == OptimizationDirection::Minimize ? "minimize" : "maximize";
            outData.open(prefix + "results_" + direction);
            for (std::size_t i = 0; i < result.size(); i++) {

                // output state info
                bool hasStateValuations = this->getModel().hasStateValuations();
                if(hasStateValuations) {
                    outData << this->getModel().getStateValuations().toString(i, true);
                } else {
                    outData << i;
                }
                outData << "  Result:" << std::fixed << std::setprecision(5) << result.at(i) << std::endl;
            }
        }


        template<typename SparseMdpModelType>
        void SparseMdpPrctlModelChecker<SparseMdpModelType>::computeStateActionRanking(const std::vector<ValueType>& choiceValues) {
            std::ofstream outData;
            outData.open("action_ranking");

            STORM_LOG_ERROR_COND(outData.is_open(), "File of 'action_ranking' couldn't be opened!");
            STORM_LOG_ERROR_COND(choiceValues.size() == this->getModel().getTransitionMatrix().getRowCount(), "State-Action Ranking requires choiceValues!");

            this->stateValueMapping.reserve(this->getModel().getTransitionMatrix().getRowGroupCount());

            std::size_t worstStateActionOffset = 0;
            auto choice_val_it = choiceValues.begin();

            for (storm::storage::SparseMatrixIndexType groupId = 0; groupId < this->getModel().getTransitionMatrix().getRowGroupCount(); groupId++) {
                STORM_LOG_DEBUG("Current Group (= State) ID: " << groupId);
                STORM_LOG_DEBUG(" > Group Size: " << this->getModel().getTransitionMatrix().getRowGroupSize(groupId));

                auto choice_val_it2 = choice_val_it + this->getModel().getTransitionMatrix().getRowGroupSize(groupId);

                // fetch list of val(s, a) for all 'a' for given state 's' (which corresponds to groupId).

                std::vector<ValueType> stateActionValues (choice_val_it, choice_val_it2);

                // determine val(s) := |local_max_val - local_min_val|

                const auto [lMin, lMax] = std::minmax_element(stateActionValues.cbegin(), stateActionValues.cend());
                const ValueType stateValue = *lMax - *lMin;
                stateValueMapping.push_back(stateValue);

                // set worst action for this state in bitvector

                const std::size_t actionOffset = lMin - stateActionValues.begin();
                const std::size_t globActionOffset = worstStateActionOffset + actionOffset;

                // Update iterator / WA Offset for next iteration

                choice_val_it = choice_val_it2;
                worstStateActionOffset += this->getModel().getTransitionMatrix().getRowGroupSize(groupId);

                STORM_LOG_DEBUG(" > State Action Values = { " << stateActionValues << " }");
                STORM_LOG_DEBUG(" > (min, max) = (" << *lMin << ", " << *lMax << ")");
                STORM_LOG_DEBUG(" > val(s) = " << stateValue);
                STORM_LOG_DEBUG(" > Worst Action Temp  Offset: " << actionOffset);
                STORM_LOG_DEBUG(" > Worst-Action Final Offset: " << globActionOffset);
            }

            // normalize to get rank(s) := (val(s) - global_min_val) / (global_max_val - global_min_val)
            // edge case: rank(s) = 0 if all val(s) is for all s equal.

            const auto [gMin, gMax] = std::minmax_element(stateValueMapping.cbegin(), stateValueMapping.cend());
            const ValueType normFactor = (*gMax - *gMin);

            STORM_LOG_WARN_COND(normFactor != 0, "State-Action-Ranking is uniform: All state values are equal!");
            STORM_LOG_DEBUG("(global min, global max) = (" << *gMin << ", " << *gMax << ")");

            std::function<ValueType(ValueType)> regularNorm = [=](ValueType val) { return ((val - *gMin) / normFactor); };
            std::function<ValueType(ValueType)> constantZero = [=]([[maybe_unused]] ValueType val) { return static_cast<ValueType>(0); };

            std::transform(stateValueMapping.cbegin(), stateValueMapping.cend(), stateValueMapping.begin(), (normFactor != 0 ? regularNorm : constantZero));

            //STORM_LOG_ASSERT(stateValueMapping.size() == this->getModel().getTransitionMatrix().getRowGroupCount(), "Count of state-value map entries doesn't match transition matrix group count!");
            //STORM_LOG_ASSERT(std::find_if(stateValueMapping.cbegin(), stateValueMapping.cend(), [=](const auto& val) { return (val > 1 || val < 0); }) == stateValueMapping.cend(), "Normalization Failure - Values are not between [0; 1]!");

            bool hasStateValuations = this->getModel().hasStateValuations();
            for (std::size_t i = 0; i < stateValueMapping.size(); i++) {

                // output state info

                if(hasStateValuations) {
                    outData << this->getModel().getStateValuations().toString(i, true);
                } else {
                    outData << i;
                }
                outData << "  Value:" << std::fixed << std::setprecision(5) << stateValueMapping.at(i) << "\t Choices:";
                const std::size_t nextRowGroupIdx = this->getModel().getTransitionMatrix().getRowGroupIndices().at(i + 1);
                for (std::size_t rowIdx = this->getModel().getTransitionMatrix().getRowGroupIndices().at(i); rowIdx < nextRowGroupIdx; rowIdx++) {

                    // prepare bitmask (set all bits of relevant group to 1)

                    // pre choice label in form: "LABEL_1, LABEL_2, LABEL_3[, ...]"

                    std::string labelNames = "";

                    if (this->getModel().hasChoiceLabeling()) {
                        auto labels = this->getModel().getChoiceLabeling().getLabelsOfChoice(rowIdx);
                        auto labels_it = labels.begin();
                        if (labels_it != labels.end()) { labelNames = *(labels_it++); }
                        for (; labels_it != labels.end(); labels_it++) { labelNames += ", " + (*labels_it); }
                    }

                    // final output

                    outData << labelNames << ":" << choiceValues.at(rowIdx) << (rowIdx + 1 == nextRowGroupIdx ? "\n" : ",");
                }
            }

            outData.close();
        }

        // </state_action_ranking>

        template class SparseMdpPrctlModelChecker<storm::models::sparse::Mdp<double>>;

#ifdef STORM_HAVE_CARL
        template class SparseMdpPrctlModelChecker<storm::models::sparse::Mdp<storm::RationalNumber>>;
#endif
    }
}
