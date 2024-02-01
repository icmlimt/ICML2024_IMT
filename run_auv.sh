#!/bin/sh

cd /UAV_Reach_Avoid
source env/bin/activate

export TEMPEST_BINARY=/tempest-devel/build/bin/storm

epoch="$(date +%s)"

cd /UAV_Reach_Avoid/noise01
# Ensure that state valuations are present in current directory...
cp ../MDP_state_valuations .
python3 ../test_model.py --tra Abstraction_interval.tra --lab Abstraction_interval.lab --rew safety --stra PRISM_interval_policy.txt --refinement-steps 500
python3 ../test_model.py --tra Abstraction_interval.tra --lab Abstraction_interval.lab --rew safety --stra PRISM_interval_policy.txt --refinement-steps 500 --ablation
python3 ../test_model.py --tra Abstraction_interval.tra --lab Abstraction_interval.lab --rew safety --stra PRISM_interval_policy.txt --refinement-steps 500 --random 3

cd /UAV_Reach_Avoid/noise025
# Ensure that state valuations are present in current directory...
cp ../MDP_state_valuations .
python3 ../test_model.py --tra Abstraction_interval.tra --lab Abstraction_interval.lab --rew safety --stra PRISM_interval_policy.txt --refinement-steps 500
python3 ../test_model.py --tra Abstraction_interval.tra --lab Abstraction_interval.lab --rew safety --stra PRISM_interval_policy.txt --refinement-steps 500 --ablation
python3 ../test_model.py --tra Abstraction_interval.tra --lab Abstraction_interval.lab --rew safety --stra PRISM_interval_policy.txt --refinement-steps 500 --random 3

cd /UAV_Reach_Avoid/noise05
# Ensure that state valuations are present in current directory...
cp ../MDP_state_valuations .
python3 ../test_model.py --tra Abstraction_interval.tra --lab Abstraction_interval.lab --rew safety --stra PRISM_interval_policy.txt --refinement-steps 500
python3 ../test_model.py --tra Abstraction_interval.tra --lab Abstraction_interval.lab --rew safety --stra PRISM_interval_policy.txt --refinement-steps 500 --ablation
python3 ../test_model.py --tra Abstraction_interval.tra --lab Abstraction_interval.lab --rew safety --stra PRISM_interval_policy.txt --refinement-steps 500 --random 3

cd /UAV_Reach_Avoid/noise075
# Ensure that state valuations are present in current directory...
cp ../MDP_state_valuations .
python3 ../test_model.py --tra Abstraction_interval.tra --lab Abstraction_interval.lab --rew safety --stra PRISM_interval_policy.txt --refinement-steps 500
python3 ../test_model.py --tra Abstraction_interval.tra --lab Abstraction_interval.lab --rew safety --stra PRISM_interval_policy.txt --refinement-steps 500 --ablation
python3 ../test_model.py --tra Abstraction_interval.tra --lab Abstraction_interval.lab --rew safety --stra PRISM_interval_policy.txt --refinement-steps 500 --random 3

cd /UAV_Reach_Avoid/noise1
# Ensure that state valuations are present in current directory...
cp ../MDP_state_valuations .
python3 ../test_model.py --tra Abstraction_interval.tra --lab Abstraction_interval.lab --rew safety --stra PRISM_interval_policy.txt --refinement-steps 500
python3 ../test_model.py --tra Abstraction_interval.tra --lab Abstraction_interval.lab --rew safety --stra PRISM_interval_policy.txt --refinement-steps 500 --ablation
python3 ../test_model.py --tra Abstraction_interval.tra --lab Abstraction_interval.lab --rew safety --stra PRISM_interval_policy.txt --refinement-steps 500 --random 3

cd /UAV_Reach_Avoid
