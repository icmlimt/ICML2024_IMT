#!/bin/sh

cd /Minigrid
source env/bin/activate

export TEMPEST_BINARY=/tempest-devel/build/bin/storm

epoch="$(date +%s)"

python3 test_model.py --env "MyCliffWalking-S9-v0" --policy "trained_models/policy_1" --safety --reward "SafetyNoBFS" --refinement-steps 10 --dest "IMT_${epoch}"
python3 test_model.py --env "MyCliffWalking-S9-v0" --policy "trained_models/policy_1" --safety --reward "SafetyNoBFS" --refinement-steps 10 --randomMT --dest "EMT_${epoch}"
python3 test_model.py --env "MyCliffWalking-S9-v0" --policy "trained_models/policy_1" --safety --reward "SafetyNoBFS" --random 10 --dest "RT_${epoch}"
python3 test_model.py --env "MyCliffWalking-S9-v0" --policy "trained_models/policy_2" --safety --reward "SafetyNoBFS" --refinement-steps 10 --dest "IMT_${epoch}"
python3 test_model.py --env "MyCliffWalking-S9-v0" --policy "trained_models/policy_2" --safety --reward "SafetyNoBFS" --refinement-steps 10 --randomMT --dest "EMT_${epoch}"

python3 test_model.py --env "Barcelona-v0" --policy "trained_models/barcelona_good_policy" --oneways --reward "Time" --refinement-steps 15 --dest "IMT_${epoch}"
python3 test_model.py --env "Barcelona-v0" --policy "trained_models/barcelona_good_policy" --oneways --reward "Time" --refinement-steps 15 --randomMT --dest "EMT_${epoch}"
python3 test_model.py --env "Barcelona-v0" --policy "trained_models/barcelona_good_policy" --oneways --reward "Time" --random 100 --dest "RT_${epoch}"

python3 test_model.py --env "Barcelona-v0" --policy "trained_models/barcelona_worse_policy" --oneways --reward "Time" --refinement-steps 15 --dest "IMT_${epoch}"
python3 test_model.py --env "Barcelona-v0" --policy "trained_models/barcelona_worse_policy" --oneways --reward "Time" --refinement-steps 15 --randomMT --dest "EMT_${epoch}"
python3 test_model.py --env "Barcelona-v0" --policy "trained_models/barcelona_worse_policy" --oneways --reward "Time" --random 100 --dest "RT_${epoch}"


ls -altr
