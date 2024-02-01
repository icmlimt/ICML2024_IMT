## Docker Image

For an easy and minimally invasive method to reproduce our results, we have chosen to provide a Dockerfile that will compile all the needed code into a single image. This image can then easily be used to run the experiments.

In order to build the container (assuming that docker is installed and enabled on your machine) you have to run the following command:

  DOCKER_BUILDKIT=1 sudo docker image build -t icml_imt .

The docker container can then be started with the following command:

  sudo docker run -it --rm icml_imt

This will open a shell in the root directory ("/") of the image.

By executing:

  ./run_minigrid.sh

one iteration for policy pi_1, pi_2, pi_3 and pi_4 will be run, each for IMT, EMT and RT (where applicable).

The results can the be viewed live during execution or in subfolders located in "./Minigrid"

By executing:

 ./run_uav.sh

one iteration for the policies for noise levels 0.1, 0.75 and 1.0 are run with the EMT and IMT approach.

The results can the be viewed live during execution or in subfolders located in "./UAV_Reach_Avoid"

## Additional Images

We provide some more images for the examples from the paper. We want to especially draw your attention to

 barcelona_bad_policy_iterations.png,
 barcelona_good_policy_iterations.png,and
 policy_1_policy_with_queries.png

showcasing the querying results for the different policies. The two rows in each .png show the results of the policy queries and the verdicts from IMT.

The first row shows a triangles in:
 - "red" for forward movement,
 - "green" for turning left,
 - "blue" for turning right, and
 - "gray" for standing still.
