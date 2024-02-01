from stable_baselines3 import PPO, DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure, Image
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
from gym_minigrid.wrappers import RGBImgObsWrapper, ImgObsWrapper, MiniWrapper

import os
from subprocess import call
import time
import argparse
import gym

LOG_MODE = True

def create_env(name="MiniGrid-Empty-8x8-v0", info_keywords=""):
    env = gym.make(name, training=True)
    env = RGBImgObsWrapper(env) # Get pixel observationse
    env = ImgObsWrapper(env)    # Get rid of the 'mission' field
    env = MiniWrapper(env)      # Project specific changes
    env = Monitor(env, info_keywords=info_keywords)
    return env

class ImageRecorderCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_training_start(self):
        image = self.training_env.render(mode="rgb_array")
        self.logger.record("trajectory/image", Image(image, "HWC"), exclude=("stdout", "log", "json", "csv"))

    def _on_step(self):
        return True


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        infos = self.locals["infos"][0]
        self.logger.record("info/reached_goal", infos["reached_goal"])
        self.logger.record("info/ran_into_lava", infos["ran_into_lava"])
        if "picked_up" in infos:
            self.logger.record("info/picked_up", infos["picked_up"])
        return True

# ........................................................................... #

def LOG(text : str) -> None:
    if (LOG_MODE):
        print(text)

# ........................................................................... #

def main():
    LOG("> Start Application ...")

    #################################################################
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, required=True, help='Environment (for now only "Testing-v0" possible)')
    parser.add_argument('--tb-logs-dir', type=str, required=True, help='')
    parser.add_argument('--steps', type=int, required=True, help='Number of steps for training')
    parser.add_argument('--model-dir', type=str, required=False, help='The directory for the saved model.', default="trained_models")
    #parser.add_argument('--log_output', type=bool, default=False, help='Print output log (default = false)')
    args = parser.parse_args()

    directory = args.model_dir
    if not os.path.exists(directory):
      os.makedirs(directory)

    HEAD_hash = os.popen('git rev-parse --short HEAD').read().strip()
    dirty = os.popen("git diff --quiet || echo 'dirty'").read().strip()
    if dirty != "":
        HEAD_hash = HEAD_hash + "_" + "dirty"
    start_time = time.strftime("%b%d-%Y-%H:%M:%S")
    env_tb_dir = os.environ.get('POLRE_TB_DIR')
    if env_tb_dir == None:
        tb_path = "/tmp/sb3_log"
    else:
        tb_path = env_tb_dir

    #tmp_path = f"{tb_path}/{args.env}-{start_time}-{HEAD_hash}"
    tmp_path = f"{tb_path}/{args.tb_logs_dir}"
    file_name = f"{args.env}__{start_time}_{HEAD_hash}_{args.steps}"
    new_logger = configure(tmp_path, ["stdout", "tensorboard"])


    #################################################################
    # Create & Setup environment
    info_keywords = ["ran_into_lava", "reached_goal"]

    env = create_env(args.env, info_keywords)
    eval_env = create_env(args.env, info_keywords)
    observation = env.reset()
    eval_callback = EvalCallback(eval_env, best_model_save_path=args.model_dir + "/" + file_name,
                                 log_path=tmp_path, eval_freq=max(500,  int(args.steps/30)),
                                 deterministic=True, render=False)
    checkpoint_callback = CheckpointCallback(save_freq=max(500,  int(args.steps/30)),
                                             save_path=args.model_dir + "/" + file_name,
                                             name_prefix="intermediate_model")
    ###############################################################
    # LEARN with stable-baselines3
    LOG("> Start Learning ...")
    #model = DQN("CnnPolicy", env, verbose=1, buffer_size=20000, exploration_fraction=0.9, exploration_initial_eps=0.7, exploration_final_eps=0.01, gamma=0.95, train_freq=3, learning_starts=0, device="cuda")
    model = DQN("CnnPolicy", env, verbose=1, buffer_size=20000, gamma=0.95, train_freq=3, learning_starts=0, device="cuda")
    print(model.policy)
    assert(False)
    model.set_logger(new_logger)

    model.learn(total_timesteps=args.steps,callback=[ImageRecorderCallback(), TensorboardCallback(), eval_callback])
    LOG("> Finished Learning!")

    #################################################################
    # Save Model

    model.save(f"{directory}/{file_name}")
    LOG(f"> Saved model in {directory} as: {file_name}")

    #################################################################
    # Final Cleanup
    env.close()
    LOG(env.printGrid(init=True))
    LOG("> Finished Application!")

# ........................................................................... #

if __name__ == '__main__':
    main()
