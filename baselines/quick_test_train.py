from os.path import exists
from pathlib import Path
import uuid

import gymnasium as gym
import numpy as np
import torch as th

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList, EvalCallback

from sb3_contrib.common.envs import InvalidActionEnvDiscrete
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from splendor_env import SplendorEnv
from splendor_selfplay_env import SplendorSelfPlayEnv, SelfPlayCallback


def mask_fn(env: gym.Env) -> np.ndarray:
    # Do whatever you'd like in this function to return the action mask
    # for the current env. In this example, we assume the env has a
    # helpful method we can rely on.
    return env.last_obs["action_mask"]

def make_env(rank=0, config={}, seed=0):

    def __init():
        env = SplendorSelfPlayEnv()

        env = Monitor(env)
        env.reset(seed=(seed + rank))
        return env
    
    return __init

if __name__ == '__main__':

    sess_id = str(uuid.uuid4())[:8]
    sess_path = Path(f'experiments/session_{sess_id}')

    ep_length = 2048 * 10 * 4
    cpu_multiplier = 0.5
    env_config = {}
    n_steps = int(5120 // cpu_multiplier)
    num_cpu = int(32 * cpu_multiplier)
    #env = make_env(0, env_config)
    env = SubprocVecEnv([make_env(i, env_config) for i in range(num_cpu)])
    checkpoint_callback = CheckpointCallback(save_freq=ep_length, save_path=sess_path,
                                     name_prefix='splendor')
    
    eval_callback = SelfPlayCallback(eval_env=env,best_model_save_path="experiments/self_play",log_path="experiments/self_play", eval_freq=n_steps, verbose=1)

    callbacks = [checkpoint_callback]#eval_callback]

    learn_steps = 40
    
    model = MaskablePPO("MultiInputPolicy", env, verbose=1, n_steps=n_steps, batch_size=512, n_epochs=10, gamma=0.998, tensorboard_log=sess_path, ent_coef=0.01, learning_rate=0.0003, vf_coef=0.5)
    
    for i in range(learn_steps):
        model.learn(total_timesteps=ep_length*num_cpu*1000, callback=CallbackList(callbacks))
        model.save("splendor_ppo")

