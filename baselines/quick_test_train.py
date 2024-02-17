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
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList, EvalCallback

from sb3_contrib.common.envs import InvalidActionEnvDiscrete
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from splendor_env import SplendorEnv
from visual_splendor_env import VisualSplendorEnv
from splendor_selfplay_env import SplendorSelfPlayEnv, SelfPlayCallback
from logger_callback import TensorboardCallback
from agents.base_agent import ModelAgent, Agent


def mask_fn(env: gym.Env) -> np.ndarray:
    # Uncomment to make masking a no-op
    # return np.ones_like(env.action_mask)
    return np.array(env.game_state.action_mask)

def get_wrapper(env: gym.Env) -> gym.Env:
    return ActionMasker(env, mask_fn)

def make_env(rank=0, config={}, seed=0):

    agent = ModelAgent("experiments\self_play\\best_model.zip")
    agents=(
            agent,
            agent,
            agent,
            )

    def __init():
        env = SplendorEnv(agents=agents)
        #env = InvalidActionEnvDiscrete()
        env = get_wrapper(env)

        #env = Monitor(env)
        env.reset(seed=None)
        return env
    
    return __init

def make_visual_env(rank=0, config={}, seed=0):

    # This doesnt work cause the masking?
    def __init():
        env = VisualSplendorEnv()
        env = Monitor(env)
        env.reset(seed=(seed + rank))
        return env
    
    return __init


existing_session = "d86dbdba"#"8637404d"
steps = "20971520"#"11796480"
#model_path = f'experiments/session_{existing_session}/splendor_{steps}_steps123.zip'
model_path = f'experiments/self_play/best_model.zip'


if __name__ == '__main__':

    ep_length = 2048 * 1 * 4
    cpu_multiplier = 1
    env_config = {}
    n_steps = int(300 // cpu_multiplier) # 30 ep length * 10 games
    num_cpu = int(32 * cpu_multiplier)

    sess_id = str(uuid.uuid4())[:8]
    sess_path = Path(f'experiments/session_{sess_id}')



    #env = make_env(0, env_config)()
    
    env = make_vec_env(make_env(), n_envs=num_cpu, env_kwargs=env_config, seed=0, vec_env_cls=DummyVecEnv, vec_env_kwargs={})
    #env = SubprocVecEnv([make_env(i, env_config) for i in range(num_cpu)])
    checkpoint_callback = CheckpointCallback(save_freq=ep_length, save_path=sess_path,
                                     name_prefix='splendor')
    

    #tensorboard_callback = TensorboardCallback()

    eval_callback = SelfPlayCallback()

    callbacks = [eval_callback,checkpoint_callback]

    
    learn_steps = 40

    if exists(model_path):
        print("Loading existing model") 
        model = MaskablePPO.load(model_path, env=env, tensorboard_log=sess_path, verbose=1)
        model.n_steps = ep_length
        model.n_envs = num_cpu
        model.rollout_buffer.buffer_size = ep_length
        model.rollout_buffer.n_envs = num_cpu
        model.rollout_buffer.reset()
        #quit()
    else:
        
        model = MaskablePPO("MultiInputPolicy", env, verbose=1, n_steps=n_steps, batch_size=64, n_epochs=10, gamma=0.998, tensorboard_log=sess_path, ent_coef=0.01, learning_rate=0.0003, vf_coef=0.5)

    for i in range(learn_steps):
        model.learn(total_timesteps=ep_length*num_cpu*1000, callback=CallbackList(callbacks))
        model.save("splendor_ppo")

