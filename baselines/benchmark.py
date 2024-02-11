from splendor_env import SplendorEnv, SPLENDOR_ACTIONS
from visual_splendor_env import VisualSplendorEnv

from stable_baselines3.common.utils import explained_variance, get_schedule_fn, obs_as_tensor, safe_mean
from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks, is_masking_supported


import gymnasium as gym
import numpy as np
import torch as th


env = VisualSplendorEnv()
obs, _ = env.reset(0)

# Load a Model and play 1000 games against Random Agent to get a Win Rate

filename = "experiments\session_da3362f7\splendor_73400320_steps.zip"

#model = MaskablePPO.load(filename, env=env)

games_played = 0
wins = 0
episode_rewards = []

while games_played < 1000:

    
    #action, _ = model.predict(obs, deterministic=False)
    #if env.action_masks()[action] == 0:
    #    continue
    action = env.predict()


    obs, reward, terminated, truncated, info = env.step(action)
    
    #print(action)
    #print(SPLENDOR_ACTIONS[action])
    #print(reward)

    episode_rewards.append(reward)

    if terminated or truncated:

        mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
        #print(mean_reward, std_reward)
        episode_rewards = []
        games_played += 1
        if env.player_num in env.game_state.get_winner():
            wins += 1
        #print(env.game_state.static_eval(env.player_num))
        obs, _ = env.reset(seed=games_played)

    for i in range(0, 500):
        env.render()


print(round(wins/games_played*100) , "%")


#with th.no_grad():
#    # Convert to pytorch tensor or to TensorDict
#    obs_tensor = obs_as_tensor(model._last_obs, model.device)
    # This is the only change related to invalid action masking
#    action_masks = get_action_masks(env)
#    actions, values, log_probs = model.policy(obs_tensor, action_masks=action_masks)