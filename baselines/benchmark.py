from splendor_env import SplendorEnv, SPLENDOR_ACTIONS
from visual_splendor_env import VisualSplendorEnv

from agents.base_agent import Agent, ModelAgent, StaticEvalAgent, HeuristicEvalAgent, BlendedEvalAgent, SearchAgent

from stable_baselines3.common.utils import explained_variance, get_schedule_fn, obs_as_tensor, safe_mean
from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks, is_masking_supported


import gymnasium as gym
import numpy as np
import torch as th
import time
import random

env = VisualSplendorEnv(agents=(
    BlendedEvalAgent(random_threshold=0),
    BlendedEvalAgent(random_threshold=0),
    BlendedEvalAgent(random_threshold=0),
    #ModelAgent("experiments\session_d86dbdba\splendor_20971520_steps.zip", random_threshold=0), 
    #ModelAgent("experiments\session_dd90eade\splendor_22544384_steps.zip", random_threshold=0),))
    ))
obs, _ = env.reset(None)

# Load a Model and play 1000 games against Random Agent to get a Win Rate

filename = "experiments\self_play\\best_model.zip"

model = MaskablePPO.load(filename, env=env)

games_played = 0
wins = 0
draws = 0
episode_rewards = []

while games_played < 1000:

    
    action, _ = model.predict(obs, deterministic=False, action_masks=env.action_masks())
    
    if env.action_masks()[action] == 0:
        print(f"predicted invalid action {SPLENDOR_ACTIONS[action]}")
        action = random.choice([i for i, a in enumerate(env.action_masks()) if a == 1])    
        #continue


    #action = env.predict()

    #(child, action), seval, heval, eval   = env.game_state.get_best_action(3)



    obs, reward, terminated, truncated, info = env.step(action)
    
    
    #print(action)
    #print(SPLENDOR_ACTIONS[action])
    #print(reward)

    episode_rewards.append(reward)
    #time.sleep(0.2)

    if terminated or truncated:

        games_played += 1


        for i in range(0,4):

            if i == env.player_num:

                if env.player_num in env.game_state.get_winner() and len(env.game_state.get_winner()) == 1:
                    wins += 1
                    print(f"WIN - {games_played} games played")
                elif env.player_num in env.game_state.get_winner():
                    draws += 1/len(env.game_state.get_winner())
                    print(f"DRAW - {games_played} games played")
                else:
                    print(f"LOSS - {games_played} games played")

            else:

                agent_w = i if i < env.player_num else i-1

                if i in env.game_state.get_winner() and len(env.game_state.get_winner()) == 1:
                    env.agents[agent_w].win()
                elif i in env.game_state.get_winner():
                    env.agents[agent_w].draw(len(env.game_state.get_winner()))
                else:
                    env.agents[agent_w].loss()


        if games_played % 100 == 0:
            for a in env.agents:
                a.update_random_threshold()

            print(round(wins/games_played*100) , "% WIN RATE")
            print(round((wins+draws)/games_played*100) , "% WIN OR DRAW RATE")


        mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
        #print(round(mean_reward,3), round(std_reward,3))
        episode_rewards = []
        
        #if env.player_num in env.game_state.get_winner():
            #wins += 1
            #print(env.game_state.static_eval(env.player_num))
        obs, _ = env.reset(seed=games_played)


        
print(round(wins/games_played*100) , "% WIN RATE")
print(round((wins+draws)/games_played*100) , "% WIN OR DRAW RATE")

#with th.no_grad():
#    # Convert to pytorch tensor or to TensorDict
#    obs_tensor = obs_as_tensor(model._last_obs, model.device)
    # This is the only change related to invalid action masking
#    action_masks = get_action_masks(env)
#    actions, values, log_probs = model.policy(obs_tensor, action_masks=action_masks)