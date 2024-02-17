
from splendor_env import SplendorEnv
from pathlib import Path
import os
from shutil import copyfile # keep track of generations

from stable_baselines3 import PPO
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback

import random
import numpy as np

LOGDIR = "experiments/self_play"
BEST_THRESHOLD = 0.40 # 15% games over average win rate 25%

def merge_dicts_by_mean(dicts):

    std_dict = {}
    min_dict = {}
    max_dict = {}
    mean_dict = {}

    for k in dicts:
        max_dict[k] = np.max(dicts[k])
        min_dict[k] = np.min(dicts[k])
        std_dict[k] = np.std(dicts[k])
        mean_dict[k] = np.mean(dicts[k])

    return mean_dict, std_dict, min_dict, max_dict #{k: len(dicts[k]) for k in dicts.keys()}


class SelfPlayCallback(BaseCallback):

    def __init__(self, *args, **kwargs):
        super(SelfPlayCallback, self).__init__(*args, **kwargs)
        self.generation = 0
        self.i_dict = {} # individual end_game_state_data
        self.threshold_count = 0
        

    def add_dicts(self, done_dicts):
        for d in done_dicts:
            if d["average_turns"] == 0:
                continue

            for k, v in d.items():
                
                if k not in self.i_dict.keys():
                    self.i_dict[k] = [v]
                else:
                    self.i_dict[k].append(v)

                if len(self.i_dict[k]) > 100:
                    self.i_dict[k].pop(0)

    def _on_rollout_start(self) -> None:
        

        super(SelfPlayCallback, self)._on_rollout_start()

        mean_dict, std_dict, min_dict, max_dict = merge_dicts_by_mean(self.i_dict)

        for k in mean_dict.keys():

            self.logger.record(f"game_stats_{k}/mean", mean_dict[k])

            if "action_type" in k:
                continue
            
            self.logger.record(f"game_stats_{k}/std", std_dict[k])
            self.logger.record(f"game_stats_{k}/min", min_dict[k])
            self.logger.record(f"game_stats_{k}/max", max_dict[k])
            #self.logger.record(f"game_stats_{k}/count", count_dict[k])
        
        self.i_dict = {}

        self.logger.record(f"selfplay/generation", self.generation)

        if "win_rate" in mean_dict.keys():

            last_win_rate = mean_dict["last_score"]
            print(f"Average Win rate {last_win_rate}")
            
            if last_win_rate >= BEST_THRESHOLD:
                self.threshold_count += 1
            else:
                self.threshold_count = 0

            if self.threshold_count > 2 or last_win_rate > 0.50: #  


                print(f"SELFPLAY: win_rate achieved: {last_win_rate}>={BEST_THRESHOLD}")
                self.generation += 1

                print("SELFPLAY: new best model, bumping up generation to", self.generation)
                self.model.save(os.path.join(LOGDIR, "best_model.zip"))
                self.model.save(os.path.join(LOGDIR, f"history_{self.generation}.zip"))
                self.training_env.unwrapped.env_method("reset_agents")

    def _on_step(self) -> bool:
        
        done_vector = [over for over in self.training_env.unwrapped.get_attr("env_reset")]
        if not any(done_vector):
            return True
        
        
        dict_vector = self.training_env.unwrapped.env_method("get_game_stats")
        
        dict_vector_done = [d for d, done in zip(dict_vector, done_vector) if done]

        for i in dict_vector_done:
            for j in i.keys():
                print(f"{j}: {i[j]}")
            print("-----")

        self.add_dicts(dict_vector_done)
        
        return True



class SplendorSelfPlayEnv(SplendorEnv):

    def __init__(self):
        self.best_model = None
        self.best_model_filename = None
        super(SplendorSelfPlayEnv, self).__init__()

    def predict(self):
        if self.best_model is None:
            action_choice = [i for i, a in enumerate(self.game_state.action_mask) if a == 1]
            i = random.choice(action_choice)
            return i

        else:

            action_choice = [i for i, a in enumerate(self.game_state.action_mask) if a == 1]
            obs = self.get_obs()
            action = self.best_model.predict(obs)
            if action not in action_choice:
                i = random.choice(action_choice)
                return i
            else:
                return action
        
    def reset(self,seed=None, options=None):
        # load model if it's there
        modellist = [f for f in os.listdir(LOGDIR) if f.startswith("history_")]
        modellist.sort()

        if len(modellist) > 0:
            filename = os.path.join(LOGDIR, modellist[-1])  # the latest best model
            if filename != self.best_model_filename:
                print("loading model: ", filename)
                self.best_model_filename = filename
                if self.best_model is not None:
                    del self.best_model
                self.best_model = MaskablePPO.load(filename, env=self)

        return super(SplendorSelfPlayEnv, self).reset(seed=seed, options=options)