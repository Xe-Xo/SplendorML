
from splendor_env import SplendorEnv
from pathlib import Path
import os
from shutil import copyfile # keep track of generations

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback

import random

LOGDIR = "experiments/self_play"
BEST_THRESHOLD = 0

class SelfPlayCallback(EvalCallback):

    def __init__(self, *args, **kwargs):
        super(SelfPlayCallback, self).__init__(*args, **kwargs)
        self.best_mean_reward = -1
        self.generation = 0

    def _on_step(self) -> bool:

        result = super(SelfPlayCallback, self)._on_step()
        if result and self.best_mean_reward > BEST_THRESHOLD:
            self.generation += 1
            print("SELFPLAY: mean_reward achieved:", self.best_mean_reward)
            print("SELFPLAY: new best model, bumping up generation to", self.generation)

            source_file = os.path.join(LOGDIR, "best_model.zip")
            backup_file = os.path.join(LOGDIR, "history_"+str(self.generation).zfill(8)+".zip")
            copyfile(source_file, backup_file)
            self.best_mean_reward = BEST_THRESHOLD


        return result

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
                self.best_model = PPO.load(filename, env=self)

        return super(SplendorSelfPlayEnv, self).reset(seed=seed, options=options)