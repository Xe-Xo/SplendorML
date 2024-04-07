

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Figure


import numpy as np

from einops import rearrange


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

    return mean_dict, std_dict, min_dict, max_dict


class TensorboardCallback(BaseCallback):

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.i_dict = {} # individual end_game_state_data

    def add_dicts(self, done_dicts):
        for d in done_dicts:
            if d["average_turns"] == 0:
                continue

            for k, v in d.items():
                
                if k not in self.i_dict.keys():
                    self.i_dict[k] = [v]
                else:
                    self.i_dict[k].append(v)


    def _on_step(self) -> bool:
        
        done_vector = [over for over in self.training_env.unwrapped.get_attr("env_reset")]
        if not any(done_vector):
            return True
        
        dict_vector = self.training_env.unwrapped.env_method("get_game_stats")
        dict_vector_done = [d for d, done in zip(dict_vector, done_vector) if done]

        for i in dict_vector_done:
            print("Game Done! ", i["average_turns"])

        self.add_dicts(dict_vector_done)
        
        return True
    
    # Need to track i_dict over rollout and calculate means at rollout time instead?

    def on_rollout_start(self) -> None:

        mean_dict, std_dict, min_dict, max_dict = merge_dicts_by_mean(self.i_dict)

        for k in mean_dict.keys():
            self.logger.record(f"game_stats_{k}/mean", mean_dict[k])
            self.logger.record(f"game_stats_{k}/std", std_dict[k])
            self.logger.record(f"game_stats_{k}/min", min_dict[k])
            self.logger.record(f"game_stats_{k}/max", max_dict[k])
        
        self.i_dict = {}