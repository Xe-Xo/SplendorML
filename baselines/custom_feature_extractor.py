from typing import Callable, Dict, List, Optional, Tuple, Type, Union

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, is_image_space, get_flattened_obs_dim, NatureCNN, TensorDict, gym
from gymnasium import spaces
import torch as th
from torch import nn

class CustomFeatureExtractor(BaseFeaturesExtractor):

    """
    
    """

    def __init__(
            self,
            observation_space: spaces.Dict,
            cnn_output_dim: int = 256,
            ):
        
        super().__init__(observation_space, features_dim=1)

        self.action_mask_feature_extractor = nn.Sequential(
            nn.Linear(50, 8),
            nn.ReLU(),
            nn.Linear(8, 8),
            nn.ReLU(),
            nn.Linear(8, 50),
        )

        

