
from sb3_contrib.ppo_mask import MaskablePPO
import random

class Agent():

    """ Base Class for all Agents """

    def __init__(self,random_threshold=0.00):
        self.random_threshold = random_threshold
        self.random_threshold_decay = 0.01
        self.random_threshold_min = 0.00
        self.random_threshold_max = 1.00

        self.wins = 0
        self.draws = 0
        self.losses = 0
        self.games = 0

    def win(self):
        print(self, "wins")
        self.wins += 1
        self.games += 1

    def loss(self):
        self.losses += 1
        self.games += 1

    def draw(self, num_players):
        print(self, "draws")
        self.wins += 1/num_players
        self.games += 1


    def __repr__(self) -> str:
        return self.__class__.__name__

    def setup(self,env):
        pass

    def get_action(self,env):
        return self.get_random_action(env)

    def get_random_action(self,env):
        action_choice = [i for i, a in enumerate(env.game_state.action_mask) if a == 1]
        i = random.choice(action_choice)
        return i
    
    def predict(self,env):
        if random.random() < self.random_threshold:
            #print(self, "-random")
            return self.get_random_action(env)
        else:
            #print(self, "-action")
            return self.get_action(env)
    
    def update_random_threshold(self):

        win_rate = self.wins / self.games
        
        # If the Player of Env is winning too much, it should be more random

        if win_rate > 0.25:
            self.random_threshold += self.random_threshold_decay
            self.random_threshold = max(self.random_threshold, self.random_threshold_min)
            self.random_threshold = min(self.random_threshold, self.random_threshold_max)
            self.random_threshold = round(self.random_threshold, 2)
            print(f"{self} updating random threshold up to {self.random_threshold} {round(win_rate,2)*100}%")
        elif win_rate < 0.25:
            self.random_threshold -= self.random_threshold_decay
            self.random_threshold = max(self.random_threshold, self.random_threshold_min)
            self.random_threshold = min(self.random_threshold, self.random_threshold_max)
            self.random_threshold = round(self.random_threshold, 2)
            print(f"{self} updating random threshold down to {self.random_threshold} {round(win_rate,2)*100}%")
        else:
            print(f"{self} random threshold unchanged {self.random_threshold} {round(win_rate,2)*100}%")

        self.wins = 0
        self.draws = 0
        self.losses = 0
        self.games = 0

            
class ModelAgent(Agent):

    """ Uses a trained model to predict the best action"""

    def __init__(self, model_filename,random_threshold=0.00):
        super().__init__(random_threshold=random_threshold)
        self.model_filename = model_filename
        self.action_since_model_load = 0
        self.model = None

    def setup(self, env):
        if self.action_since_model_load > 0 or self.model is None:
            print(f"Model not loaded, {self.action_since_model_load} actions ")
            self.load_model(env)
        else:
            print(f"Model already loaded, {self.action_since_model_load} actions since last load")
        
    
    def load_model(self,env):

        try:
            # Load the Model from the given filename
            print(f"loading model from {self.model_filename}")
            self.model = MaskablePPO.load(self.model_filename, env=env)
            self.action_since_model_load = 0
        except:
            print("Failed to load model")
            self.model = None
    
    def get_action(self,env):

        if self.model is None:
            #print("Model not loaded")
            
            return self.get_random_action(env)
        else:
            self.action_since_model_load += 1
            action, _ = self.model.predict(env.get_obs(), deterministic=False, action_masks=env.game_state.action_mask)
            return action

class StaticEvalAgent(Agent):

    """ Randomly selects from the best static eval choices available immediately post action (doesnt look past next players turn)"""

    def __init__(self, random_threshold=0.00):
        super().__init__(random_threshold=random_threshold)


    def get_action(self,env):

        children = env.game_state.get_children()
        best_children = []
        best_eval = -1000
        
        for child,  action_int, splendor_action in children:
            seval = child.static_eval(env.game_state.current_player)
            if seval > best_eval:
                best_children = [action_int]
                best_eval = seval
            elif seval == best_eval:
                best_children.append(action_int)

        return random.choice(best_children)

class HeuristicEvalAgent(Agent):
    
    """ Randomly selects from the best heuristic eval choices available immediately post action (doesnt look past next players turn)"""

    def __init__(self, random_threshold=0.00):
        super().__init__(random_threshold=random_threshold)

    def get_action(self,env):

        children = env.game_state.get_children()
        best_children = []
        best_eval = -1000
        
        for child,  action_int, splendor_action in children:
            heval = child.heuristic_eval(env.game_state.current_player)
            if heval > best_eval:
                best_children = [action_int]
                best_eval = heval
            elif heval == best_eval:
                best_children.append(action_int)

        return random.choice(best_children)
    
class BlendedEvalAgent(Agent):

    """ Randomly selects from the best blended eval choices available immediately post action (doesnt look past next players turn)"""

    def __init__(self, random_threshold=0.00):
        super().__init__(random_threshold=random_threshold)

    def get_action(self,env):

        children = env.game_state.get_children()
        best_children = []
        best_eval = -1000
        
        for child,  action_int, splendor_action in children:
            beval = child.blend_eval(env.game_state.current_player)
            if beval > best_eval:
                best_children = [action_int]
                best_eval = beval
            elif beval == best_eval:
                best_children.append(action_int)

        return random.choice(best_children)
    
class SearchAgent(Agent):

    def __init__(self, random_threshold=0.00):
        super().__init__(random_threshold=random_threshold)

    def get_action(self,env):

        (child, action_int), s, h, e = env.game_state.get_best_action(3)
        return action_int




