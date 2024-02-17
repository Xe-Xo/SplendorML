from gymnasium import Env, spaces
from enum import Enum, Flag
import numpy as np
import random
from itertools import combinations, permutations, combinations_with_replacement
import time
import math
from typing import List, Optional

from agents.base_agent import Agent


class SplendorActionState(Enum):
    TAKE_TURN = 0
    DISCARD_GEM = 1
    VISIT_NOBLE = 2
    GAME_OVER = 3
    RANDOM_CARD = 4
    

class SplendorActionType(Enum):
    TAKE_3_GEMS = 0
    TAKE_2_GEMS = 1
    RESERVE_CARD = 2
    BUY_CARD = 3
    BUY_RESERVED_CARD = 4
    VISIT_NOBLE_CHOICE = 5
    DISCARD_GEM = 6

    def __str__(self):
        return self.name.replace('_', ' ').title()
    
    def __repr__(self) -> str:
        return self.name.replace('_', ' ').title()

class SplendorGem(Flag):
    WHITE = 1
    BLUE = 2
    GREEN = 4
    RED = 8
    BLACK = 16
    GOLD = 32

    def get_index(self):
        return [SplendorGem.WHITE, SplendorGem.BLUE, SplendorGem.GREEN, SplendorGem.RED, SplendorGem.BLACK, SplendorGem.GOLD].index(self)

    def count_types(self):
        return sum(1 for i in range(5) if self.value & (1 << i))

    def gem_index(value):
        return [SplendorGem.WHITE, SplendorGem.BLUE, SplendorGem.GREEN, SplendorGem.RED, SplendorGem.BLACK, SplendorGem.GOLD][value]


SPLENDOR_ACTIONS = [
    # 10 - Pick 3 gems of different colors
    # 5 - Pick 2 gems of same color
    # 12 - Reserve a card
    # 12 - Buy a card
    # 3 - Buy a reserved card
    # 5 - Pick which noble you visit
    # 6 - Discard a gem
]

count_actions = {}
for i in SplendorActionType:
    count_actions[i] = 0

for i in range(12):
    SPLENDOR_ACTIONS.append((SplendorActionType.BUY_CARD, (i % 4, i // 4)))
    count_actions[SplendorActionType.BUY_CARD] += 1

for i in range(3):
    SPLENDOR_ACTIONS.append((SplendorActionType.BUY_RESERVED_CARD, i))
    count_actions[SplendorActionType.BUY_RESERVED_CARD] += 1

for i in range(64):
    if SplendorGem(i).count_types() == 3 and SplendorGem(i) & SplendorGem.GOLD != SplendorGem.GOLD:
        SPLENDOR_ACTIONS.append((SplendorActionType.TAKE_3_GEMS, SplendorGem(i)))
        count_actions[SplendorActionType.TAKE_3_GEMS] += 1

for i in range(5):
    SPLENDOR_ACTIONS.append((SplendorActionType.TAKE_2_GEMS, SplendorGem.gem_index(i)))
    #print(SplendorGem.gem_index(i))
    count_actions[SplendorActionType.TAKE_2_GEMS] += 1

for i in range(12):
    SPLENDOR_ACTIONS.append((SplendorActionType.RESERVE_CARD, (i % 4, i // 4)))
    count_actions[SplendorActionType.RESERVE_CARD] += 1



for i in range(5):
    SPLENDOR_ACTIONS.append((SplendorActionType.VISIT_NOBLE_CHOICE, i))
    count_actions[SplendorActionType.VISIT_NOBLE_CHOICE] += 1

for i in range(6):
    SPLENDOR_ACTIONS.append((SplendorActionType.DISCARD_GEM, SplendorGem.gem_index(i)))
    count_actions[SplendorActionType.DISCARD_GEM] += 1


#print(count_actions)
#print(len(SPLENDOR_ACTIONS))

from sklearn import preprocessing

    
class SplendorEnv(Env):

    def __init__(self, agents=(Agent(), Agent(), Agent())):
        
        self.agents = agents



        self.action_space = spaces.Discrete(len(SPLENDOR_ACTIONS))
        self.observation_space = spaces.Dict({
            
            "action_state": spaces.Discrete(4),
            "action_mask": spaces.Box(low=0, high=1, shape=(len(SPLENDOR_ACTIONS),), dtype=np.uint8),
            "player_gems": spaces.Box(low=0, high=9, shape=(4,6), dtype=np.uint8), 

            "options_card_cost": spaces.Box(low=0, high=10, shape=(3,4,5), dtype=np.uint8), 
            "options_card_type": spaces.Box(low=0, high=5, shape=(3, 4), dtype=np.uint8),
            "options_card_points": spaces.Box(low=0, high=7, shape=(3, 4), dtype=np.uint8),
            "options_card_tier": spaces.Box(low=0, high=3, shape=(3,4), dtype=np.uint8),

            "noble_cost": spaces.Box(low=0, high=10, shape=(5,5), dtype=np.uint8),
            "noble_points": spaces.Box(low=0, high=7, shape=(5,), dtype=np.uint8),

            "player_engine": spaces.Box(low=0, high=25, shape=(4,5), dtype=np.uint8),

            #"player_card_cost": spaces.Box(low=0, high=10, shape=(4,27,5), dtype=np.uint8), 
            #"player_card_type": spaces.Box(low=0, high=5, shape=(4, 27), dtype=np.uint8),
            #"player_card_points": spaces.Box(low=0, high=7, shape=(4, 27), dtype=np.uint8),
            #"player_card_tier": spaces.Box(low=0, high=3, shape=(4, 27), dtype=np.uint8),

            "reserved_card_cost": spaces.Box(low=0, high=10, shape=(4,3,5), dtype=np.uint8), 
            "reserved_card_type": spaces.Box(low=0, high=5, shape=(4, 3), dtype=np.uint8),
            "reserved_card_points": spaces.Box(low=0, high=7, shape=(4, 3), dtype=np.uint8),
            #"reserved_card_tier": spaces.Box(low=0, high=3, shape=(4, 3), dtype=np.uint8),
      
            "player_points": spaces.Box(low=0, high=15, shape=(4,), dtype=np.uint8),

            #"all_card_cost": spaces.Box(low=0, high=10, shape=(90,5), dtype=np.uint8),
            #"all_card_type": spaces.Box(low=0, high=5, shape=(90,), dtype=np.uint8),
            #"all_card_points": spaces.Box(low=0, high=7, shape=(90,), dtype=np.uint8),
            #"all_card_tier": spaces.Box(low=0, high=3, shape=(90,), dtype=np.uint8),
            #"all_card_purchased": spaces.Box(low=0, high=5, shape=(90,)), # 0 - Available Purchase, 1 - Purchased by Current Player, 2 - Reserved by Current Player, 3 - Other Players Purchased, 4 - Other Players Reserved, 5 - Not Seen


        })
        self.player_num = None
        self.game_state = SplendorGameState()
        self.seed = None
        self.last_obs = None
        self.games_played = []
        self.turns_played = []
        self.total_games = 0
        self.env_reset = False

        for i in self.agents:
            i.setup(self)
        self.count_action_type = {}

        self.reset(None)
        


    #def action_masks(self) -> List[bool]:
    #    
    #    return [True if i == 1 else False for i in self.game_state.action_mask]

    def reset_agents(self):
        for i in self.agents:
            i.setup(self)
        self.games_played = []
        self.count_action_type = {}

    def get_obs(self):

        action_state_obs = self.game_state.get_action_state_obs()
        action_mask_obs = self.game_state.get_action_mask_obs()
        player_gems_obs = self.game_state.get_player_gems_obs()
        card_options_obs = self.game_state.get_card_options_obs()
        noble_options_obs = self.game_state.get_noble_options_obs()
        player_engine_obs = self.game_state.get_player_engine_obs()
        #player_cards_obs = self.game_state.get_player_cards_obs()
        reserved_cards_obs = self.game_state.get_reserved_cards_obs()
        player_points_obs = self.game_state.get_player_points_obs()
        #card_details_obs = self.game_state.get_cards_details_obs()

        return {
            "action_state": action_state_obs,
            "action_mask": action_mask_obs,
            "player_gems": player_gems_obs, 

            "options_card_cost": card_options_obs["card_cost"], 
            "options_card_type": card_options_obs["card_type"],
            "options_card_points": card_options_obs["card_points"],
            "options_card_tier": card_options_obs["card_tier"],

            "noble_cost": noble_options_obs["noble_cost"],
            "noble_points": noble_options_obs["noble_points"],

            "player_engine": player_engine_obs,

            #"player_card_cost": player_cards_obs["card_cost"], 
            #"player_card_type": player_cards_obs["card_type"], 
            #"player_card_points": player_cards_obs["card_points"], 
            #"player_card_tier": player_cards_obs["card_tier"], 

            "reserved_card_cost": reserved_cards_obs["card_cost"], 
            "reserved_card_type": reserved_cards_obs["card_type"], 
            "reserved_card_points": reserved_cards_obs["card_points"], 
            #"reserved_card_tier": reserved_cards_obs["card_tier"], 
      
            "player_points": player_points_obs,

            #"all_card_cost": card_details_obs["card_cost"], 
            #"all_card_type": card_details_obs["card_type"], 
            #"all_card_points": card_details_obs["card_points"], 
            #"all_card_tier": card_details_obs["card_tier"], 
        }

    def step(self, action):

        assert self.game_state.action_mask[action] == 1, "Invalid Action"
        assert self.game_state.done == False, "Game is Over"
        assert self.player_num == self.game_state.current_player, f"Player {self.player_num} is not the current player {self.game_state.current_player}"


        if SPLENDOR_ACTIONS[action][0] not in self.count_action_type.keys():
            self.count_action_type[SPLENDOR_ACTIONS[action][0]] = 1
        else:
            self.count_action_type[SPLENDOR_ACTIONS[action][0]] += 1


        if self.last_obs["action_mask"][action] != 1:
            return self.last_obs, -1000, False, self.game_state.done, {}

        current_eval = self.game_state.static_eval(self.player_num)

        self.game_state.take_action(action)


        while self.game_state.current_player != self.player_num:
            
            if self.game_state.done:
                break

            i = self.predict(self.game_state.current_player)
            self.game_state.take_action(i, skip_random=True)


        self.last_obs = self.get_obs()
        info = {}

        new_eval = self.game_state.static_eval(self.player_num)

        reward = (new_eval - current_eval)/30 * 0.01 # max static eval is 15 and min is -15 so reward can range from -1 to 1. scale it down so it doesnt compare to the end reward

        if self.game_state.done:
            self.env_reset = True
            if self.player_num in self.game_state.get_winner():
                reward = 1 / len(self.game_state.get_winner())
            else:
                reward = -1

        #print(action, reward, self.game_state.done, self.game_state.current_player, self.player_num, self.game_state.players_turns[self.player_num])
        if max(self.game_state.players_turns) > 100:
            self.env_reset = True
            print(f"TOO MANY TURNS - {[i for i in self.game_state.players_turns]} {[p.get_victory_points() for p in self.game_state.players]} \n{[self.game_state.action_list[::-1][0:10]]}")
            reward = -1

        return self.last_obs, reward, self.game_state.done, max(self.game_state.players_turns) > 100, info

    def predict(self,player_num):

        if player_num == self.player_num:
            raise Exception("Player is the current player")
        agent_num = player_num if player_num < self.player_num else player_num - 1
        return self.agents[agent_num].predict(self)

    def get_game_stats(self):

        game_stats_dict = {}

        if self.env_reset == True:

            win_rate = 0 if len(self.games_played) == 0 else round(sum(self.games_played)/len(self.games_played),2)
            average_turns = 0 if len(self.turns_played) == 0 else round(sum(self.turns_played)/len(self.turns_played),2)
            last_score = 0 if len(self.games_played) == 0 else self.games_played[-1] 

            game_stats_dict = {
                'win_rate': win_rate,
                'average_turns': average_turns,
                'total_games': self.total_games,
                'last_score': last_score,
            }

            total_count = 0
            for i in self.count_action_type.keys():
                game_stats_dict[f"action_type/{i.name}"] = self.count_action_type[i]
                total_count += self.count_action_type[i]

            for i in self.count_action_type.keys():
                game_stats_dict[f"action_type_perc/{i.name}"] = round(self.count_action_type[i] / total_count,2)
                self.count_action_type[i] = 0 


            self.env_reset = False

        return game_stats_dict

    def is_over(self):
        return self.game_state.done and self.player_num is not None

    def reset(self,seed=None, options=None):
        
        
        if self.game_state.done:
        
            if len(self.games_played) >= 10:
                self.games_played.pop(0)

            if self.player_num in self.game_state.get_winner():
                self.games_played.append(1/len(self.game_state.get_winner()))
            else:
                self.games_played.append(0)

            self.turns_played.append(max(self.game_state.players_turns))
            self.total_games += 1


        random.seed(None)
        self.player_num = random.choice([0,1,2,3])
        agents = list(self.agents)
        random.shuffle(agents)
        self.agents = tuple(agents)

        random.seed(seed)
        self.seed = seed
        self.game_state.reset(seed)

        while self.game_state.current_player != self.player_num:
            
            if self.game_state.done:
                break

            i = self.predict(self.game_state.current_player)
            self.game_state.take_action(i, skip_random=True)

        self.last_obs = self.get_obs()
        return self.last_obs, {}

    def render(self):       
        observation = self.get_obs()
        return observation

GEM_COLOR = ["WHITE", "BLUE", "GREEN", "RED", "BLACK", "GOLD"]

class Noble():
    def __init__(self, noble_cost=(0,0,0,0,0), noble_points=0):
        self.noble_cost = noble_cost
        self.noble_points = noble_points

    def __repr__(self):
        noble_string = ""
        for i in range(5):
            if self.noble_cost[i] > 0:
                noble_string += f"{GEM_COLOR[i]}:{self.noble_cost[i]} "

        return f"Noble: {noble_string} - VPS:{self.noble_points}"
    
    def copy(self):
        new = Noble(self.noble_cost, self.noble_points)
        return new

class Player():
    def __init__(self):
        self.gems = [0,0,0,0,0,0]
        self.cards = []
        self.reserved_cards = []
        self.nobles = []
        self.total_purchase_amount_cache = None
        self.cache_invalid = True

    def __repr__(self):
        return "Player: " + " ".join([f"{GEM_COLOR[i]}:{self.gems[i]}" for i in range(5)]) + f" - Cards: {len(self.cards)} - Reserved: {len(self.reserved_cards)} - Nobles: {len(self.nobles)} VPS: {self.get_victory_points()}"

    def copy(self):
        new = Player()
        new.gems = self.gems.copy()
        new.cards = self.cards.copy()
        new.reserved_cards = self.reserved_cards.copy()
        new.nobles = self.nobles.copy()
        new.total_purchase_amount_cache = self.total_purchase_amount_cache
        new.cache_invalid = self.cache_invalid

        return new

    def get_victory_points(self):
        total_points = 0
        for c in self.cards:
            total_points += c.victory_points
        for n in self.nobles:
            total_points += n.noble_points        
        return total_points

    def can_visit_noble(self, noble):

        card_purchase_amount = self.get_card_purchase_amount()
        if all([card_purchase_amount[i] >= noble.noble_cost[i] for i in range(5)]):
            return True
        return False

    def get_card_purchase_amount(self):
        
        amount = [0,0,0,0,0]
        for i in self.cards:
            amount[i.gem_color] += 1
        return amount

    def get_gem_purchase_amount(self):

        base_gems = self.gems[0:5]
        wild_gems = self.gems[5]

        if wild_gems == 0:
            return [(base_gems,0)]

        gems_combinations = []


        for comb in combinations_with_replacement([0,1,2,3,4,None], wild_gems): # (2,3,3)
            comb_gems = base_gems.copy()
            count_wild = 0
            for gidx in comb: # 2... 3... 3...
                if gidx is not None:
                    comb_gems[gidx] += 1
                    count_wild += 1

            
            gems_combinations.append((comb_gems, count_wild)) 
        
        return gems_combinations
    
    def get_total_purchase_amount(self):

        if self.cache_invalid or self.total_purchase_amount_cache is not None:

            card_purchase_amount = self.get_card_purchase_amount()
            gem_purchase_list = self.get_gem_purchase_amount()

            total_purchase_amount = []

            for i, w in gem_purchase_list:
                
                total = []
                for j in range(5):
                    total.append(card_purchase_amount[j] + i[j])

                total_purchase_amount.append((total, w))
            
            self.total_purchase_amount_cache = total_purchase_amount
            self.cache_invalid = False
            return total_purchase_amount
        else:

            return self.total_purchase_amount_cache

    def perc_visit_noble(self, noble):

        total_cost = sum(noble.noble_cost)
        calc_cost = 0
        
        card_purchase_amount = self.get_card_purchase_amount()
        for i in range(5):
            if noble.noble_cost[i] == 0:
                continue
            calc_cost = min(card_purchase_amount[i], noble.noble_cost[i])

        return calc_cost/total_cost

    def perc_buy(self, card_cost):

        # Get the cost of 
        pass

    def perc_buy_no_gems(self, card_cost):

        total_cost = sum(card_cost)
        if total_cost == 0:
            raise Exception(card_cost)
        calc_cost = 0
        
        card_purchase_amount = self.get_card_purchase_amount()
        for i in range(5):
            if card_cost[i] == 0:
                continue
            calc_cost = min(card_purchase_amount[i], card_cost[i])

        return calc_cost/total_cost

    def can_buy_no_gems(self, card_cost):
        card_purchase_amount = self.get_card_purchase_amount()
        return all([card_purchase_amount[i] >= card_cost[i] for i in range(5)])

    def can_buy(self, card_cost):

        total_purchase_amount = self.get_total_purchase_amount() 
        new_total_purchase_amount = []
        for i,w in total_purchase_amount:
            if sum(i) < sum(card_cost):
                continue
            
            if any([i[j] < card_cost[j] for j in range(5)]):
                continue

            new_total_purchase_amount.append((i,w))

        # Automatically Sort To have the lowest cost first and the lowest amount of wilds

        new_total_purchase_amount.sort(key=lambda x: (x[1], sum(x[0])))

        return len(new_total_purchase_amount) > 0, new_total_purchase_amount
        
class Card:
    def __init__(self, tier, gem_color, victory_points, white, blue, green, red, black):
        self.tier = tier
        self.gem_color = gem_color
        self.victory_points = victory_points
        self.card_cost = (white, blue, green, red, black)
        assert sum(self.card_cost) > 0, f"Card Cost is 0 {self.card_cost}"

    def __eq__(self, other) -> bool:
        return self.tier == other.tier and self.gem_color == other.gem_color and self.victory_points == other.victory_points and np.all(self.cost == other.cost) 

    def __repr__(self):
        card_cost_string = " ".join([f"{GEM_COLOR[i]}:{self.card_cost[i]}" for i in range(5) if self.card_cost[i] > 0])

        return f"Card: {card_cost_string} - Type: {GEM_COLOR[self.gem_color]} - Points: {self.victory_points} - Tier: {self.tier}"
    
    def copy(self):
        new = Card(self.tier, self.gem_color, self.victory_points, *self.card_cost)
        return new


TIER_1_DECK = [
    Card(1, 4, 0, 1, 1, 1, 1, 0),
    Card(1, 4, 0, 1, 2, 1, 1, 0),
    Card(1, 4, 0, 2, 2, 0, 1, 0),
    Card(1, 4, 0, 0, 0, 1, 3, 1),
    Card(1, 4, 0, 0, 0, 2, 1, 0),
    Card(1, 4, 0, 2, 0, 2, 0, 0),
    Card(1, 4, 0, 0, 0, 3, 0, 0),
    Card(1, 4, 1, 0, 4, 0, 0, 0),
    Card(1, 1, 0, 1, 0, 1, 1, 1),
    Card(1, 1, 0, 1, 0, 1, 2, 1),
    Card(1, 1, 0, 1, 0, 2, 2, 0),
    Card(1, 1, 0, 0, 1, 3, 1, 0),
    Card(1, 1, 0, 1, 0, 0, 0, 2),
    Card(1, 1, 0, 0, 0, 2, 0, 2),
    Card(1, 1, 0, 0, 0, 0, 0, 3),
    Card(1, 1, 1, 0, 0, 0, 4, 0),
    Card(1, 0, 0, 0, 1, 1, 1, 1),
    Card(1, 0, 0, 0, 1, 2, 1, 1),
    Card(1, 0, 0, 0, 2, 2, 0, 1),
    Card(1, 0, 0, 3, 1, 0, 0, 1),
    Card(1, 0, 0, 0, 0, 0, 2, 1),
    Card(1, 0, 0, 0, 2, 0, 0, 2),
    Card(1, 0, 0, 0, 3, 0, 0, 0),
    Card(1, 0, 1, 0, 0, 4, 0, 0),
    Card(1, 2, 0, 1, 1, 0, 1, 1),
    Card(1, 2, 0, 1, 1, 0, 1, 2),
    Card(1, 2, 0, 0, 1, 0, 2, 2),
    Card(1, 2, 0, 1, 3, 1, 0, 0),
    Card(1, 2, 0, 2, 1, 0, 0, 0),
    Card(1, 2, 0, 0, 2, 0, 2, 0),
    Card(1, 2, 0, 0, 0, 0, 3, 0),
    Card(1, 2, 1, 0, 0, 0, 0, 4),
    Card(1, 3, 0, 1, 1, 1, 0, 1),
    Card(1, 3, 0, 2, 1, 1, 0, 1),
    Card(1, 3, 0, 2, 0, 1, 0, 2),
    Card(1, 3, 0, 1, 0, 0, 1, 3),
    Card(1, 3, 0, 0, 2, 1, 0, 0),
    Card(1, 3, 0, 2, 0, 0, 2, 0),
    Card(1, 3, 0, 3, 0, 0, 0, 0),
    Card(1, 3, 1, 4, 0, 0, 0, 0),
]

TIER_2_DECK = [
    Card(2, 4, 1, 3, 2, 2, 0, 0),
    Card(2, 4, 1, 3, 0, 3, 0, 2),
    Card(2, 4, 2, 0, 1, 4, 2, 0),
    Card(2, 4, 2, 0, 0, 5, 3, 0),
    Card(2, 4, 2, 5, 0, 0, 0, 0),
    Card(2, 4, 3, 0, 0, 0, 0, 6),
    Card(2, 1, 1, 0, 2, 2, 3, 0),
    Card(2, 1, 1, 0, 2, 3, 0, 3),
    Card(2, 1, 2, 5, 3, 0, 0, 0),
    Card(2, 1, 2, 2, 0, 0, 1, 4),
    Card(2, 1, 2, 0, 5, 0, 0, 0),
    Card(2, 1, 3, 0, 6, 0, 0, 0),
    Card(2, 0, 1, 0, 0, 3, 2, 2),
    Card(2, 0, 1, 2, 3, 0, 3, 0),
    Card(2, 0, 2, 0, 0, 1, 4, 2),
    Card(2, 0, 2, 0, 0, 0, 5, 3),
    Card(2, 0, 2, 0, 0, 0, 5, 0),
    Card(2, 0, 3, 6, 0, 0, 0, 0),
    Card(2, 2, 1, 3, 0, 2, 3, 0),
    Card(2, 2, 1, 2, 3, 0, 0, 2),
    Card(2, 2, 2, 4, 2, 0, 0, 1),
    Card(2, 2, 2, 0, 5, 3, 0, 0),
    Card(2, 2, 2, 0, 0, 5, 0, 0),
    Card(2, 2, 3, 0, 0, 6, 0, 0),
    Card(2, 3, 1, 2, 0, 0, 2, 3),
    Card(2, 3, 1, 0, 3, 0, 2, 3),
    Card(2, 3, 2, 1, 4, 2, 0, 0),
    Card(2, 3, 2, 3, 0, 0, 0, 5),
    Card(2, 3, 2, 0, 0, 0, 0, 5),
    Card(2, 3, 3, 0, 0, 0, 6, 0),
]

TIER_3_DECK = [
    Card(3, 4, 3, 3, 3, 5, 3, 0),
    Card(3, 4, 4, 0, 0, 0, 7, 0),
    Card(3, 4, 4, 0, 0, 3, 6, 3),
    Card(3, 4, 5, 0, 0, 0, 7, 3),
    Card(3, 1, 3, 3, 0, 3, 3, 5),
    Card(3, 1, 4, 7, 0, 0, 0, 0),
    Card(3, 1, 4, 6, 3, 0, 0, 3),
    Card(3, 1, 5, 7, 3, 0, 0, 0),
    Card(3, 0, 3, 0, 3, 3, 5, 3),
    Card(3, 0, 4, 0, 0, 0, 0, 7),
    Card(3, 0, 4, 3, 0, 0, 3, 6),
    Card(3, 0, 5, 3, 0, 0, 0, 7),
    Card(3, 2, 3, 5, 3, 0, 3, 3),
    Card(3, 2, 4, 0, 7, 0, 0, 0),
    Card(3, 2, 4, 3, 6, 3, 0, 0),
    Card(3, 2, 5, 0, 7, 3, 0, 0),
    Card(3, 3, 3, 3, 5, 3, 0, 3),
    Card(3, 3, 4, 0, 0, 7, 0, 0),
    Card(3, 3, 4, 0, 3, 6, 3, 0),
    Card(3, 3, 5, 0, 0, 7, 3, 0),
]

NOBLES_DECK = [

    # (white, blue, green, red, black)
    Noble((0,0,0,4,4), 3), # 1 - Mary Stuart
    Noble((0,0,3,3,3), 3), # 2 - Charles V
    Noble((0,4,4,0,0), 3), # 3 - Machiavelli
    Noble((4,4,0,0,0), 3), # 4 - Isabella of Castile
    Noble((3,3,0,0,3), 3), # 5 - Suleiman the Magnificent
    Noble((0,0,4,4,0), 3), # 6 - Catherine of Medici
    Noble((0,3,3,3,0), 3), # 7 - Anne of Brittany
    Noble((4,0,0,0,4), 3), # 8 - Henry the 8th
    Noble((3,3,3,0,0), 3), # 9 - Elizabeth of Austria
    Noble((3,0,0,3,3), 3)  # 10 - Francis I of France
    
    ]

class SplendorGameState():

    def __init__(self):
        self.action_state = SplendorActionState.TAKE_TURN
        self.players = [Player(), Player(), Player(), Player()]
        self.players_turns = (0,0,0,0)
        self.current_player = 0
        self.tier_1_deck = []
        self.tier_2_deck = []
        self.tier_3_deck = []
        self.nobles_deck = []
        self.tier_1_cards = []
        self.tier_2_cards = []
        self.tier_3_cards = []
        self.nobles = []
        self.gems = []
        self.last_state = None
        self.count_no_action = 0
        self.probability = 1
        self.action_list = []
        self.done = False
        self.reset(None)
    
    def is_game_over(self):

        if self.done:
            return True
        vps = [p.get_victory_points() >= 15 for p in self.players]
        max_turns = max(self.players_turns)
        all_turns = [p >= max_turns for p in self.players_turns]
        if (any(vps) and all(all_turns)) or self.action_state == SplendorActionState.GAME_OVER:
            self.done = True

        return (any(vps) and all(all_turns)) or self.action_state == SplendorActionState.GAME_OVER

    def __repr__(self):
        return f"Players: {self.players} - Tier 1: {self.tier_1_cards} - Tier 2: {self.tier_2_cards} - Tier 3: {self.tier_3_cards} - Nobles: {self.nobles} - Gems: {self.gems}"

    def copy(self):
        new = SplendorGameState()
        new.done = self.done
        new.action_state = self.action_state
        new.players = [player.copy() for player in self.players]
        new.tier_1_deck = self.tier_1_deck.copy()
        new.tier_2_deck = self.tier_2_deck.copy()
        new.tier_3_deck = self.tier_3_deck.copy()
        new.nobles_deck = self.nobles_deck.copy()
        new.tier_1_cards = self.tier_1_cards.copy()
        new.tier_2_cards = self.tier_2_cards.copy()
        new.tier_3_cards = self.tier_3_cards.copy()
        new.nobles = self.nobles.copy()
        new.gems = self.gems.copy()
        new.current_player = self.current_player

        if self.action_mask is not None:
            new.action_mask = self.action_mask.copy()
        else:
            new.action_mask = None
        new.probability = self.probability
        new.action_list = self.action_list.copy()
        new.players_turns = tuple(self.players_turns)
        return new

    # TO DO BETTER EVALS

    def static_eval(self,player_index):
        ## Base Evaluation of State for Player
        # Really shitty static eval but its fast

        def se(pi):
            p = self.players[pi]
            return min(p.get_victory_points(),15)

        return se(player_index) - max([se(i) for i in range(4) if i != player_index])

    def heuristic_eval(self,player_index,print_info=False):
        ## Attempt Better Evalutaion of State for Player
        # Best Engine
        
        if self.is_game_over():
            return self.static_eval(player_index)

        def he(pi):


            assert pi in range(0,4), f"Player Index {pi} is not in range 0-3"

            p = self.players[pi]

            # Evaluate the Engine of the Player for the Cards remaining * VP of the Cards
            unseen_card_reward = 0
            UNSEEN_CARDS = self.tier_1_deck + self.tier_2_deck + self.tier_3_deck
            for card in UNSEEN_CARDS:
                unseen_card_reward += p.perc_buy_no_gems(card.card_cost) * card.victory_points * len(UNSEEN_CARDS)

            unpurchased_card_reward = [0]
            UNPURCHASED_CARDS = self.tier_1_cards + self.tier_2_cards + self.tier_3_cards
            #UNPURCHASED_CARDS = self.tier_3_cards
            for card in UNPURCHASED_CARDS:
                unpurchased_card_reward.append(p.perc_buy_no_gems(card.card_cost) * card.victory_points)
            unpurchased_card_reward = max(unpurchased_card_reward)
                    
            # Evaluate the Engine of the Player for their own reserved cards * VP of the Cards
            reserved_cards_reward = [0]
            for card in self.players[player_index].reserved_cards:
                reserved_cards_reward.append(p.perc_buy_no_gems(card.card_cost) * card.victory_points)
            reserved_cards_reward = max(reserved_cards_reward)

            # Evaluate the Engine of the Player for the Nobles remaining * VP of the Nobles
            noble_reward = [0]
            for i in range(len(self.nobles)):
                #print(self.nobles[i].noble_cost, p.get_card_purchase_amount(), p.perc_visit_noble(self.nobles[i])) if print_info == True else None


                noble_reward.append(p.perc_visit_noble(self.nobles[i]) * self.nobles[i].noble_points)
            noble_reward = max(noble_reward)

            #gem_reward = min(10,sum(p.gems)) / 5

            reward_list = [
                unpurchased_card_reward,
                reserved_cards_reward,
                noble_reward
            ]
            if print_info == True:
                print(reward_list, min(sum(reward_list),15))
            return min(sum(reward_list),15)
        


        return he(player_index) - max([he(i) for i in range(4) if i != player_index]) 

    def blend_eval(self,player_index):

        if self.is_game_over():
            return self.static_eval(player_index)
        
        he = self.heuristic_eval(player_index)
        se = self.static_eval(player_index)

        perc_split = (self.players_turns[player_index]-1) / max(self.players_turns[player_index]-1,25)
        #print(he, se ,perc_split, (he * (1-perc_split)) + (se * perc_split))

        return (he * (1-perc_split)) + (se * perc_split)

    def eval(self,start_player,turns=None,depth=10,alpha=-math.inf, beta=math.inf,prune=True):

        if turns == None:
            turns = self.players_turns


        # )
        if self.done or self.is_game_over() or self.action_state == SplendorActionState.GAME_OVER:

            #heuristic_eval = self.heuristic_eval(start_player)
            static_eval = self.blend_eval(start_player)

            #print(f"Depth: {depth} - Eval: {heuristic_eval} Static Eval: {static_eval}, Actions: {self.action_list}")
            
            # Static VP + Card Engine - Other Players VP - Other Players Card Engine
            #print(f"GEval {static_eval} Depth: {depth} - Alpha: {alpha} - Beta: {beta} - Prune: {prune} - Action State: {self.action_state} - Start {start_player} Player: {self.current_player} - Turns: {turns[start_player]}<-{self.players_turns[start_player]} - Action List: {self.action_list}")
            return static_eval
        #print(depth)

        if depth <= 0 or self.players_turns[start_player] >= turns[start_player] + 1:
            heuristic_eval = self.blend_eval(start_player)
            #print(f"DEval {heuristic_eval} Depth: {depth} - Alpha: {alpha} - Beta: {beta} - Prune: {prune} - Action State: {self.action_state} - Start {start_player} Player: {self.current_player} - Turns: {turns[start_player]}<-{self.players_turns[start_player]} - Action List: {self.action_list}")
            return heuristic_eval


        if self.action_state == SplendorActionState.RANDOM_CARD:
            children = self.get_children()


            # Sort by worst heuristic eval
            children.sort(key=lambda x: x[0].blend_eval(start_player))
            filter(lambda x: x[0].blend_eval(start_player) == children[0][0].blend_eval(start_player),children)
            # And randomly select from them to

            random_eval = random.choice(children)[0].eval(start_player,turns=turns,depth=depth-1,alpha=alpha,beta=beta,prune=prune)
            #print(f"REval {random_eval} Depth: {depth} - Alpha: {alpha} - Beta: {beta} - Prune: {prune} - Action State: {self.action_state} - Start {start_player} Player: {self.current_player} - Turns: {turns[start_player]}<-{self.players_turns[start_player]} - Action List: {self.action_list}")

            return random_eval
            
            total_eval = 0



            for ci, (child, action_int, splendor_action) in enumerate(children):
                total_eval += child.eval(start_player,turns=turns,depth=depth-1,alpha=alpha,beta=beta,prune=prune) * child.probability
                
                guess_eval = total_eval / (ci+1/len(children))
                beta = min(beta,guess_eval)

                if beta <= alpha and prune==True:
                    print(f"Pruned Random Leaf @ depth {depth} {alpha} {beta}")
                    break 
                else:
                    print(f"Didnt RPrune @ {depth} => {child.action_list[-10+depth:-1]}")         
            return total_eval


        if self.action_state == SplendorActionState.DISCARD_GEM:
            children = self.get_children()

            # Sort by worst heuristic eval of discarding gems
            children.sort(key=lambda x: x[0].blend_eval(start_player))
            filter(lambda x: x[0].blend_eval(start_player) == children[0][0].blend_eval(start_player),children)
            # And randomly select from them to further search

            random_eval = random.choice(children)[0].eval(start_player,turns=turns,depth=depth-1,alpha=alpha,beta=beta,prune=prune)
            #print(f"DiEval {random_eval} Depth: {depth} - Alpha: {alpha} - Beta: {beta} - Prune: {prune} - Action State: {self.action_state} - Start {start_player} Player: {self.current_player} - Turns: {turns[start_player]}<-{self.players_turns[start_player]} - Action List: {self.action_list}")

            return random_eval
            

        # Maximize the Current Player

        if start_player == self.current_player:

            max_eval = -math.inf

            children = self.get_children()

            for child, action_int, splendor_action in children:
                eval = child.eval(start_player,turns=turns,depth=depth-1,alpha=alpha,beta=beta,prune=prune)
                max_eval = max(max_eval,eval)
                alpha = max(alpha,eval)
                if beta <= alpha and prune==True:
                    #print(f"Pruned Alpha Leaf @ depth {depth} {splendor_action} {self.action_list[-1]}")
                    break


            #print(f"CEval {max_eval} Depth: {depth} - Alpha: {alpha} - Beta: {beta} - Prune: {prune} - Action State: {self.action_state} - Start {start_player} Player: {self.current_player} - Turns: {turns[start_player]}<-{self.players_turns[start_player]} - Action List: {self.action_list[-20+depth:-1]}")


            return max_eval
        
        # Minimize the Other Players

        else:

            children = self.get_children()


            # Sort by worst heuristic eval
            children.sort(key=lambda x: x[0].blend_eval(start_player)) #Worst to Best
            filter(lambda x: x[0].blend_eval(start_player) == children[0][0].blend_eval(start_player),children)
            # And randomly select from them to
            
            
            
            
            random_eval = random.choice(children)[0].eval(start_player,turns=turns,depth=depth-1,alpha=alpha,beta=beta,prune=prune)
            #print(f"OEval {random_eval} Depth: {depth} - Alpha: {alpha} - Beta: {beta} - Prune: {prune} - Action State: {self.action_state} - Start {start_player} Player: {self.current_player} - Turns: {turns[start_player]}<-{self.players_turns[start_player]} - Action List: {self.action_list}")


            return random_eval            
                        



            # min_eval = math.inf
            # for child, action_int, splendor_action in self.get_children()[::-1]:
            #     eval = child.eval(start_player,depth-1,alpha,beta)
            #     min_eval = min(min_eval,eval)
            #     beta = min(beta,eval)
            #     if beta <= alpha and prune==True:

            #         print(f"Pruned Beta @ {depth} => {child.action_list[-10+depth:-1]}")
            #         break
            #     else:
            #         print(f"Didnt Prune @ {depth} => {child.action_list[-10+depth:-1]}")
            
            # return min_eval
        
    def get_children(self):

        # Get all possible children states
        children = []

        if self.action_state == SplendorActionState.RANDOM_CARD:
            max_int = self.get_random_card_range()
            for i in range(0,max_int):
                new_state = self.copy()
                new_state.action_random_event(i)
                new_state.probability = 1/max_int
                new_state.action_state = SplendorActionState.TAKE_TURN
                new_state.current_player = (self.current_player + 1) % 4
                new_state.action_mask = new_state.get_valid_actions()
                new_state.action_list = self.action_list.copy()
                new_state.action_list.append((self.current_player,("RANDOM_CARD",i)))

                new_state.check()
                children.append((new_state, i, ("RANDOM_CARD",i)))
            
            return children
        else:
            for i in range(len(self.action_mask)):
                if self.action_mask[i] == 1:
                    new_state = self.copy()
                    new_state.take_action(i,skip_random=False)
                    new_state.probability = 1
                    new_state.action_list = self.action_list.copy()
                    new_state.action_list.append((self.current_player,SPLENDOR_ACTIONS[i]))
                    children.append((new_state, i, SPLENDOR_ACTIONS[i]))
            
            children.sort(key=lambda x: x[0].blend_eval(self.current_player), reverse=True)
            #print(f"new children {len(children)} {self.get_count_actions_dict()}")
            return children

    def get_random_best_action(self,depth=3,prune=True):
        best_children = self.get_best_options(depth=depth,prune=prune)
        return random.choice(best_children)[1]

    def get_best_action(self,depth=3,prune=True):

        best_children, eval = self.get_best_options(depth=depth,prune=prune)
        # sort them by the best heuristical eval
        best_children = sorted(best_children, key=lambda x: x[0].heuristic_eval(self.current_player), reverse=True)
        #print(best_children[0][0].heuristic_eval(self.current_player))
        #print(best_children[-1][0].heuristic_eval(self.current_player))
        # keep only the best ones
        filter(lambda x: x[0].heuristic_eval(self.current_player) == best_children[0][0].heuristic_eval(self.current_player),best_children)
        child = random.choice(best_children)
        return child, child[0].static_eval(self.current_player), child[0].heuristic_eval(self.current_player), eval  

    def get_best_options(self,depth=3,prune=True):

        # Return the Index of the Best Action

        if self.action_state == SplendorActionState.RANDOM_CARD:
            return (None,self.get_random_card_index()), None
        
        if self.action_state == SplendorActionState.GAME_OVER:
            return (None, None), None
        
        children = self.get_children()
        #print(f"Player {self.current_player} - Children: {len(children)}")

        max_eval = -math.inf
        alpha = -math.inf
        beta = math.inf

        best_children = []

        for child, action_int, splendor_action in children:
            eval = child.eval(self.current_player,turns=self.players_turns,depth=depth-1,alpha=alpha,beta=beta,prune=prune)
            if eval > max_eval:
                max_eval = eval
                best_children = []
                best_children.append((child,action_int))
            elif eval == max_eval:
                best_children.append((child,action_int))
            
            max_eval = max(max_eval,eval)
            alpha = max(alpha,eval)
            if beta <= alpha:
                #print("Pruned Top Level Alpha?")
                break
        
        return best_children, max_eval

    def get_best_random_type(self):
        if self.action_state == SplendorActionState.RANDOM_CARD:
            return self.get_random_card_index()
        
        if self.action_state == SplendorActionState.GAME_OVER:
            return None
        
        children = self.get_children()

        action_type_dict = {}
        action_type_avg = {}
        for child, action_int, splendor_action in children:
            at, av = splendor_action
            hc = child.heuristic_eval(self.current_player)
            if at not in action_type_dict.keys():
                action_type_dict[at] = [(child,action_int)]
                action_type_avg[at] = [hc]
            else:
                action_type_dict[at].append((child,action_int))
                action_type_avg[at].append(hc)

        best_action_type = None
        best_avg = -math.inf
        for k in action_type_dict.keys():
            avg = sum(action_type_avg[k])/len(action_type_avg[k])
            if best_action_type is None or avg > best_avg:
                best_action_type = k
                best_avg = avg
        

        return random.choice(action_type_dict[best_action_type])[1]



    ####

    def take_action(self, action, skip_random=True):

        
        assert self.action_state != SplendorActionState.GAME_OVER
        assert self.action_state != SplendorActionState.RANDOM_CARD
        
        current_player = self.current_player
        self.last_state = self.copy()
        self.last_state.action_list.append(action)

        assert self.action_mask[action] == 1
        action_type, action_value = SPLENDOR_ACTIONS[action]
        #print(f"Player {self.current_player} IS TAKING ACTION {action_type} - {action_value}")


        if action_type == SplendorActionType.TAKE_3_GEMS:
            self.take_3_gems(action_value)
        elif action_type == SplendorActionType.TAKE_2_GEMS:
            self.take_2_gems(action_value)
        elif action_type == SplendorActionType.RESERVE_CARD:
            self.reserve_card(action_value)
        elif action_type == SplendorActionType.BUY_CARD:
            self.buy_card(action_value)
        elif action_type == SplendorActionType.BUY_RESERVED_CARD:
            self.buy_reserved_card(action_value)
        elif action_type == SplendorActionType.VISIT_NOBLE_CHOICE:
            self.visit_noble(action_value)
        elif action_type == SplendorActionType.DISCARD_GEM:
            self.discard_gems(action_value)
        else:
            raise Exception(f"Invalid Action {action_type}")

        # Possible Returns Not to Increment Current Player

        if len(self.nobles_possible()) > 1:
            self.action_state = SplendorActionState.VISIT_NOBLE      
            self.action_mask = self.get_valid_actions()
            self.check()       
            return
        else:
            self.allocate_nobles()

        if sum(self.players[self.current_player].gems) > 10:
            #print([sum(i.gems) for i in self.players])
            self.action_state = SplendorActionState.DISCARD_GEM
            self.action_mask = self.get_valid_actions()
            self.check()
            return

        if self.random_card_required():
            if skip_random:
                self.action_state = SplendorActionState.RANDOM_CARD
                rci = self.get_random_card_index()
                self.action_random_event(rci)
            else:
                self.action_state = SplendorActionState.RANDOM_CARD
                self.action_mask = None
                self.check()
                return


        self.action_state = SplendorActionState.TAKE_TURN
        self.players_turns = [self.players_turns[0],self.players_turns[1],self.players_turns[2],self.players_turns[3]]
        self.players_turns[self.current_player] += 1
        self.players_turns = tuple(self.players_turns)
        self.current_player = (self.current_player + 1) % 4

        self.action_mask = self.get_valid_actions()
        self.check()

    def check(self):
        
        if self.action_state == SplendorActionState.RANDOM_CARD:
            assert self.random_card_required() == True, f"{(len(self.tier_1_cards),len(self.tier_1_deck),len(self.tier_2_cards),len(self.tier_2_deck),len(self.tier_3_cards),len(self.tier_3_deck) > 0)}"
        else:
            
            if self.action_mask.count(1) == 0:
                for i, (at, av) in enumerate(SPLENDOR_ACTIONS):
                    if at == SplendorActionType.TAKE_3_GEMS:
                        self.take_3_gems_valid(av, print_info=True)
            
            
            assert self.action_mask.count(1) > 0, f"{self.action_mask}, {self.action_state}, {self.current_player} {self.gems}"



        # Must always be 40 gems in the game
        assert sum([sum(i.gems) for i in self.players]) + sum(self.gems) == 40, f"Gems are not 40 \n{[[i.gems] for i in self.players]} <- {[[i.gems] for i in self.last_state.players]} \n {self.gems} <- {self.last_state.gems} "

        # Must always be Total of 40 Level 1 Cards
        a = sum([1 if i.tier == 1 else 0 for i in self.tier_1_cards])
        b = sum([1 if i.tier == 1 else 0 for i in self.tier_1_deck])
        c = 0
        for p in self.players:
            for i in p.cards:
                if i.tier == 1:
                    c += 1
            for i in p.reserved_cards:
                if i.tier == 1:
                    c += 1

        assert a+b+c == 40, f"Level 1 Cards are not 40 {a,b,c}"


        # Must always be Total of 30 Level 2 Cards
        a = sum([1 if i.tier == 2 else 0 for i in self.tier_2_cards])
        b = sum([1 if i.tier == 2 else 0 for i in self.tier_2_deck])
        c = 0
        for p in self.players:
            for i in p.cards:
                if i.tier == 2:
                    c += 1
            for i in p.reserved_cards:
                if i.tier == 2:
                    c += 1


        if a+b+c != 30:
            a1 = sum([1 if i.tier == 2 else 0 for i in self.last_state.tier_2_cards])
            b1 = sum([1 if i.tier == 2 else 0 for i in self.last_state.tier_2_deck])
            c1 = 0
            for p in self.last_state.players:
                for i in p.cards:
                    if i.tier == 2:
                        c += 1
                for i in p.reserved_cards:
                    if i.tier == 2:
                        c += 1

            #print(a,b,c)
            #print(a1,b1,c1)
            #print(self.action_state)

        assert a+b+c == 30, f"Level 2 Cards are not 30 {a,b,c}"
        

        # Must always be Total of 20 Level 3 Cards

        a = sum([1 if i.tier == 3 else 0 for i in self.tier_3_cards])
        b = sum([1 if i.tier == 3 else 0 for i in self.tier_3_deck])
        c = 0
        for p in self.players:
            for i in p.cards:
                if i.tier == 3:
                    c += 1
            for i in p.reserved_cards:
                if i.tier == 3:
                    c += 1

        assert a+b+c == 20, f"Level 3 Cards are not 20 {a,b,c}"

        # Gems should never be a negative

        assert all([i >= 0 for i in self.gems]), f"Gems are negative {self.gems}"

        # Players should never have more than 10 gems

        
        for i, p in enumerate(self.players):
            if i == self.current_player:
                if self.action_state != SplendorActionState.DISCARD_GEM:
                    assert sum(p.gems) <= 10, f"Current Player has more than 10 gems {p.gems}"
            else:
                assert sum(p.gems) <= 10, f"Player {i} has more than 10 gems {p.gems} , {self.current_player}"


        # Players should not have negative gems
        assert all([all([j >= 0 for j in i.gems]) for i in self.players]), f"Players have negative gems {[i.gems for i in self.players]}"

        if self.is_game_over():
            self.action_state = SplendorActionState.GAME_OVER

    def get_count_actions_dict(self):

        action_mask = self.get_valid_actions()
        count_actions = {}
        for i in SplendorActionType:
            count_actions[i] = 0

        for i in range(0, len(SPLENDOR_ACTIONS)):
            if action_mask[i] == 1:
                count_actions[SPLENDOR_ACTIONS[i][0]] += 1
        #print(count_actions)
        return count_actions

    def get_valid_actions(self,trim_actions=False):
        action_mask = []
        for i in range(0, len(SPLENDOR_ACTIONS)):
            action_type, action_value = SPLENDOR_ACTIONS[i]
            if action_type == SplendorActionType.TAKE_3_GEMS:
                action_mask.append(self.take_3_gems_valid(action_value) * 1)
            elif action_type == SplendorActionType.TAKE_2_GEMS:
                action_mask.append(self.take_2_gems_valid(action_value) * 1)
            elif action_type == SplendorActionType.BUY_CARD:
                action_mask.append(self.buy_card_valid(action_value) * 1)
            elif action_type == SplendorActionType.BUY_RESERVED_CARD:
                action_mask.append(self.buy_reserved_card_valid(action_value) * 1)
            elif action_type == SplendorActionType.RESERVE_CARD:
                # Remove Reserving Cards if you can take gems
                if action_mask.count(1) > 0 and trim_actions == True:
                    action_mask.append(0)
                else:
                    action_mask.append(self.reserve_card_valid(action_value) * 1)

            elif action_type == SplendorActionType.VISIT_NOBLE_CHOICE:
                action_mask.append(self.visit_noble_valid(action_value) * 1)
            elif action_type == SplendorActionType.DISCARD_GEM:
                action_mask.append(self.discard_gems_valid(action_value) * 1)
        
        action_list = [i for i in range(len(SPLENDOR_ACTIONS)) if action_mask[i] == 1]    
        action_string = " ".join([f"{i}:{SPLENDOR_ACTIONS[i]} \n" for i in action_list])

        if action_mask.count(1) == 0:
            action_mask = []
            
            for i in range(0, len(SPLENDOR_ACTIONS)):
                action_type, action_value = SPLENDOR_ACTIONS[i]
                if action_type == SplendorActionType.TAKE_3_GEMS:
                    action_mask.append(1)
                else:
                    action_mask.append(0)        

        return action_mask
 
    def get_tier_1_deck(self):
        self.tier_1_deck = [card.copy() for card in TIER_1_DECK]
        random.shuffle(self.tier_1_deck)
            
    def get_tier_2_deck(self):
        self.tier_2_deck = [card.copy() for card in TIER_2_DECK]
        random.shuffle(self.tier_2_deck)

    def get_tier_3_deck(self):
        self.tier_3_deck = [card.copy() for card in TIER_3_DECK]
        random.shuffle(self.tier_3_deck)

    def get_nobles_deck(self):
        self.nobles_deck = [card.copy() for card in NOBLES_DECK]
        random.shuffle(self.nobles_deck)

    def reset(self,seed=None):


        self.seed = seed
        random.seed(seed)
        self.action_state = SplendorActionState.TAKE_TURN
        self.players = [Player(), Player(), Player(), Player()]
        self.done = False
        self.get_tier_1_deck()
        self.get_tier_2_deck()
        self.get_tier_3_deck()
        self.get_nobles_deck()
        self.tier_1_cards = []
        self.tier_2_cards = []
        self.tier_3_cards = []
        self.nobles = []
        self.count_no_action = 0
        self.players_turns = (0,0,0,0)

        for i in range(4):
            self.tier_1_cards.append(self.tier_1_deck.pop())
            self.tier_2_cards.append(self.tier_2_deck.pop())
            self.tier_3_cards.append(self.tier_3_deck.pop())
            
        for i in range(5):
            self.nobles.append(self.nobles_deck.pop())

        self.gems = [7,7,7,7,7,5]
        self.action_mask = self.get_valid_actions()

    def random_card_required(self):
        return len(self.tier_1_cards) < 4 and len(self.tier_1_deck) > 0 or len(self.tier_2_cards) < 4 and len(self.tier_2_deck) > 0 or len(self.tier_3_cards) < 4 and len(self.tier_3_deck) > 0

    def get_random_event_prob(self):
        if self.action_state == SplendorActionState.RANDOM_CARD:
            if len(self.tier_1_cards) < 4:
                return 1/len(self.tier_1_deck)
            if len(self.tier_2_cards) < 4:
                return 1/len(self.tier_2_deck)
            if len(self.tier_3_cards) < 4:
                return 1/len(self.tier_3_deck)           
        return 1

    def action_random_event(self,random_card_index):

        assert self.action_state == SplendorActionState.RANDOM_CARD

        pre_count = len(self.tier_1_cards) + len(self.tier_2_cards) + len(self.tier_3_cards) + len(self.tier_1_deck) + len(self.tier_2_deck) + len(self.tier_3_deck)
        

        if len(self.tier_1_cards) < 4 and len(self.tier_1_deck) > 0:
            self.tier_1_cards.append(self.tier_1_deck.pop(random_card_index))
            assert len(self.tier_1_cards) == 4
            
        if len(self.tier_2_cards) < 4 and len(self.tier_2_deck) > 0:
            self.tier_2_cards.append(self.tier_2_deck.pop(random_card_index))
            assert len(self.tier_2_cards) == 4
        
        if len(self.tier_3_cards) < 4 and len(self.tier_3_deck) > 0:
            self.tier_3_cards.append(self.tier_3_deck.pop(random_card_index))
            assert len(self.tier_3_cards) == 4

        assert pre_count == len(self.tier_1_cards) + len(self.tier_2_cards) + len(self.tier_3_cards) + len(self.tier_1_deck) + len(self.tier_2_deck) + len(self.tier_3_deck)

    def get_random_card_range(self):

        assert self.action_state == SplendorActionState.RANDOM_CARD
        assert self.random_card_required() ==  True, f"Random Card Required {len(self.tier_1_cards), len(self.tier_1_deck), len(self.tier_2_cards), len(self.tier_2_deck), len(self.tier_3_cards), len(self.tier_3_deck)}"

        if len(self.tier_1_cards) < 4 and len(self.tier_1_deck) > 0:
            return len(self.tier_1_deck)
        if len(self.tier_2_cards) < 4 and len(self.tier_2_deck) > 0:
            return len(self.tier_2_deck)
        if len(self.tier_3_cards) < 4 and len(self.tier_3_deck) > 0:
            return len(self.tier_3_deck)
        
        return 0

    def get_random_card_index(self):

        # Return the index of the card to be taken

        assert self.action_state == SplendorActionState.RANDOM_CARD
        assert self.random_card_required() ==  True

        if len(self.tier_1_cards) < 4 and len(self.tier_1_deck) > 0:
            return random.randint(0,len(self.tier_1_deck)-1)

        if len(self.tier_2_cards) < 4 and len(self.tier_2_deck) > 0:
            return random.randint(0,len(self.tier_2_deck)-1)

        if len(self.tier_3_cards) < 4 and len(self.tier_3_deck) > 0:
            return random.randint(0,len(self.tier_3_deck)-1)
        

        raise Exception(f"Invalid Random Card {len(self.tier_1_cards), len(self.tier_1_deck), len(self.tier_2_cards), len(self.tier_2_deck), len(self.tier_3_cards), len(self.tier_3_deck)}")

    def convert_gems_to_binary(self,gems):
        return [int(bit) for bit in bin(gems.value)[2:].zfill(6)][::-1]

    def take_3_gems_valid(self, gems, print_info=False):

        gems = self.convert_gems_to_binary(gems)

        count_g = 0
        for i in range(5):
            if gems[i] <= self.gems[i] and gems[i]>0:
                count_g += 1
        

        if print_info:
            count_g = 0
            for i in range(5):
                if gems[i] <= self.gems[i] and gems[i]>0:
                    count_g += 1
    

            #print("VALIDATING TAKE 3 GEMS: ", gems, self.gems, self.players[self.current_player].gems, self.action_state, count_g > 1 or sum(self.gems[0:5]) == 0)

        if self.action_state != SplendorActionState.TAKE_TURN:
            return False

        # take_3_gems is valid if you are taking at least 1 gem or there are no gems available
        return (count_g > 1 or sum(self.gems[0:5]) == 0)

    def take_3_gems(self, gems):
        
        gems = self.convert_gems_to_binary(gems)

        for i in range(5):
            gems_taking = min(gems[i], self.gems[i])
            self.gems[i] -= gems_taking
            self.players[self.current_player].gems[i] += gems_taking
            self.players[self.current_player].cache_invalid = True

    def take_2_gems_valid(self, gems, print_info=False):

        if self.action_state != SplendorActionState.TAKE_TURN:
            return False

        gems = self.convert_gems_to_binary(gems)
        gems = [gems[i]*2 for i in range(5)]

        if sum(gems) != 2:
            return False

        for i in range(5):
            if gems[i] != 2:
                continue
                                   
            if self.gems[i] < 7:
                return False
            
        return True

    def take_2_gems(self, gems):
        
        gems = self.convert_gems_to_binary(gems)
        gems = [gems[i]*2 for i in range(5)]


        for i in range(5):
            self.gems[i] -= gems[i]
            self.players[self.current_player].gems[i] += gems[i]
            self.players[self.current_player].cache_invalid = True

    def reserve_card_valid(self, card, print_info=False):

        if self.action_state != SplendorActionState.TAKE_TURN:
            return False

        if self.buy_card_valid(card):
            return False

        if len(self.players[self.current_player].reserved_cards) == 3:
            return False

        card_index, tier_index = card

        if tier_index == 0:
            relevant_cards = self.tier_1_cards
        elif tier_index == 1:
            relevant_cards = self.tier_2_cards
        elif tier_index == 2:
            relevant_cards = self.tier_3_cards
        else:
            return False
        
        if len(relevant_cards) == 0:
            return False
        
        if card_index >= len(relevant_cards):
            return False

        return True

    def reserve_card(self, card):
        
        card_index, tier_index = card

        if tier_index == 0:
            relevant_cards = self.tier_1_cards
        elif tier_index == 1:
            relevant_cards = self.tier_2_cards
        elif tier_index == 2:
            relevant_cards = self.tier_3_cards
        else:
            raise Exception("Invalid Tier Index")
        
        self.players[self.current_player].reserved_cards.append(relevant_cards.pop(card_index))
        
        if self.gems[5] > 0:
            self.gems[5] -= 1
            self.players[self.current_player].gems[5] += 1
            self.players[self.current_player].cache_invalid = True

        if tier_index == 0 and len(self.tier_1_deck) > 0:
            self.tier_1_cards.append(self.tier_1_deck.pop())
        elif tier_index == 1 and len(self.tier_2_deck) > 0:
            self.tier_2_cards.append(self.tier_2_deck.pop())
        elif tier_index == 2 and len(self.tier_3_deck) > 0:
            self.tier_3_cards.append(self.tier_3_deck.pop())

    def buy_card_valid(self, card, print_info=False):

        if self.action_state != SplendorActionState.TAKE_TURN:
            return False

        card_index, tier_index = card
        if tier_index == 0:
            relevant_cards = self.tier_1_cards
        elif tier_index == 1:
            relevant_cards = self.tier_2_cards
        elif tier_index == 2:
            relevant_cards = self.tier_3_cards
        else:
            return False        

        if len(relevant_cards) == 0:
            return False

        if card_index >= len(relevant_cards):
            return False

        card = relevant_cards[card_index]
        card_cost = card.card_cost       
        can_buy, purchase_amount = self.players[self.current_player].can_buy(card_cost)
        return can_buy

    def buy_card(self, card):

        card_index, tier_index = card
        if tier_index == 0:
            relevant_cards = self.tier_1_cards
        elif tier_index == 1:
            relevant_cards = self.tier_2_cards
        elif tier_index == 2:
            relevant_cards = self.tier_3_cards
        else:
            raise Exception("Invalid Tier Index")

        if len(relevant_cards) == 0:
            raise Exception("No more cards")

        if len(relevant_cards) <= card_index:
            raise Exception("Invalid Card Index")
        
        card = relevant_cards[card_index]
        card_cost = list(card.card_cost)
 

        can_buy, purchase_amount = self.players[self.current_player].can_buy(card_cost)
        assert can_buy == True
        available , w = purchase_amount[0]

        player_card_value = self.players[self.current_player].get_card_purchase_amount()
        player_gem_value = self.players[self.current_player].gems[0:5]

        # How much player needs to spend
        gem_reduce = [0,0,0,0,0]
        wild_reduce = 0

        for i in range(5):
            card_cost[i] -= player_card_value[i]
            card_cost[i] = max(0, card_cost[i])
            # card cost without cards involved
            if card_cost[i] > 0:

                # if player has enough gems
                if player_gem_value[i] >= card_cost[i]:
                    gem_reduce[i] = card_cost[i]
                else:
                    gem_reduce[i] = player_gem_value[i]
                    wild_reduce += card_cost[i] - player_gem_value[i]

        for i in range(5):
            self.players[self.current_player].gems[i] -= gem_reduce[i]
            self.gems[i] += gem_reduce[i]

        self.players[self.current_player].gems[5] -= wild_reduce
        self.gems[5] += wild_reduce


        self.players[self.current_player].cards.append(relevant_cards.pop(card_index))

    def buy_reserved_card_valid(self, card, print_info=False):

        if self.action_state != SplendorActionState.TAKE_TURN:
            return False


        if len(self.players[self.current_player].reserved_cards) == 0:
            return False
        
        if len(self.players[self.current_player].reserved_cards) <= card:
            return False
        
        card = self.players[self.current_player].reserved_cards[card]
        card_cost = card.card_cost
        can_buy, purchase_amount = self.players[self.current_player].can_buy(card_cost)

        return can_buy

    def buy_reserved_card(self, card):

        card_index = card

        if len(self.players[self.current_player].reserved_cards) == 0:
            return False
        
        if len(self.players[self.current_player].reserved_cards) <= card:
            return False
        
        card = self.players[self.current_player].reserved_cards[card]
        card_cost = list(card.card_cost)

        can_buy, purchase_amount = self.players[self.current_player].can_buy(card_cost)
        available , w = purchase_amount[0]

        player_card_value = self.players[self.current_player].get_card_purchase_amount()
        player_gem_value = self.players[self.current_player].gems[0:5]

        gem_reduce = [0,0,0,0,0]
        wild_reduce = 0

        for i in range(5):
            if card_cost[i] > player_card_value[i]:
                gem_reduce[i] = card_cost[i] - player_card_value[i]
            
            if gem_reduce[i] > player_gem_value[i]:
                wild_reduce += gem_reduce[i] - player_gem_value[i]
                gem_reduce[i] -= gem_reduce[i] - player_gem_value[i]

        for i in range(5):
            self.players[self.current_player].gems[i] -= gem_reduce[i]
            self.gems[i] += gem_reduce[i]

        self.players[self.current_player].gems[5] -= wild_reduce
        self.gems[5] += wild_reduce

        self.players[self.current_player].cards.append(self.players[self.current_player].reserved_cards.pop(card_index))

    def allocate_nobles(self):
        for i,n in enumerate(self.nobles):
            if self.players[self.current_player].can_visit_noble(n):
                self.players[self.current_player].nobles.append(self.nobles.pop(i))
                #print(f"Player {self.current_player} had a noble visit {n}")
                break

    def nobles_possible(self):
        nobles = []
        for noble in self.nobles:
            if self.players[self.current_player].can_visit_noble(noble):
                nobles.append(noble)
        return nobles

    def visit_noble_valid(self, noble, print_info=False):

        
        
        if self.action_state != SplendorActionState.VISIT_NOBLE:
            return False
        
        #print(f"VISIT NOBLE VALIDATION {noble} {len(self.nobles)} {len(self.nobles_possible())}")

        if len(self.nobles_possible()) == 0:
            return False
        
        if noble >= len(self.nobles):
            return False
        
        return self.players[self.current_player].can_visit_noble(self.nobles[noble])

    def visit_noble(self, noble):
        self.players[self.current_player].nobles.append(self.nobles.pop(noble))

    def discard_gems_valid(self, gems, print_info=False):
        if self.action_state != SplendorActionState.DISCARD_GEM:
            return False
        
        if sum(self.players[self.current_player].gems) <= 10:

            #print([sum(i.gems) for i in self.players])

            raise Exception("How is it getting to here?")
        
        i = gems.get_index()

        if self.players[self.current_player].gems[i] <= 0:
            return False

        return True

    def discard_gems(self, gems):

        if self.action_state != SplendorActionState.DISCARD_GEM:
            raise Exception("Not in Discard Gem State")

        i = gems.get_index()
        self.gems[i] += 1
        self.players[self.current_player].gems[i] -= 1

        assert self.players[self.current_player].gems[i] >= 0

    def get_winner(self):
        winners = [i for i in range(4) if self.players[i].get_victory_points() >= 15]
        return winners

    # Observations for Environment

    def get_action_state_obs(self):
        return int(self.action_state.value)

    def get_action_mask_obs(self):
        return np.array(self.action_mask, dtype=np.uint8)
    
    def get_player_gems_obs(self):
        return np.array([self.players[i % 4].gems for p in range(self.current_player, self.current_player+4)], dtype=np.uint8)

    def get_card_options_obs(self):
        card_options_obs = {}

        card_cost = []
        card_points = []
        card_tier = []
        card_type = []

        for tier in [self.tier_1_cards,self.tier_2_cards,self.tier_3_cards]:

            cc = []
            cp = []
            cti = []
            cty = []

            for card in tier:
                cc.append(card.card_cost)
                cp.append(card.victory_points)
                cti.append(card.tier)
                cty.append(card.gem_color)

            for i in range(4-len(tier)):
                cc.append([0,0,0,0,0])
                cp.append(0)
                cti.append(0)
                cty.append(0)                

            card_cost.append(cc.copy())
            card_points.append(cp.copy())
            card_tier.append(cti.copy())
            card_type.append(cty.copy())

        card_options_obs["card_cost"] = np.array(card_cost, dtype=np.uint8)
        card_options_obs["card_points"] = np.array(card_points, dtype=np.uint8)
        card_options_obs["card_tier"] = np.array(card_tier, dtype=np.uint8)
        card_options_obs["card_type"] = np.array(card_type, dtype=np.uint8)
        return card_options_obs

    def get_noble_options_obs(self):
        nobles_options_obs = {}

        noble_cost = [noble.noble_cost for noble in self.nobles]
        noble_points = [noble.noble_points for noble in self.nobles]

        for i in range(5-len(self.nobles)):
            noble_cost.append([0,0,0,0,0])
            noble_points.append(0)


        nobles_options_obs["noble_cost"] = np.array(noble_cost, dtype=np.uint8)
        nobles_options_obs["noble_points"] = np.array(noble_points, dtype=np.uint8)
        return nobles_options_obs

    def get_player_engine_obs(self):

        # sort in order of upcoming players

        player_engine_obs = np.array([self.players[pi % 4].get_card_purchase_amount() for pi in range(self.current_player, self.current_player+4)], dtype=np.uint8)
        return player_engine_obs

    def get_player_cards_obs(self):

        # sort in order of upcoming players

        player_cards_obs = {}

        card_cost = []
        card_points = []
        card_tier = []
        card_type = []

        for p_i in range(self.current_player, self.current_player+4):

            p = self.players[p_i % 4]

            cc = []
            cp = []
            cti = []
            cty = []

            for card in p.cards:
                cc.append(card.card_cost)
                cp.append(card.victory_points)
                cti.append(card.tier)
                cty.append(card.gem_color)

            for i in range(27-len(p.cards)) :
                cc.append([0,0,0,0,0])
                cp.append(0)
                cti.append(0)
                cty.append(0)

            card_cost.append(cc.copy())
            card_points.append(cp.copy())
            card_tier.append(cti.copy())
            card_type.append(cty.copy())

        player_cards_obs["card_cost"] = np.array(card_cost, dtype=np.uint8)
        player_cards_obs["card_points"] = np.array(card_points, dtype=np.uint8)
        player_cards_obs["card_tier"] = np.array(card_tier, dtype=np.uint8)
        player_cards_obs["card_type"] = np.array(card_type, dtype=np.uint8)

        return player_cards_obs

    def get_player_points_obs(self):
        return np.array([min(self.players[p % 4].get_victory_points(),15) for p in range(self.current_player, self.current_player+4)], dtype=np.uint8)

    def get_reserved_cards_obs(self):

        # sort in order of upcoming players

        player_cards_obs = {}

        card_cost = []
        card_points = []
        card_tier = []
        card_type = []

        for p_i in range(self.current_player, self.current_player+4):

            p = self.players[p_i % 4]

            cc = []
            cp = []
            cti = []
            cty = []

            for card in p.reserved_cards:
                cc.append(card.card_cost)
                cp.append(card.victory_points)
                cti.append(card.tier)
                cty.append(card.gem_color)

            for i in range(3-len(p.reserved_cards)) :
                cc.append([0,0,0,0,0])
                cp.append(0)
                cti.append(0)
                cty.append(0)

            card_cost.append(cc.copy())
            card_points.append(cp.copy())
            card_tier.append(cti.copy())
            card_type.append(cty.copy())


        player_cards_obs["card_cost"] = np.array(card_cost, dtype=np.uint8)
        player_cards_obs["card_points"] = np.array(card_points, dtype=np.uint8)
        player_cards_obs["card_tier"] = np.array(card_tier, dtype=np.uint8)
        player_cards_obs["card_type"] = np.array(card_type, dtype=np.uint8)

        return player_cards_obs

    def get_cards_details_obs(self):

        # sort in order of upcoming players

        cards_details_obs = {}

        card_cost = []
        card_points = []
        card_tier = []
        card_type = []



        for card in TIER_1_DECK + TIER_2_DECK + TIER_3_DECK:
            card_cost.append(card.card_cost)
            card_points.append(card.victory_points)
            card_tier.append(card.tier)
            card_type.append(card.gem_color)


        cards_details_obs["card_cost"] = np.array(card_cost,dtype=np.uint8)
        cards_details_obs["card_points"] = np.array(card_points,dtype=np.uint8)
        cards_details_obs["card_tier"] = np.array(card_tier, dtype=np.uint8)
        cards_details_obs["card_type"] = np.array(card_type, dtype=np.uint8)

        return cards_details_obs





          




        



