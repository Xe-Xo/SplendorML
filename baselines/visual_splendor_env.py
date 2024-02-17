from splendor_env import SplendorEnv, SPLENDOR_ACTIONS, SplendorActionType
import pygame
import time
from agents.base_agent import Agent, ModelAgent


COLOR_SCHEME = [

    (255,249,233), # White
    (50,140,176), # Blue
    (125,166,106), # Green
    (216,27,63), # Red
    (49,37,38), # Black
    (255,204,0), # Gold
]


class VisualSplendorEnv(SplendorEnv):
    def __init__(self, agents=(Agent(0), Agent(1), Agent(2))):
        pygame.init()
        pygame.font.init()
        self.pygame_font = pygame.font.SysFont('Arial', 20, True)
        self.pygame_font2 = pygame.font.SysFont("Arial", 40, True)
        self.pygame_window = pygame.display.set_mode((1280,720), pygame.SRCALPHA)
        self.player_game_surfaces = [None, None, None, None]
        self.player_surface_cache = [False, False, False, False]

        self.gem_image = []
        self.gem_image.append(pygame.transform.scale(pygame.image.load("assets/gem-white.png"),(32,32)))
        self.gem_image.append(pygame.transform.scale(pygame.image.load("assets/gem-blue.png"),(32,32)))
        self.gem_image.append(pygame.transform.scale(pygame.image.load("assets/gem-green.png"),(32,32)))
        self.gem_image.append(pygame.transform.scale(pygame.image.load("assets/gem-red.png"),(32,32)))
        self.gem_image.append(pygame.transform.scale(pygame.image.load("assets/gem-black.png"),(32,32)))
        self.gem_image.append(pygame.transform.scale(pygame.image.load("assets/gem-gold.png"),(32,32)))

        super(VisualSplendorEnv, self).__init__(agents=agents)


    def action_masks(self):
        return self.game_state.action_mask

    def step(self, action):
        self.player_surface_cache = [False, False, False, False]

        #rint(SPLENDOR_ACTIONS[action])

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        self.render()
        s = super().step(action)
        #print(s[1])
        return s


    def reset(self,seed):
        return super().reset(seed)

    def render(self):
        self.pygame_window.fill((0, 0, 0))
        self.pygame_window.blit(self.get_player_surface(0,400,160), (0, 10))
        self.pygame_window.blit(self.get_player_surface(1,400,160), (0, 190))
        self.pygame_window.blit(self.get_player_surface(2,400,160), (0, 370))
        self.pygame_window.blit(self.get_player_surface(3,400,160), (0, 560))
        self.pygame_window.blit(self.get_game_surface(800,720), (400, 0))

        self.new_state = False
        pygame.display.flip()
        observation = self.get_obs()
        return observation

    def get_player_surface(self, player_num, width, height):
        
        # Gets the Rendered container of each player        

        if True: #self.player_surface_cache[player_num] == False:
            self.player_surface_cache[player_num] = True
            self.player_game_surfaces[player_num] = pygame.Surface((width, height))
            if self.player_num == player_num:
                self.player_game_surfaces[player_num].fill((100, 80, 80, 255))
            else:
                self.player_game_surfaces[player_num].fill((100, 100, 100, 255))

            # Draw the gems

            for i in range(0,6):
                for n in range(0,self.game_state.players[player_num].gems[i]):
                    self.player_game_surfaces[player_num].blit(self.gem_image[i], (10+i*32, n*10))

            # Draw the cards

            for i in range(0,5):
                for n in range(0,self.game_state.players[player_num].get_card_purchase_amount()[i]):
                    pygame.draw.rect(self.player_game_surfaces[player_num], (0,0,0,255), (i*32, n*15 + 80, 32, 32))
                    pygame.draw.rect(self.player_game_surfaces[player_num], COLOR_SCHEME[i], (i*32+1, n*15 + 80+1, 30, 30))

            # Draw the nobles
                    
            for i in range(0,5):

                pygame.draw.rect(self.player_game_surfaces[player_num], (0,0,0,255), (220 + 5 + (i*35), 10, 30, 30))
                if len(self.game_state.players[player_num].nobles) <= i:
                    pygame.draw.rect(self.player_game_surfaces[player_num], (255,255,255,255), (220 + 5 + (i*35), 10, 28, 28))
                else:
                    pygame.draw.rect(self.player_game_surfaces[player_num], (200,0,0,255), (220 + 5 + (i*35), 10, 28, 28))

                
            # Draw the points
            vps = self.game_state.players[player_num].get_victory_points()
            reserved = len(self.game_state.players[player_num].reserved_cards)
            vp_text = self.pygame_font2.render(f"VPS:{vps}/{reserved}", False, (255,0,0,255))
            self.player_game_surfaces[player_num].blit(vp_text, (250, 40))

            # Draw the Eval
            eval = self.game_state.blend_eval(player_num)
            eval_text = self.pygame_font2.render(f"Eval:{round(eval,2)}", False, (255,0,0,255))
            self.player_game_surfaces[player_num].blit(eval_text, (250, 100))

        return self.player_game_surfaces[player_num]

    def get_game_surface(self, width, height):

        if True:

        # Gets the Rendered container of the game

            self.game_surface = pygame.Surface((width, height))
            self.game_surface.fill((100, 100, 100, 255))

            # Draw the cards
            for t in range(0,3):
                for n in range(0,4):

                    if t == 0:
                        if n > len(self.game_state.tier_1_cards)-1:
                            continue
                        card = self.game_state.tier_1_cards[n]
                    elif t == 1:
                        if n > len(self.game_state.tier_2_cards)-1:
                            continue

                        card = self.game_state.tier_2_cards[n]
                    else:
                        if n > len(self.game_state.tier_3_cards)-1:
                            continue

                        card = self.game_state.tier_3_cards[n]

                    if self.game_state.players[self.player_num].can_buy(card.card_cost)[0]:
                        pygame.draw.rect(self.game_surface, (255,0,0,255), (20+n*140, 20+(2-t)*200, 120, 180))
                    else:
                        pygame.draw.rect(self.game_surface, (0,0,0,255), (20+n*140, 20+(2-t)*200, 120, 180))
                    pygame.draw.rect(self.game_surface, COLOR_SCHEME[card.gem_color], (20+n*140, 20+(2-t)*200, 110, 170))

                    pygame.draw.rect(self.game_surface, (255, 218, 117,255), (20+n*140, 20+(2-t)*200, 30, 170))
                    pygame.draw.rect(self.game_surface, (200, 218, 117,255), (20+n*140+30, 20+(2-t)*200, 30, 170))

                    # Draw the cost
                    bg_pygame_font = pygame.font.SysFont("Arial", 35, True)
                    cost_pygame_font = pygame.font.SysFont("Arial", 30, True)

                    ic = -1
                    nec = -1

                    for i in range(0,5):
                        card_cost = card.card_cost[i]
                        player_engine = self.game_state.players[self.player_num].get_card_purchase_amount()[i]
                        card_cost_minus_engine = card_cost - player_engine

                        
                        if card_cost > 0:
                            ic += 1
                            b_cost_text = bg_pygame_font.render(f"{card_cost}", False, (0,0,0,255))
                            cost_text = cost_pygame_font.render(f"{card_cost}", False, COLOR_SCHEME[i])
                            self.game_surface.blit(b_cost_text, (20+n*140+5, 20+(2-t)*200+5+ic*30-2))
                            self.game_surface.blit(cost_text, (20+n*140+5, 20+(2-t)*200+5+ic*30))

                        if card_cost_minus_engine > 0:
                            nec += 1
                            b_cost_text = bg_pygame_font.render(f"{card_cost_minus_engine}", False, (0,0,0,255))
                            cost_text = cost_pygame_font.render(f"{card_cost_minus_engine}", False, COLOR_SCHEME[i])
                            self.game_surface.blit(b_cost_text, (20+n*140+5+30, 20+(2-t)*200+5+ic*30-2))
                            self.game_surface.blit(cost_text, (20+n*140+5+30, 20+(2-t)*200+5+ic*30))


                    # Draw the VPS
                    bg_pygame_font = pygame.font.SysFont("Arial", 45, True)
                    b_vp_text = bg_pygame_font.render(f"{card.victory_points}", False, (0, 0, 0,255))
                    vp_text = self.pygame_font2.render(f"{card.victory_points}", False, (255, 218, 117,255))
                    self.game_surface.blit(b_vp_text, (20+n*140+85-2, 20+(2-t)*200))
                    self.game_surface.blit(vp_text, (20+n*140+85, 20+(2-t)*200))

            # Draw the nobles
            
            # Draw the cost
            bg_pygame_font = pygame.font.SysFont("Arial", 35, True)
            cost_pygame_font = pygame.font.SysFont("Arial", 30, True)

            for i in range(0,5):

                if i > len(self.game_state.nobles)-1:
                    continue
                noble = self.game_state.nobles[i]

                pygame.draw.rect(self.game_surface, (0,0,0,255), (600, 10+i*140 , 120, 120))
                pygame.draw.rect(self.game_surface, (255, 218, 117,255), (600, 10+i*140, 110, 110))

                ic = -1
                for c in range(0,5):
                    noble_cost = noble.noble_cost[c]
                    if noble_cost > 0:
                        ic += 1
                        b_cost_text = bg_pygame_font.render(f"{noble_cost}", False, (0,0,0,255))
                        cost_text = cost_pygame_font.render(f"{noble_cost}", False, COLOR_SCHEME[c])
                        self.game_surface.blit(b_cost_text, (600+5, 10+i*140+5+ic*30-2))
                        self.game_surface.blit(cost_text, (600+5, 10+i*140+5+ic*30))

            eval_pygame_font = pygame.font.SysFont("Arial", 30, True)

            cbeval = round(self.game_state.blend_eval(self.player_num),2)

            for (new_state, action_int, splendor_action) in self.game_state.get_children():

                beval = round(new_state.blend_eval(self.player_num),2)
                action_type, action_value = splendor_action

                if action_type == SplendorActionType.BUY_CARD:
                    
                    
                    beval_text = eval_pygame_font.render(f"{round(beval-cbeval,2)}", False, (255, 255, 255,255), (0,0,0,255))
                    self.game_surface.blit(beval_text, (action_value[0]*140+85-2, 20+130+(2-action_value[1])*200))


                elif action_type == SplendorActionType.RESERVE_CARD:

                    beval_text = eval_pygame_font.render(f"{round(beval-cbeval,2)}", False, (255, 0, 0,255), (0,0,0,255))
                    self.game_surface.blit(beval_text, (action_value[0]*140+85-2, 20+130+(2-action_value[1])*200))

            # Draw the Gem Pool
            
            pygame.draw.rect(self.game_surface, (0,0,0,255), (0, 720-100, 400, 100))

            for i in range(0,6):
                
                for n in range(0,self.game_state.gems[i]):
                
                    self.game_surface.blit(self.gem_image[i], (20+60*i, 720-32-(n*10)))

            



        return self.game_surface



