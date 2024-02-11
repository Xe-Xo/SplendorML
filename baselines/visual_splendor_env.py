from splendor_env import SplendorEnv, SPLENDOR_ACTIONS
import pygame

class VisualSplendorEnv(SplendorEnv):
    def __init__(self):
        super(VisualSplendorEnv, self).__init__()
        pygame.init()
        pygame.font.init()
        self.pygame_font = pygame.font.SysFont('Arial', 20, True)
        self.pygame_window = pygame.display.set_mode((1280,720), pygame.SRCALPHA)
        self.player_game_surface = None
        self.card_surfaces_map = {"tier1": [], "tier2": [], "tier3": []}
        self.player_card_surface = None
        self.gem_surface = None
        self.gem_pool_surface = None
        self.gem_image = {}
        self.gem_image["black"] = pygame.image.load("assets/gem-black.png")
        self.gem_image["white"] = pygame.image.load("assets/gem-white.png")
        self.gem_image["blue"] = pygame.image.load("assets/gem-blue.png")
        self.gem_image["red"] = pygame.image.load("assets/gem-red.png")
        self.gem_image["green"] = pygame.image.load("assets/gem-green.png")
        self.gem_image["gold"] = pygame.image.load("assets/gem-gold.png")
        self.new_state = True


    def step(self, action):
        self.new_state = True

        print(SPLENDOR_ACTIONS[action])

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        self.render()
        return super().step(action)


    def reset(self,seed):
        return super().reset(seed)


    def render(self):
        self.pygame_window.fill((255, 0, 0))
        self.pygame_window.blit(self.get_player_gem_surface(self.player_num,1280,720), (0, 0))
        self.pygame_window.blit(self.get_player_card_surface(self.player_num,1280,720), (0, 64))
        self.new_state = False
        pygame.display.flip()
        observation = self.get_obs()
        return observation

    def get_player_surface(self, player_num, width, height):
        
        # Gets the Rendered container of each player        
        pass

    def get_game_surface(self):
        # Gets the Rendered container of the game
        pass

    def make_card_surface(self, card_cost, card_points, card_color, width, height):
        # Makes a card surface
        pass

    def make_noble_surface(self, noble_points, noble_color, width, height):
        # Makes a noble surface
        pass









    def get_player_gem_surface(self,player_num,width,height):
        if self.player_game_surface is None:
            self.player_game_surface = pygame.Surface((width, height))
            self.new_state = True
        
        if self.new_state:
            self.player_game_surface.fill((255, 255, 255, 255))
            self.player_game_surface = pygame.transform.scale(self.player_game_surface, (width, height))
                
            self.player_game_surface.blit(self.gem_image["white"], (0, 0))
            self.player_game_surface.blit(self.gem_image["blue"], (64, 0))
            self.player_game_surface.blit(self.gem_image["green"], (128, 0))
            self.player_game_surface.blit(self.gem_image["red"], (192, 0))
            self.player_game_surface.blit(self.gem_image["black"], (256, 0))
            self.player_game_surface.blit(self.gem_image["gold"], (320, 0))
            text_surface_white = self.pygame_font.render(f"{self.game_state.players[player_num].gems[0]}",False,(100,100,100,255))
            text_surface_blue = self.pygame_font.render(f"{self.game_state.players[player_num].gems[1]}",False,(100,100,100,255))
            text_surface_green = self.pygame_font.render(f"{self.game_state.players[player_num].gems[2]}",False,(100,100,100,255))
            text_surface_red = self.pygame_font.render(f"{self.game_state.players[player_num].gems[3]}",False,(100,100,100,255))
            text_surface_black = self.pygame_font.render(f"{self.game_state.players[player_num].gems[4]}",False,(100,100,100,255))
            text_surface_gold = self.pygame_font.render(f"{self.game_state.players[player_num].gems[5]}",False,(100,100,100,255))
            self.player_game_surface.blit(text_surface_white,(32, 22))
            self.player_game_surface.blit(text_surface_blue,(64+32, 22))
            self.player_game_surface.blit(text_surface_green,(128+32, 22))
            self.player_game_surface.blit(text_surface_red,(192+32, 22))
            self.player_game_surface.blit(text_surface_black,(256+32, 22))
            self.player_game_surface.blit(text_surface_gold,(320+32, 22))

        return self.player_game_surface

    def get_player_card_surface(self,player_num,width,height):
        if self.player_card_surface is None:
            self.player_card_surface = pygame.Surface((width, height))
            self.new_state = True
        


        if self.new_state:
            self.player_card_surface.fill((255, 255, 255, 255))
            self.player_card_surface = pygame.transform.scale(self.player_card_surface, (width, height))
                
            pygame.draw.rect(self.player_card_surface, (100,100,100,255), (5, 2, 54, 60))
            pygame.draw.rect(self.player_card_surface, (100,100,100,255), (0+64+5, 2, 54, 60))
            pygame.draw.rect(self.player_card_surface, (100,100,100,255), (64+64+5, 2, 54, 60))
            pygame.draw.rect(self.player_card_surface, (100,100,100,255), (128+64+5, 2, 54, 60))
            pygame.draw.rect(self.player_card_surface, (100,100,100,255), (192+64+5, 2, 54, 60))

            card_count = self.game_state.players[player_num].get_card_purchase_amount()
            text_surface_white = self.pygame_font.render(f"{card_count[0]}",False,(0,0,0,255))
            text_surface_blue = self.pygame_font.render(f"{card_count[1]}",False,(0,0,0,255))
            text_surface_green = self.pygame_font.render(f"{card_count[2]}",False,(0,0,0,255))
            text_surface_red = self.pygame_font.render(f"{card_count[3]}",False,(0,0,0,255))
            text_surface_black = self.pygame_font.render(f"{card_count[4]}",False,(0,0,0,255))

            self.player_card_surface.blit(text_surface_white,(32, 22))
            self.player_card_surface.blit(text_surface_blue,(64+32, 22))
            self.player_card_surface.blit(text_surface_green,(128+32, 22))
            self.player_card_surface.blit(text_surface_red,(192+32, 22))
            self.player_card_surface.blit(text_surface_black,(256+32, 22))




        return self.player_card_surface

