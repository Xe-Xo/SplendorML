from splendor_env import SplendorEnv, SPLENDOR_ACTIONS
import pygame

COLOR_SCHEME = [

    (255,249,233), # White
    (50,140,176), # Blue
    (125,166,106), # Green
    (216,27,63), # Red
    (49,37,38), # Black
    (255,204,0), # Gold
]


class VisualSplendorEnv(SplendorEnv):
    def __init__(self):
        super(VisualSplendorEnv, self).__init__()
        pygame.init()
        pygame.font.init()
        self.pygame_font = pygame.font.SysFont('Arial', 20, True)
        self.pygame_font2 = pygame.font.SysFont("Arial", 72, True)
        self.pygame_window = pygame.display.set_mode((1280,720), pygame.SRCALPHA)
        self.player_game_surfaces = [None, None, None, None]
        self.player_surface_cache = [False, False, False, False]

        self.gem_image = []
        self.gem_image.append(pygame.image.load("assets/gem-white.png"))
        self.gem_image.append(pygame.image.load("assets/gem-blue.png"))
        self.gem_image.append(pygame.image.load("assets/gem-green.png"))
        self.gem_image.append(pygame.image.load("assets/gem-red.png"))
        self.gem_image.append(pygame.image.load("assets/gem-black.png"))
        self.gem_image.append(pygame.image.load("assets/gem-gold.png"))



    def step(self, action):
        self.player_surface_cache = [False, False, False, False]

        print(SPLENDOR_ACTIONS[action])

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        self.render()
        return super().step(action)


    def reset(self,seed):
        return super().reset(seed)


    def render(self):
        self.pygame_window.fill((0, 0, 0))
        self.pygame_window.blit(self.get_player_surface(0,400,160), (0, 10))
        self.pygame_window.blit(self.get_player_surface(1,400,160), (0, 190))
        self.pygame_window.blit(self.get_player_surface(2,400,160), (0, 370))
        self.pygame_window.blit(self.get_player_surface(3,400,160), (0, 560))

        self.new_state = False
        pygame.display.flip()
        observation = self.get_obs()
        return observation

    def get_player_surface(self, player_num, width, height):
        
        # Gets the Rendered container of each player        

        if self.player_surface_cache[player_num] == False:
            self.player_surface_cache[player_num] = True
            self.player_game_surfaces[player_num] = pygame.Surface((width, height))
            if self.player_num == player_num:
                self.player_game_surfaces[player_num].fill((100, 80, 80, 255))
            else:
                self.player_game_surfaces[player_num].fill((100, 100, 100, 255))

            # Draw the gems

            for i in range(0,6):
                for n in range(0,self.game_state.players[player_num].gems[i]):
                    self.player_game_surfaces[player_num].blit(self.gem_image[i], (i*64, n*10))

            # Draw the cards

            for i in range(0,5):
                for n in range(0,self.game_state.players[player_num].get_card_purchase_amount()[i]):
                    pygame.draw.rect(self.player_game_surfaces[player_num], (0,0,0,255), (i*64, n*15 + 80, 60, 60))
                    pygame.draw.rect(self.player_game_surfaces[player_num], COLOR_SCHEME[i], (i*64+5, n*15 + 80+5, 50, 50))

            # Draw the nobles
                    
            for i in range(0,3):

                if len(self.game_state.players[player_num].nobles) <= i:
                    continue
                
                pygame.draw.rect(self.player_game_surfaces[player_num], (100,100,100,255), (0, 64+64, 60, 60))
            

            # Draw the points
            vps = self.game_state.players[player_num].get_victory_points()
            vp_text = self.pygame_font2.render(f"{vps} VPS", False, (255,0,0,255))
            self.player_game_surfaces[player_num].blit(vp_text, (364, 64))

        return self.player_game_surfaces[player_num]

    def get_game_surface(self):
        # Gets the Rendered container of the game
        pass

    def make_card_surface(self, card_cost, card_points, card_color, width, height):
        # Makes a card surface
        pass

    def make_noble_surface(self, noble_points, noble_color, width, height):
        # Makes a noble surface
        pass





