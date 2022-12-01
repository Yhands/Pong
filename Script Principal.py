import pygame
from pong import Game
import neat
import os
import pickle


class PongGame:                                  
   #-------------------------------------------------------
   #Define os objetos do jogo de pong   
    def _init_(self, window, largura, altura):         
        self.game = Game(window, largura, altura)     
        self.left_paddle = self.game.left_paddle      
        self.right_paddle = self.game.right_paddle    
        self.ball = self.game.ball                    
   #-------------------------------------------------------

    def PvP(self, genome, config): #Modo humano x humano
        run = True
        clock = pygame.time.Clock() #Clock define com que frequencia os quadros são atualizados
        while run:
            clock.tick(75)
            for event in pygame.event.get():
                if event.type == pygame.QUIT: #Se o jogo for fechado, ele vai parar de rodar 
                    run = False
                    break

            keys = pygame.key.get_pressed()
            """
            Essa sessão define quais teclas serão usadas, e as suas funções. 
            O parâmetro left especifica que paddle será movido
            O parâmetro up especifica a direção   
            """    
            if keys[pygame.K_w]:
                self.game.move_paddle(left=True, up=True)
            if keys[pygame.K_s]:
                self.game.move_paddle(left=True, up=False)

            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]:
                self.game.move_paddle(left=False, up=True)
            if keys[pygame.K_DOWN]:
                self.game.move_paddle(left=False, up=False)
            
         
            game_info = self.game.loop()
            self.game.draw(True, False)
            pygame.display.update()

        pygame.quit()



    def PvE(self, genome, config): #Humano x IA
        net = neat.nn.FeedForwardNetwork.create(genome, config)

        run = True
        clock = pygame.time.Clock()
        while run:
            clock.tick(75)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                    break

            keys = pygame.key.get_pressed()
            if keys[pygame.K_w]:
                self.game.move_paddle(left=True, up=True)
            if keys[pygame.K_s]:
                self.game.move_paddle(left=True, up=False)
            
           #----------------------------------------------------------------------------------------------------
           #Configura os controles da IA
            output = net.activate(
                (self.right_paddle.y, self.ball.y, abs(self.right_paddle.x - self.ball.x)))
            decisao = output.index(max(output))

            if decisao == 0:
                pass
            elif decisao == 1:
                self.game.move_paddle(left=False, up=True)
            else:
                self.game.move_paddle(left=False, up=False)
           #----------------------------------------------------------------------------------------------------
            game_info = self.game.loop()
            self.game.draw(True, False)
            pygame.display.update()

        pygame.quit()


    def EvE(self, genome, config): #IA x IA
        net = neat.nn.FeedForwardNetwork.create(genome, config)

        run = True
        clock = pygame.time.Clock()
        while run:
            clock.tick(75)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                    break

            output = net.activate(
                (self.left_paddle.y, self.ball.y, abs(self.left_paddle.x - self.ball.x)))
            decisao = output.index(max(output))

            if decisao == 0:
                pass
            elif decisao == 1:
                self.game.move_paddle(left=True, up=True)
            else:
                self.game.move_paddle(left=True, up=False)
            

            output = net.activate(
                (self.right_paddle.y, self.ball.y, abs(self.right_paddle.x - self.ball.x)))
            decisao = output.index(max(output))

            if decisao == 0:
                pass
            elif decisao == 1:
                self.game.move_paddle(left=False, up=True)
            else:
                self.game.move_paddle(left=False, up=False)

            game_info = self.game.loop()
            self.game.draw(True, False)
            pygame.display.update()

        pygame.quit()


#----------------------------------------------------------
#Aqui estão as configurações de cada modo. 
def PvP(config):
    largura, altura = 1000, 750
    window = pygame.display.set_mode((largura, altura))
   #^Tamanho da janela^

    with open("best.pickle", "rb") as f:
        vencedor = pickle.load(f)        
   #^Define vencedor, e o que fazer com ele^
    game = PongGame(window, largura, altura)
    game.PvP(vencedor, config)




def PvE(config):
    largura, altura = 1000, 750
    window = pygame.display.set_mode((largura, altura))

    with open("best.pickle", "rb") as f:
        vencedor = pickle.load(f)

    game = PongGame(window, largura, altura)
    game.PvE(vencedor, config)




def EvE(config):
    largura, altura = 1000, 750
    window = pygame.display.set_mode((largura, altura))

    with open("best.pickle", "rb") as f:
        vencedor = pickle.load(f)

    game = PongGame(window, largura, altura)
    game.EvE(vencedor, config)
#----------------------------------------------------------
#Esse if procura pela configuração que será usada pelo NEAT
if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config.txt")

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
#----------------------------------------------------------
    
PvP(config) #Por fim, define qual configuração será usada