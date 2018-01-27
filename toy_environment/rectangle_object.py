import pygame
import numpy as np

class RectangleObstacle(object):

    def __init__(self, image_size, top_left, bottom_right):
        self.tl = top_left
        self.br = bottom_right
        self.image_size = image_size
        #self.rect_coords = np.array([self.tl[0], self.tl[1], self.br[0] - self.tl[0], self.br[1], self.tl[1]])
        self.rect = pygame.Rect(image_size*self.tl[0], image_size*self.tl[1], image_size*(self.br[0] - self.tl[0]), image_size*(self.br[1] - self.tl[1]))

    def draw(self, screen, color):
        ims = self.image_size
        pygame.draw.rect(screen, color, (ims*self.tl[0], ims*self.tl[1], ims*(self.br[0] - self.tl[0]), ims*(self.br[1] - self.tl[1])))

    def collides(self, agent_rect):
        return self.rect.colliderect(agent_rect)