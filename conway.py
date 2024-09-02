import numpy as np
import pygame

def conway_step(X):
    """Conway's Game of Life step using numpy array X"""
    neighbors_count = sum(np.roll(np.roll(X, i, 0), j, 1)
                    for i in (-1, 0, 1) for j in (-1, 0, 1)
                    if (i != 0 or j != 0))
    return (neighbors_count == 3) | (X & (neighbors_count == 2))

def create_random_state(x, y, prob=0.5):
    """Create a random initial state of x by y grid"""
    return np.random.choice([False, True], size=(x, y), p=[1-prob, prob])

height, width = 1000, 1000
X = create_random_state(height, width, prob=0.2)

pygame.init()
screen = pygame.display.set_mode((width, height))
surface = pygame.Surface((width, height))
clock = pygame.time.Clock()

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    X = conway_step(X)
    
    pygame.surfarray.blit_array(surface, X * 0xFFFFFF)
    screen.blit(surface, (0, 0))
    
    pygame.display.flip()
    clock.tick(10)  # Limit to 10 frames per second

pygame.quit()
