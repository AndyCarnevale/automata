from typing import Tuple
import numpy as np
import pygame

def conway_step(X: np.ndarray) -> np.ndarray:
    """Conway's Game of Life step using numpy array X"""
    neighbors_count = sum(np.roll(np.roll(X, i, 0), j, 1)
                    for i in (-1, 0, 1) for j in (-1, 0, 1)
                    if (i != 0 or j != 0))
    return (neighbors_count == 3) | (X & (neighbors_count == 2))

def create_random_state(size: Tuple[int, int], prob: float = 0.2) -> np.ndarray:
    """Create a random initial state of x by y grid"""
    return np.random.choice([False, True], size=size, p=[1-prob, prob])

class Simulation:
    def __init__(self, size: Tuple[int, int]) -> None:
        self.size = size
        self.state = create_random_state(size)
        
    def update(self) -> None:
        self.state = conway_step(self.state)

class Renderer:
    def __init__(self, window_size: Tuple[int, int]) -> None:
        self.window_size = window_size
        self.screen: pygame.surface.Surface = pygame.display.set_mode(window_size)
        self.clock: pygame.time.Clock = pygame.time.Clock()

    def draw(self, state: np.ndarray) -> None:
        pygame.surfarray.blit_array(self.screen, state * 0xFFFFFF)
        pygame.display.flip()

def main() -> None:
    width: int = 1000
    height: int = 1000
    conway: Simulation = Simulation((width, height))
    renderer: Renderer = Renderer((width, height))

    pygame.init()
    clock: pygame.time.Clock = pygame.time.Clock()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        conway.update()
        renderer.draw(conway.state)

        clock.tick(10)  # Limit to 10 frames per second

    pygame.quit()

if __name__ == "__main__":
    main()
