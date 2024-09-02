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

class SimulationModel:
    def __init__(self, size: Tuple[int, int]) -> None:
        self.size = size
        self.state = create_random_state(size)

    def update(self) -> None:
        self.state = conway_step(self.state)

class SimulationView:
    def __init__(self, window_size: Tuple[int, int]) -> None:
        self.window_size = window_size
        self.screen: pygame.surface.Surface = pygame.display.set_mode(window_size)
        self.clock: pygame.time.Clock = pygame.time.Clock()

    def draw(self, state: np.ndarray) -> None:
        pygame.surfarray.blit_array(self.screen, state * 0xFFFFFF)
        pygame.display.flip()

class SimulationController:
    def __init__(self, size: Tuple[int, int]) -> None:
        self.size = size
        self.model = SimulationModel(size)
        self.view = SimulationView(size)

        pygame.init()
        self.clock: pygame.time.Clock = pygame.time.Clock()

    def handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame

    def update(self) -> None:
        self.model.update()
        self.view.draw(self.model.state)
        self.clock.tick(10)

    def close(self) -> None:
        pygame.quit()

def main():
    width: int = 1000
    height: int = 1000
    controller = SimulationController((width, height))
    running = True
    while running:
        controller.handle_events()
        controller.update()

    controller.close()

if __name__ == "__main__":
    main()
