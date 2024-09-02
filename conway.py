from typing import Any, Tuple
from abc import ABC, abstractmethod
import numpy as np
import pygame

class InitialStateRule(ABC):
    @abstractmethod
    def rule(self, size: Tuple[int, int]) -> np.ndarray:
        pass

class UpdateRule(ABC):
    @abstractmethod
    def update(self, current_state: np.ndarray) -> np.ndarray:
        pass

class RandomInit(InitialStateRule):
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def rule(self, size) -> np.ndarray:
        return np.random.choice([0, 1], size=size, p=[1 - self.p, self.p])

class ConwayUpdate(UpdateRule):
    def update(self, current_state: np.ndarray) -> np.ndarray:
        neighbors_count = sum(np.roll(np.roll(current_state, i, 0), j, 1)
                              for i in (-1, 0, 1) for j in (-1, 0, 1)
                              if (i != 0 or j != 0))
        return (neighbors_count == 3) | (current_state & (neighbors_count == 2))

class SimulationModel:
    def __init__(self,
                 size: Tuple[int, int],
                 init_rule: InitialStateRule,
                 sim_rule: UpdateRule
                 ) -> None:
        self.size = size
        self.init_rule = init_rule
        self.sim_rule = sim_rule
        self.state = init_rule.rule(size)

    def update(self) -> None:
        self.state = self.sim_rule.update(self.state)

class SimulationView:
    def __init__(self, window_size: Tuple[int, int]) -> None:
        self.window_size = window_size
        self.screen: pygame.surface.Surface = pygame.display.set_mode(window_size)
        self.clock: pygame.time.Clock = pygame.time.Clock()

    def draw(self, state: np.ndarray) -> None:
        pygame.surfarray.blit_array(self.screen, state * 0xFFFFFF)
        pygame.display.flip()

class SimulationController:
    def __init__(self, model: SimulationModel, view: SimulationView) -> None:
        self.model = model
        self.view = view

        pygame.init()
        self.clock: pygame.time.Clock = pygame.time.Clock()

    def handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

    def update(self) -> None:
        self.model.update()
        self.view.draw(self.model.state)
        self.clock.tick(10)

    def close(self) -> None:
        pygame.quit()

def main():
    width: int = 1000
    height: int = 1000
    grid_size = (width, height)
    window_size = (width, height)

    init_rule = RandomInit(p=0.75)
    sim_rule = ConwayUpdate()
    model = SimulationModel(grid_size, init_rule, sim_rule)

    view = SimulationView(window_size)

    controller = SimulationController(model, view)

    is_running = True
    while is_running:
        is_running = controller.handle_events()
        controller.update()

    controller.close()

if __name__ == "__main__":
    main()
