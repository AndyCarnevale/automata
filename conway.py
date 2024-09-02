import numpy as np

from simulation import SimulationModel, SimulationView, SimulationController, InitialStateRule, UpdateRule

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
