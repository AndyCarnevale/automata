import numpy as np
import matplotlib.pyplot as plt

def conway_step(X):
    """Conway's Game of Life step using numpy array X"""
    neighbors_count = sum(np.roll(np.roll(X, i, 0), j, 1)
                    for i in (-1, 0, 1) for j in (-1, 0, 1)
                    if (i != 0 or j != 0))
    return (neighbors_count == 3) | (X & (neighbors_count == 2))

def create_random_state(x, y, prob=0.5):
    """Create a random initial state of x by y grid"""
    return np.random.choice([False, True], size=(x, y), p=[prob, 1-prob])

def conway_game(x=100, y=100, steps=100, prob=0.5):
    """Run Conway's Game of Life on x by y grid for steps"""
    X = create_random_state(x, y, prob)
    for _ in range(steps):
        X = conway_step(X)
        plt.imshow(X, cmap='Greys', interpolation='nearest')
        plt.show()

conway_game()