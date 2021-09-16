import numpy as np
import pyastar2d


# The start and goal coordinates are in matrix coordinates (i, j).
start = (1, 0)
goal = (3, 3)
dist_weight=0.01

# The minimum cost must be 1 for the heuristic to be valid.
weights = np.ones((4, 4), dtype=np.float32) - np.array([[1, 0, 0, 0],
                    [1, 0, 0, 1],
                    [0, 0, 0, 1],
                    [0, 1, 1, 1],], dtype=np.float32)
print("Cost matrix:")
print(weights)
path = pyastar2d.astar_path(weights, dist_weight, start, goal, allow_diagonal=False)

# The path is returned as a numpy array of (i, j) coordinates.
print(f"Best path from {start} to {goal} found:")
print(path)


start = (1, 3)
goal = (3, 3)
# The minimum cost must be 1 for the heuristic to be valid.
weights = np.ones((4, 4), dtype=np.float32) - np.array([[0, 0, 0, 0],
                    [0, 0, 0, 1],
                    [0, 0, 0, 1],
                    [0, 1, 1, 1],], dtype=np.float32)
print("Cost matrix:")
print(weights)
path = pyastar2d.astar_path(weights, dist_weight, start, goal, allow_diagonal=False)

# The path is returned as a numpy array of (i, j) coordinates.
print(f"Best path from {start} to {goal} found:")
print(path)


start = (2, 1)
goal = (3, 3)
# The minimum cost must be 1 for the heuristic to be valid.
weights = np.ones((4, 4), dtype=np.float32) - np.array([[0, 0, 0, 0],
                    [0, 1, 1, 1],
                    [0, 1, 0, 1],
                    [0, 1, 1, 1],], dtype=np.float32)
print("Cost matrix:")
print(weights)
path = pyastar2d.astar_path(weights, dist_weight, start, goal, allow_diagonal=False)

# The path is returned as a numpy array of (i, j) coordinates.
print(f"Best path from {start} to {goal} found:")
print(path)

start = (1, 2)
goal = (3, 2)
# The minimum cost must be 1 for the heuristic to be valid.
weights = np.ones((4, 4), dtype=np.float32) - np.array([[0, 0, 0, 0],
                    [0, 1, 1, 1],
                    [0, 1, 0, 1],
                    [0, 1, 1, 1],], dtype=np.float32)
print("Cost matrix:")
print(weights)
path = pyastar2d.astar_path(weights, dist_weight, start, goal, allow_diagonal=False)

# The path is returned as a numpy array of (i, j) coordinates.
print(f"Best path from {start} to {goal} found:")
print(path)