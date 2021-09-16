import numpy as np
import pyastar2d


# The start and goal coordinates are in matrix coordinates (i, j).
start = (1, 0)
goal = (3, 3)

# The minimum cost must be 1 for the heuristic to be valid.
weights = np.ones((4, 4), dtype=np.float32) - np.array([[1, 0, 0, 0],
                    [1, 0, 0, 1],
                    [0, 0, 0, 1],
                    [0, 1, 1, 1],], dtype=np.float32)
print("Cost matrix:")
print(weights)
pos = pyastar2d.astar_pos(weights, start, goal, allow_diagonal=False)

# The path is returned as a numpy array of (i, j) coordinates.
print(f"Cell with max value along path from {start} to {goal}:")
print(pos)


start = (1, 3)
goal = (3, 3)
# The minimum cost must be 1 for the heuristic to be valid.
weights = np.ones((4, 4), dtype=np.float32) - np.array([[0, 0, 0, 0],
                    [0, 0, 0, 1],
                    [0, 0, 0, 1],
                    [0, 1, 1, 1],], dtype=np.float32)
print("Cost matrix:")
print(weights)
pos = pyastar2d.astar_pos(weights, start, goal, allow_diagonal=False)

# The path is returned as a numpy array of (i, j) coordinates.
print(f"Cell with max value along path from {start} to {goal}:")
print(pos)


start = (2, 1)
goal = (3, 3)
# The minimum cost must be 1 for the heuristic to be valid.
weights = np.ones((4, 4), dtype=np.float32) - np.array([[0, 0, 0, 0],
                    [0, 1, 1, 1],
                    [0, 1, 0, 1],
                    [0, 1, 1, 1],], dtype=np.float32)
print("Cost matrix:")
print(weights)
pos = pyastar2d.astar_pos(weights, start, goal, allow_diagonal=False)

# The path is returned as a numpy array of (i, j) coordinates.
print(f"Cell with max value along path from {start} to {goal}:")
print(pos)

start = (1, 2)
goal = (3, 2)
# The minimum cost must be 1 for the heuristic to be valid.
weights = np.ones((4, 4), dtype=np.float32) - np.array([[0, 0, 0, 0],
                    [0, 1, 0.6, 1],
                    [0, 1, 0.55, 1],
                    [0, 1, 1, 1],], dtype=np.float32)
print("Cost matrix:")
print(weights)
pos = pyastar2d.astar_pos(weights, start, goal, dist_weights=[0.05, 0.1], allow_diagonal=False)

# The path is returned as a numpy array of (i, j) coordinates.
print(f"Cell with max value along path from {start} to {goal}:")
print(pos)