import time
import matplotlib.pyplot as plt
from maze import Maze
from astar import AStar
from bfs import BFS
from dfs import DFS

sizes = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

times_astar = []
times_bfs = []
times_dfs = []

for size in sizes:
    maze = Maze(size, size)
    start = (0, 0)
    end = (size - 1, size - 1)
    
    astar = AStar(maze, start, end)
    bfs = BFS(maze, start, end)
    dfs = DFS(maze, start, end)
    
    start_time = time.time()
    astar.solve()
    end_time = time.time()
    times_astar.append(end_time - start_time)
    
    start_time = time.time()
    bfs.solve()
    end_time = time.time()
    times_bfs.append(end_time - start_time)
    
    start_time = time.time()
    dfs.solve()
    end_time = time.time()
    times_dfs.append(end_time - start_time)

plt.plot(sizes, times_astar, label="A*")
plt.plot(sizes, times_bfs, label="BFS")
plt.plot(sizes, times_dfs, label="DFS")

plt.title("Algorithm Performance on Random Mazes")
plt.xlabel("Maze Size")
plt.ylabel("Time to Solve (seconds)")
plt.legend()
plt.show()
