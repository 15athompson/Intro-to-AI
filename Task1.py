import queue

# Define the map as a dictionary of dictionaries
map = {
    'A': {'B': 1, 'C': 4, 'coordinates': (0, 0)},
    'B': {'A': 1, 'C': 2, 'D': 5, 'coordinates': (1, 0)},
    'C': {'A': 4, 'B': 2, 'D': 1, 'coordinates': (2, 2)},
    'D': {'B': 5, 'C': 1, 'coordinates': (3, 3)}
}

# Define the heuristic function (straight-line distance to the goal)
def heuristic(node, goal):
    x1, y1 = map[node]['coordinates']
    x2, y2 = map[goal]['coordinates']
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

# BREADTH-FIRST SEARCH
def shortest_path(start, goal):
    # Create a queue for the nodes to be visited
    queue = []
    queue.append([start])
 
    # Loop until the queue is empty
    while queue:
        # Get the next node to visit from the queue
        path = queue.pop(0)
        node = path[-1]
 
        # Check if we have reached the goal
        if node == goal:
            return path
 
        # Check if the node has any neighbors we haven't visited yet
        for neighbor in map[node]:
            if neighbor != 'coordinates' and neighbor not in path:
                # Add the neighbor to the path and enqueue the new path
                new_path = list(path)
                new_path.append(neighbor)
                queue.append(new_path)
 
    # If we get here, there is no path between start and goal
    return None

# Usage
start = 'A'
goal = 'D'
path = shortest_path(start, goal)
print('BFS:', 'Shortest path between', start, 'and', goal, 'is:', path)



# UNIFORM-COST SEARCH
def shortest_path(start, goal):
    # Create a priority queue for the nodes to be visited
    pq = queue.PriorityQueue()
    pq.put((0, [start]))
    
    # Loop until the queue is empty
    while not pq.empty():
        # Get the next node to visit from the queue
        cost, path = pq.get()
        node = path[-1]
 
        # Check if we have reached the goal
        if node == goal:
            return path
 
        # Check if the node has any neighbors we haven't visited yet
        for neighbor, neighbor_cost in map[node].items():
            if neighbor != 'coordinates' and neighbor not in path:
                # Calculate the total cost of the new path
                new_cost = cost + neighbor_cost
 
                # Add the neighbor to the path and enqueue the new path with its total cost
                new_path = list(path)
                new_path.append(neighbor)
                pq.put((new_cost, new_path))
 
    # If we get here, there is no path between start and goal
    return None

# Usage
start = 'A'
goal = 'D'
path = shortest_path(start, goal)
print('UCS:', 'Shortest path between', start, 'and', goal, 'is:', path)



# A* SEARCH
def shortest_path(start, goal):
    # Create a priority queue for the nodes to be visited
    pq = queue.PriorityQueue()
    pq.put((0 + heuristic(start, goal), [start], 0))
    
    # Loop until the queue is empty
    while not pq.empty():
        # Get the next node to visit from the queue
        _, path, cost = pq.get()
        node = path[-1]
 
        # Check if we have reached the goal
        if node == goal:
            return path
 
        # Check if the node has any neighbors we haven't visited yet
        for neighbor, neighbor_cost in map[node].items():
            if neighbor != 'coordinates' and neighbor not in path:
                # Calculate the total cost of the new path
                new_cost = cost + neighbor_cost
                f_score = new_cost + heuristic(neighbor, goal)
 
                # Add the neighbor to the path and enqueue the new path with its total cost
                new_path = list(path)
                new_path.append(neighbor)
                pq.put((f_score, new_path, new_cost))
 
    # If we get here, there is no path between start and goal
    return None

# Usage
start = 'A'
goal = 'D'
path = shortest_path(start, goal)
print('A*:', 'Shortest path between', start, 'and', goal, 'is:', path)