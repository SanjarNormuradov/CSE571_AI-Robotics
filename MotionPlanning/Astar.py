import heapq
import matplotlib.pyplot as plt
import numpy as np
import time
import math
import copy

BLACK = np.array([  0,   0,   0], dtype=np.uint8)
RED   = np.array([255,   0,   0], dtype=np.uint8)
BLUE  = np.array([  0,   0, 255], dtype=np.uint8)
GREEN = np.array([  0, 255,   0], dtype=np.uint8)

# Define a class to represent nodes in the search graph
class Node:
    def __init__(self, indices:np.array, g, h, parent, epsilon):
        self.indices = indices.flatten()
        self.g = g
        self.h = h
        self.parent = parent
        self.epsilon = epsilon

    def f(self):
        return self.g + self.epsilon * self.h

    # Define __lt__ for comparison with other nodes in the priority queue
    def __lt__(self, other):
        return self.f() < other.f()


class AstarPlanner:
    def __init__(self, env, epsilon:float, output_directory:str):
        self.env = env
        self.epsilon = epsilon
        self.num_states_expnd = 0
        self.cost = 0
        self.output_directory = output_directory

        # variables for plotting
        image = np.array(self.env.c_space, dtype=np.uint8)
        image = np.repeat(image[:, :, None], repeats=3, axis=2)
        # Revert: obstacles - black; free space - white
        self.image = (1-image)*255

    def get_neighbors(self, node:Node):
        """ 
        Returns all of the neighbouring nodes.

        @param node: Current node
        
        @return: list[Node]
        """
        neighbors = []
        # ArmEnvironment (2dof_planar_robot | 3dof_planar_robot)
        #   indices = (0..num_discretize-1, 0..num_discretize-1) | (0..num_discretize-1, 0..num_discretize-1, 0..num_discretize-1)
        #   c_space.shape = (num_discretize, num_discretize) | (num_discretize, num_discretize, num_discretize) 
        indices = node.indices
        shape = self.env.c_space.shape 

        # The neighboring nodes are:
        neighbors_indices = []
        for dim, idx in enumerate(indices):
            for offset in [-1, 1]:
                new_idx = list(indices)
                # Check for neighbors around the node
                new_idx[dim] += offset
                if new_idx[dim] < 0:
                    # Skip neighbors out of c_space
                    continue
                elif new_idx[dim] >= shape[dim]:
                    # Skip neighbors out of c_space
                    continue
                neighbors_indices.append(new_idx)

        for indices in neighbors_indices:
            if self.env.c_space[tuple(indices)] == 1:
                # Skip neighbors in collision, i.e. obstacles
                continue
            neighbors.append(Node(np.array(indices), 0, 0, None, self.epsilon))

        return neighbors

    # Define the A* algorithm
    def Plan(self, start:np.array, goal:np.array):
        start_time = time.time()
        start = start.flatten()
        goal = goal.flatten()
        self.num_states_expnd = 0

        open_set = [Node(start, 0, self.h(start), None, self.epsilon)]
        self.open_set_indices = {tuple(start)}
        self.closed_set_indices = set()
        self.path = []
        while len(open_set) > 0:
            current = heapq.heappop(open_set)
            # If we reached the goal, stop the search
            if np.array_equal(current.indices, goal):
                break
            self.num_states_expnd += 1
            self.closed_set_indices.add(tuple(current.indices))

            for neighbor in self.get_neighbors(current):
                new_cost = current.g + self.env.compute_distance(current.indices, neighbor.indices)
                if (new_cost < neighbor.g) or (tuple(neighbor.indices) not in self.open_set_indices):
                    self.open_set_indices.add(tuple(neighbor.indices))
                    neighbor.g = new_cost
                    neighbor.h = self.h(neighbor.indices)
                    neighbor.parent = current
                    heapq.heappush(open_set, neighbor)

        # Connect nodes along the path starting from the goal
        self.cost = current.g
        path_node = copy.copy(current)
        while path_node.parent is not None:
            self.path.append(tuple(path_node.indices))
            path_node = copy.copy(path_node.parent)
        # Reverse the order of nodes
        self.path.reverse()
        # print(f"\nPath:\n{self.path}")
        end_time = time.time()
        self.time_cost = end_time - start_time
        # print(f"\nCost: {self.cost}\n#States Expanded: {self.num_states_expnd}")
        return self.path

    def h(self, start_config:np.array):
        """ 
        Heuristic function for A*

        @param start_config: [1 x self.env.c_space_dim] np.array of given state

        self.env.dist_to_goal(endpoint_position)
        @param endpoint_position: [self.env.c_space_dim x 1] np.array of given state as self.env.goal_joint_state
        """
        return self.env.dist_to_goal(start_config.reshape(self.env.c_space_dim, 1))
    
    def plot(self, ):
        # Plot the elements in the open set
        for index in self.open_set_indices:
            self.image[index] = GREEN

        # Plot the elements in the closed set
        for index in self.closed_set_indices:
            self.image[index] = BLUE

        # Plot the elements in the path
        for index in self.path:
            self.image[index] = RED

        self.fig = plt.figure(figsize=(10, 11))
        self.fig.suptitle(t='States visited:\nGreen - search front;\nBlue - suboptimal;\nRed - path;\nBlack - obstacles', 
                          x=.12, y=.98, ha='left', va='top', fontsize='x-large')
        self.ax = self.fig.add_subplot(111)
        self.art1 = self.fig.text(s=f"Cost = {self.cost}\n#States Expanded = {self.num_states_expnd}\nTime Cost = {self.time_cost:6.4f}",
                                 x=.36, y=.98, ha='left', va='top', fontsize='x-large')
        self.art2 = self.fig.text(s=f"Epsilon = {self.epsilon}\n#Obstacles = {len(self.env.obstacles)}\n#Discretization = {self.env.num_discretize}",
                                 x=.65, y=.98, ha='left', va='top', fontsize='x-large')
        self.im = self.ax.imshow(self.image)
        # Save the image
        imagename = f"{self.output_directory}/A*Path_2dofRobotArm_eps{self.epsilon}_obs{len(self.env.obstacles)}_nd{self.env.num_discretize}"
        plt.savefig(imagename + '.png', dpi=300, bbox_inches='tight')
        # Free up the memory used by the image
        plt.close()
        # plt.show()
        # print("\nPress Enter to close plot visualization...")
        # input()
