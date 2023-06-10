import heapq
import matplotlib.pyplot as plt
import numpy as np
import time
import copy

BLACK = np.array([  0,   0,   0], dtype=np.uint8)
RED   = np.array([255,   0,   0], dtype=np.uint8)
BLUE  = np.array([  0,   0, 255], dtype=np.uint8)
GREEN = np.array([  0, 255,   0], dtype=np.uint8)


class Key:
    def __init__(self, cost_to_go=float('inf'), cost_to_come=float('inf')):
        self.cost_to_go = cost_to_go
        self.cost_to_come = cost_to_come

    def __lt__(self, other): # Define __lt__ for comparison with other key
        return (self.cost_to_go < other.cost_to_go) or ((self.cost_to_go == other.cost_to_go) and (self.cost_to_come <= other.cost_to_come))
    

class Node: # Define a class to represent nodes in the search graph
    def __init__(self, indices:np.array, g=float('inf'), rhs=float('inf'), hweight=1.0, key=Key(), parent=None):
        self.indices = indices.flatten()
        self.g = g
        self.rhs = rhs
        self.hweight = hweight
        self.key = key
        self.parent = parent
    
    def __lt__(self, other): # Define __lt__ for comparison with other nodes in the priority queue
        return self.key < other.key
    

class DynamicPathPlanner:
    def __init__(self, env, epsilon:float, output_directory:str):
        self.env = env
        self.epsilon = epsilon
        self.output_directory = output_directory


    def heuristics(self, start_config:np.array):
        """ Heuristic function for LPA*/D*

        @param start_config: [1 x self.env.c_space_dim] np.array of given state, where c_space_dim = {2, 3} for 2dof/3dof_robot_arm

        self.env.dist_to_goal(endpoint_position)
        @param endpoint_position: [self.env.c_space_dim x 1] np.array of indices for given state as self.env.goal_joint_state (= np.array of int indices) """
        return self.env.dist_to_goal(start_config.reshape(self.env.c_space_dim, 1))
    

    def calculate_key(self, node:Node):
        return Key(cost_to_go=min(node.g, node.rhs) + node.hweight * self.heuristics(node.indices), cost_to_come=min(node.g, node.rhs))
    

    def initialize(self, start:np.array, goal:np.array):
        self.start = Node(indices=start, g=float('inf'), rhs=0.0, hweight=self.epsilon, key=Key(), parent=None)
        self.start.key = self.calculate_key(self.start)
        self.goal = Node(indices=goal, g=float('inf'), rhs=float('inf'), hweight=self.epsilon, key=Key(), parent=None)
        self.goal.key = self.calculate_key(self.goal)
        self.queue = [self.start]
        self.explored = [self.start, self.goal]
        self.explored_indices = [tuple(self.start.indices), tuple(self.goal.indices)]
        self.visited = set()
        self.path = []
        self.horizon = set()


    def get_neighbors(self, node:Node, mode:str):
        """ Returns all of the neighbouring nodes depending on mode.

        @param node: Current node   
        @return: list[Node] """
        old_neighbors = []
        new_neighbors = []
        new_neighbor_indices = []
        """ ArmEnvironment (2dof_planar_robot | 3dof_planar_robot)
        indices = (0..num_discretize-1, 0..num_discretize-1) | (0..num_discretize-1, 0..num_discretize-1, 0..num_discretize-1)
        c_space.shape = (num_discretize, num_discretize) | (num_discretize, num_discretize, num_discretize) """
        indices = node.indices
        shape = self.env.c_space.shape 

        neighbors_indices = [] # The neighboring nodes
        for dim, idx in enumerate(indices):
            for offset in [-1, 1]: # Check for neighbors around the node
                new_idx = list(indices)
                new_idx[dim] += offset
                if (new_idx[dim] < 0) or (new_idx[dim] >= shape[dim]):
                    continue # Skip neighbors out of c_space
                elif self.env.c_space[tuple(new_idx)] == 1: # Skip neighbors in collision, i.e. obstacles
                    continue
                neighbors_indices.append(tuple(new_idx))

        for indices in neighbors_indices:
            if mode == 'successor':
                try:
                    old_neighbors.append(self.explored[self.explored_indices.index(indices)])
                except ValueError:
                    new = Node(indices=np.array(indices), g=np.inf, rhs=np.inf, hweight=self.epsilon, key=Key(), parent=node)
                    new_neighbors.append(new)
                    new_neighbor_indices.append(indices)
            elif mode == 'affected':
                try:
                    old_neighbors.append(self.explored[self.explored_indices.index(indices)])
                except ValueError:
                    continue # Skip nodes not explored yet

        self.explored.extend(new_neighbors)
        self.explored_indices.extend(new_neighbor_indices)
        self.horizon.update(new_neighbor_indices)
        return old_neighbors + new_neighbors


    def get_min_predecessor(self, current:Node):
        all_neighbors = self.get_neighbors(node=current, mode='successor')
        if len(all_neighbors) != 0:
            current.rhs = float('inf') # Reset current.rhs, because its parent is in obstruction now
            for neighbor in all_neighbors:
                if current.rhs > (neighbor.g + self.env.compute_distance(current.indices, neighbor.indices)):
                    current.rhs = neighbor.g + self.env.compute_distance(current.indices, neighbor.indices)
                    current.parent = neighbor
        else: # Current node is unreachable, enclosed in obstacles
            current.g = float('inf')
            current.rhs = float('inf')
            current.parent = None


    def update_vertex(self, node:Node):
        for open_node in self.queue: # Remove node from the 'open_set' queue
            if np.array_equal(open_node.indices, node.indices):
                self.queue.remove(open_node)
            
        if node.g != node.rhs: # Put locally inconsistent (g != rhs) vertex into the queue
            heapq.heappush(self.queue, node)
            

    def compute_shortest_path(self, ):
        """ Computes the shortest path, given self.start and self.goal

        @return path: list of tuples, where each tuple of shape (1, num_joints) that is 'int' indices in range [0, self.env.num_discretize) 
                      of discretized joint angle space [joint_lower_limit, joint_upper_limit] """
        self.num_states_expnd = 0
        self.cost = 0
        self.path.clear()
        self.visited.clear()
        self.horizon.clear()
        image = np.array(self.env.c_space, dtype=np.uint8)
        image = np.repeat(image[:, :, None], repeats=3, axis=2)
        self.image = (1-image)*255 # Revert: obstacles - black; free space - white

        while (self.queue[0].key < self.goal.key) or (self.goal.rhs != self.goal.g):
            current = heapq.heappop(self.queue)
            # print(f"\ncurrent.indices:\n{current.indices}")
            # print(f"current.g: {current.g}; current.rhs: {current.rhs}")
            # print(f"current.key.cost_to_come: {current.key.cost_to_come}; current.key.cost_to_go: {current.key.cost_to_go}")
            self.visited.add(tuple(current.indices))
            # print(f"self.visited:\n{self.visited}")
            self.num_states_expnd += 1
            if current.g > current.rhs: # Locally overconsistent -> updated cost yields shorter path than current cost -> update current cost
                current.g = current.rhs
                # print(f"current.g: {current.g}; current.rhs: {current.rhs}")
                for neighbor in self.get_neighbors(node=current, mode='successor'):
                    # print(f"\nneighbor.indices:\n{neighbor.indices}")
                    # print(f"neighbor.g: {neighbor.g}; neighbor.rhs: {neighbor.rhs}")
                    # print(f"neighbor.key.cost_to_come: {neighbor.key.cost_to_come}; neighbor.key.cost_to_go: {neighbor.key.cost_to_go}")
                    # print(f"neighbor.parent.indices: {neighbor.parent.indices if neighbor.parent is not None else None}")
                    if neighbor.rhs > current.g + self.env.compute_distance(current.indices, neighbor.indices):
                        neighbor.parent = current
                        neighbor.rhs = current.g + self.env.compute_distance(current.indices, neighbor.indices)
                        neighbor.key = self.calculate_key(neighbor)
                        # print(f"neighbor.g: {neighbor.g}; neighbor.rhs: {neighbor.rhs}")
                        # print(f"neighbor.key.cost_to_come: {neighbor.key.cost_to_come}; neighbor.key.cost_to_go: {neighbor.key.cost_to_go}")
                        # print(f"neighbor.parent.indices: {neighbor.parent.indices}")
                        # input()
                    self.update_vertex(neighbor)
            else: # Locally underconsistent -> updated cost don't yield shorter path than current cost -> restimate current cost
                current.g = np.inf
                succ = self.get_neighbors(node=current, mode='successor')
                succ.append(current)
                for neighbor in succ:
                    if not np.array_equal(neighbor.indices, self.start.indices) and np.array_equal(neighbor.parent.indices, current.indices):
                        self.get_min_predecessor(current=neighbor)
                    self.update_vertex(neighbor)

        if self.goal.g != float('inf'):
            print(f"self.goal.g: {self.goal.g}")
            path_node = self.goal
            while not np.array_equal(path_node.parent.indices, self.start.indices):
                self.path.insert(0, tuple(path_node.indices))
                path_node = path_node.parent
            self.path.insert(0, tuple(path_node.indices))
            self.cost = sum([self.env.compute_distance(self.path[i], self.path[i+1]) for i in range(len(self.path)-1)])
            print(f"self.cost: {self.cost}")
            return True

        return False
    
    def find_path(self, start:np.array, goal:np.array, max_steps=100):
        self.initialize(start=start, goal=goal)
        for i in range(max_steps):
            self.time_cost = time.time()
            if self.compute_shortest_path():
                self.time_cost = time.time() - self.time_cost
                if self.env.c_space_dim == 2:
                    self.plot(step=i)
                if self.env.follow_path(self.path):
                    print("\nGOAL IS REACHED")
                    print(f"cost: {self.cost}\n#states expanded: {self.num_states_expnd}\ntime cost: {self.time_cost:6.4f}")
                    exit()
            else:
                self.time_cost = time.time() - self.time_cost
                print("\nPATH DOES NOT EXIST")
                print(f"time elapsed: {self.time_cost:6.4f}")

            self.env.randomize_obstables() # Move the obstacles slightly

            old_c_space = copy.copy(self.env.c_space)
            self.env.calculate_c_space() # Once the obstacles have changed the c_space needs to be calculated again
            self.env.set_robot_joint(self.env.joint_indices, self.env.start_joint_state) # Set robot state where it stopped in the current path
            c_space_changes = old_c_space - self.env.c_space # np.array of self.c_space.shape where '-1' new obstacles, '1' - old obstalce (new open space)
            new_clears = np.argwhere(c_space_changes == 1)
            print(f"\nnew_clears.shape: {new_clears.shape}")
            new_obstacles = np.argwhere(c_space_changes == -1)
            print(f"new_obstacles.shape: {new_obstacles.shape}")
            # input()

            for obstruct_indices in new_obstacles: # Update nodes whose parents are now in obstruction
                for node in self.queue:
                    if np.array_equal(obstruct_indices, node.indices):
                        self.queue.remove(node)
                try:
                    # print(f"obstruct_indices: {obstruct_indices}")
                    obstruct_node = self.explored[self.explored_indices.index(tuple(obstruct_indices.tolist()))]
                    # print(f"obstruct_node: {obstruct_node}")
                    # input()
                    for affected_neighbor in self.get_neighbors(node=obstruct_node, mode='affected'):
                        if not np.array_equal(affected_neighbor.indices, self.start.indices) and np.array_equal(affected_neighbor.parent.indices, obstruct_indices):
                            # 1. If affected_neighbor is not enclosed in obstacles:
                            #    rhs is set to higher value than g, parent is relinked 
                            #    self.update_vertex will place it into self.queue where it (locally underconsistent) affects neighbors up to goal 
                            # 2. Otherwise, rhs == g = float('inf') and parent = None. self.update_vertex will remove it from self.queue if it exists there
                            self.get_min_predecessor(current=affected_neighbor) 
                            if (affected_neighbor.parent is None) and np.array_equal(affected_neighbor.indices, self.goal.indices):
                                self.time_cost = time.time() - self.time_cost
                                print("\nGOAL AFTER OBSTACLE CHANGE BECAME UNREACHABLE.\nPATH DOES NOT EXIST")
                                print(f"time elapsed: {self.time_cost:6.4f}")
                                exit()                               
                            self.update_vertex(node=affected_neighbor) # it'll be placed into self.queue and affect its neighbors up to the goal, or removed from self.queue
                except ValueError:
                    continue # Skip obstructed nodes not explored yet
            
            self.start = self.explored[self.explored_indices.index(tuple(self.env.start_joint_state.flatten()))]
            self.start.g = float('inf')
            self.start.rhs = 0.0
            self.start.parent = None
            self.start.key = self.calculate_key(self.start)
            self.update_vertex(node=self.start)


    def plot(self, step:int):
        # Plot the elements in the open set
        for indices in self.horizon:
            self.image[indices] = GREEN

        # Plot the elements in the closed set
        for indices in self.visited:
            self.image[indices] = BLUE

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
        imagename = f"{self.output_directory}/LPA*Path_2dofRobotArm_eps{self.epsilon}_obs{len(self.env.obstacles)}_nd{self.env.num_discretize}_step{step}"
        plt.savefig(imagename + '.png', dpi=300, bbox_inches='tight')
        # Free up the memory used by the image
        plt.close()
        # plt.show()
        # print("\nPress Enter to close plot visualization...")
        # input()
        return [self.cost, self.num_states_expnd, self.time_cost]

