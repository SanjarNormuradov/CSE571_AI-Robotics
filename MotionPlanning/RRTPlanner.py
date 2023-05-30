from __future__ import absolute_import, print_function
import numpy as np
from RRTTree import RRTTree
import sys
import math
import time

class RRTPlanner(object):
    def __init__(self, planning_env, output_directory:str, stat_filename:str, rand_seed:int, bias = 0.05, eta = 1.0, max_iter = 10000):
        self.env = planning_env         # Map Environment
        self.tree = RRTTree(planning_env=self.env, 
                            eta=eta, 
                            bias=bias, 
                            output_directory=output_directory, 
                            rand_seed=rand_seed, 
                            stat_filename=stat_filename)
        self.bias = bias                # Goal Bias
        self.max_iter = max_iter        # Max Iterations
        self.eta = eta                  # Distance to extend

    def Plan(self, start_config:np.array, goal_config:np.array):
        start_time = time.time()
        path = None
        vertex_id, vertex = None, None
        branch_id = 0
        # Generate RRT
        #   Seed the tree
        self.tree.AddVertex(start_config)
        #   Expand the tree
        for id in range(1, self.max_iter+1):
            # Randomly sample free state from the map, as long as an 
            # x_rand.shape = (self.c_space_dim, 1)
            # c_space is space of joint angles: 
            # i.e. for 2dof_robot_arm (x, y) = (joint1_angle, joint2_angle)
            x_rand = self.sample(goal_config)
            # print(f"x_rand:\n{x_rand}")
            # Get the tree vertex nearest to x_rand
            x_near_id, x_near_dist = self.tree.GetNearestVertex(x_rand)
            x_near = self.tree.vertices[x_near_id]
            # print(f"x_near:\n{x_near}")

            # Extend from x_near towards x_rand
            x_new = self.extend(x_near, x_rand)
            # print(f"x_new:\n{x_new}")

            # Check an edge between x_near and x_rand for collision/out of map
            if self.env.edge_validity_checker(x_new, x_near):
                branch_id += 1
                # Add x_new to tree vertices
                dist = self.env.compute_distance(x_near, x_new)
                self.tree.AddVertex(x_new, cost=dist)
                # Add edge between x_near and x_new
                self.tree.AddEdge(x_near_id, branch_id)

                # If x_new satisfy the goal criterion (close to the goal within some range), 
                # start looking for vertices to build the path 
                if branch_id % 200 == 0:
                    vertex_id, vertex_dist = self.tree.GetNearestVertex(goal_config)
                    vertex = self.tree.vertices[vertex_id]
                    if self.env.goal_criterion(vertex):
                        path = []
                        break

        # Double make sure that in case of different max_iter, we won't miss the path after all iterations
        vertex_id, vertex_dist = self.tree.GetNearestVertex(goal_config)
        vertex = self.tree.vertices[vertex_id]
        if self.env.goal_criterion(vertex):
            path = []

        # Search the path
        cost = 0
        if path is not None:
            print(f"\n#Iterations: {len(self.tree.vertices)-1}")
            while not np.array_equal(vertex, start_config):
                cost += self.tree.costs[vertex_id]
                path.append(tuple(vertex.flatten()))
                # Find the parent of the current vertex
                vertex_id = self.tree.edges[vertex_id]
                vertex = self.tree.vertices[vertex_id]
            path.append(tuple(vertex.flatten()))
            # Reverse the order of vertices
            path.reverse()
            path = np.array(path)
            
        end_time = time.time()
        self.tree.time_cost = end_time - start_time
        self.tree.path_cost = cost

        return path

    def extend(self, x_near:np.array, x_rand:np.array, step_size=0.01):
        # Find action that would take from x_near=(x1,y1,z1) ---> x_rand=(x2,y2,z2)
        # Compute distance between x_near and x_rand
        dist = self.env.compute_distance(x_near, x_rand)
        # Scale the distance according to self.eta
        dist *= self.eta
        # Compute x_new=(x3,y3,z3) using trigonometry
        if self.env.c_space_dim == 2:
            # Compute inclination angle
            alpha = math.atan2(x_rand[1] - x_near[1], x_rand[0] - x_near[0])
            # Compute (x3,z3)
            x_new = [x_near[0] + dist * math.cos(alpha), 
                     x_near[1] + dist * math.sin(alpha)]
        elif self.env.c_space_dim == 3:
            # Compute inclination angle
            alpha = math.atan2(x_rand[2] - x_near[2], self.env.compute_distance(x_rand[:2], x_near[:2]))
            beta = math.atan2(x_rand[1] - x_near[1], x_rand[0] - x_near[0])
            # Compute (x3,y3,z3)
            x_new = [x_near[0] + dist * math.cos(alpha) * math.cos(beta), 
                     x_near[1] + dist * math.cos(alpha) * math.sin(beta), 
                     x_near[2] + dist * math.sin(alpha)]
        # Align approximate x_new with one of grid cells 
        x_new = np.round(np.array(x_new)).astype(int)

        return x_new


    def sample(self, goal:np.array):
        # Sample random point from map
        if np.random.uniform() < self.bias:
            return goal

        return self.env.sample()