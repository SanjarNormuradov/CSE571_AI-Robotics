import sys
import os
from envs.ArmEnvironment import ArmEnvironment
from envs.CarEnvironment import CarEnvironment
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
import time

def get_args():
    parser = argparse.ArgumentParser(description='script for testing planners')

    parser.add_argument('-s', '--scene', type=str, default='2dof_robot_arm',
                        help='The environment to plan on, 2dof_robot_arm, 3dof_robot_arm, car')
    parser.add_argument('-m', '--map', type=str, default='envs/map2.txt',
                    help='The environment to plan on')    
    parser.add_argument('-p', '--planner', type=str, default='astar',
                        help='Please specify a planner: (astar, rrt, rrtstar, nonholrrt, dynamic)')
    # parser.add_argument('-s', '--start', nargs='+', type=float, required=True)
    # parser.add_argument('-g', '--goal', nargs='+', type=float, required=True)
    parser.add_argument('-eps', '--epsilon', type=float, default=1.0, help='Epsilon for A*')
    parser.add_argument('-eta', '--eta', type=float, default=1.0, help='eta for RRT/RRT*')
    parser.add_argument('-b', '--bias', type=float, default=0.05, help='Goal bias for RRT/RRT*')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for RRT/RRT*')
    parser.add_argument('-rs', '--rand_seed', type=int, default=1, help='Number of different (increased by +1) random seed for RRT/RRT*')
    parser.add_argument('-o', '--num_obstacles', type=int, default=2, help='Number of obstacles to add to the environment')
    parser.add_argument('-v', '--visualize', action='store_true', help='Visualize the configuration space')
    parser.add_argument('-nd', '--num_discretize', type=int, default=100, help='Number of grids the environment is divided into along each dimension')

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    output_directory = f"plots_{args.planner}"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    stat_filename = f"{output_directory}/stat_filename"

    if args.num_obstacles == 0:
        obstacles = []
    elif args.num_obstacles == 1:
        obstacles = [([-0.3, 0, 1.3], [0.25, 0.25, 0.25])]
    elif args.num_obstacles == 2:
        obstacles = [([0.3, 0, 0.6], [0.25, 0.25, 0.25]),
                    ([-0.3, 0, 1.3], [0.25, 0.25, 0.25]),]
    
    # Keep the environment the same for different random seed planners (RRT/RRT*)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.scene == 'car':
        assert args.planner == 'nonholrrt'
        start_pos = np.array([40, 100, 4.71]).reshape((3, 1))
        goal_pos = np.array([350, 150, 1.57]).reshape((3, 1))
        args.map = 'envs/car_map.txt'
        env = CarEnvironment(args.map, start_pos, goal_pos)
        env.init_visualizer()
    elif args.scene == '2dof_robot_arm':
        urdf_file = "urdf/2dof_planar_robot.urdf"
        start_pos = (0, 0)
        goal_pos = [0, 0, 2.0]
        env = ArmEnvironment(urdf_file, start_pos, goal_pos, obstacles, vis_plan=args.visualize, num_discretize=args.num_discretize)
    elif args.scene == '3dof_robot_arm':
        urdf_file = "urdf/3dof_planar_robot.urdf"
        start_pos = (0, 0, 0)
        goal_pos = [0, 0, 2.0]
        env = ArmEnvironment(urdf_file, start_pos, goal_pos, obstacles, vis_plan=args.visualize, num_discretize=args.num_discretize)

    # Clear the content of stat_filename.txt before reuse
    with open(stat_filename + ".txt", "w") as file:
        pass
    with open(stat_filename + ".txt", "a") as file:
        file.write(f"Bias = {args.bias};\tEta = {args.eta};\t#Obstacles = {args.num_obstacles};\t#Discretization = {args.num_discretize}\n\tCost\t\tTime Cost")
    rand_seed = args.seed
    while rand_seed < (args.seed + args.rand_seed):
        random.seed(rand_seed)
        np.random.seed(rand_seed)

        with open(stat_filename + ".txt", "a") as file:
            file.write(f"\n\t\tSeed = {rand_seed}")

        if args.planner == "astar":
            from Astar import AstarPlanner
            # Instantiate the A* algorithm
            # A* is determenistic, changing np.random.seed() doesn't affect its performance
            planner = AstarPlanner(env, args.epsilon, output_directory)
        if args.planner.lower() == 'rrt':
            from RRTPlanner import RRTPlanner
            # Instantiate the RRT algorithm
            planner = RRTPlanner(env, output_directory, stat_filename, rand_seed, bias=args.bias, eta=args.eta)
        if args.planner.lower() == 'rrtstar':
            from RRTStarPlanner import RRTStarPlanner
            # Instantiate the RRT Star algorithm
            planner = RRTStarPlanner(env, output_directory, stat_filename, rand_seed, bias=args.bias, eta=args.eta)
        if args.planner.lower() == 'nonholrrt':
            from RRTPlannerNonholonomic import RRTPlannerNonholonomic
            # Instantiate the RRT algorithm
            planner = RRTPlannerNonholonomic(env, bias=args.bias)
        if args.planner.lower() == 'dynamic':
            from DynamicPathPlanner import DynamicPathPlanner
            # Instantiate the D*/LPA* algorithm
            planner = DynamicPathPlanner(env)
            # IMPORTANT: This variable needs to be set to True only for dynamic path planning
            env.change_env = True
            # The below variable decides how often you want to replan. Change it if required
            # env.change_env_step_threshold = 50
        
        if args.planner == 'dynamic':
            # TODO: Extra credit implementation, Modify this IF required
            goal_reached = False
            max_steps = 100

            for i in range(max_steps):
                # Get the plan based on the start state and the goal state
                path = planner.Plan(env.start_joint_state, env.goal_joint_state)
                if path is not None:
                    # Use the planner plot if you have implemented the plot function in the dynamic path planner function
                    # planner.plot()
                    if env.follow_path(path):
                        print("Goal Reached")
                        exit()
                # This function will move the obstacles slightly
                env.randomize_obstables()

                # Once the obstacles have changed the c_space needs to be calculated again
                env.calculate_c_space()

            print("Goal not reached")

        else:
            # Get the path from the planner
            start_time = time.time()
            path = planner.Plan(env.start, env.goal)
            end_time = time.time()
            
            print(f"\nTime Cost: {(end_time-start_time):6.4f}")
            
            if args.scene == 'car':
                # Visualize the final path.
                tree = None
                visited = None
                if args.planner != 'astar':
                    tree = planner.tree
                else:
                    visited = planner.visited
                env.visualize_plan(path, tree, visited)
                plt.show()
            elif (args.scene == '2dof_robot_arm') or (args.scene == '3dof_robot_arm'):
                if (args.planner == 'astar') and (args.scene == '2dof_robot_arm'):
                    planner.plot()
                else:
                    if args.visualize:
                        # env.init_visualizer()
                        tree = None
                        visited = None
                        tree = planner.tree
                        print(f"Path:\n{path}\nTree: {tree}\nVisited: {visited}")
                        env.visualize_plan(path, tree, visited, plot=True)
                if path is not None:              
                    env.follow_path(path)
                else:
                    print("No plan returned")

        path_costs = []
        time_costs = []
        with open(stat_filename + ".txt", "r") as file:
            for line in file:
                # Skip "\n" lines
                if len(line.rstrip()) != 0:
                    if any(s in line for s in ["Bias","Seed", "Cost"]):
                        continue
                    else:
                        cost_list = line.split()
                        path_costs.append(float(cost_list[0]))
                        time_costs.append(float(cost_list[1])) 

        rand_seed += 1

    with open(stat_filename + ".txt", "a") as file:
        file.write(f"\n[{np.mean(path_costs):8.4f}, {np.std(path_costs):8.4f}]\t[{np.mean(time_costs):8.4f}, {np.std(time_costs):8.4f}]")
