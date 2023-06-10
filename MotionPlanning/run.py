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
    parser.add_argument('-epsn', '--epsilon_num', type=int, default=1, help='Number of different epsilon for A*')
    parser.add_argument('-eta', '--eta', type=float, default=1.0, help='eta for RRT/RRT*')
    parser.add_argument('-etan', '--eta_num', type=int, default=1, help='Number of different eta for RRT/RRT* {1.0, 0.5}')
    parser.add_argument('-b', '--bias', type=float, default=0.05, help='Goal bias for RRT/RRT*')
    parser.add_argument('-bn', '--bias_num', type=int, default=1, help='Number of different goal biases for RRT/RRT* {0.05, 0.20}')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for RRT/RRT*')
    parser.add_argument('-rs', '--rand_seed', type=int, default=1, help='Number of different (increased by +1) random seed for RRT/RRT*')
    parser.add_argument('-o', '--num_obstacles', type=int, default=2, help='Number of obstacles to add to the environment')
    parser.add_argument('-v', '--visualize', action='store_true', help='Visualize the configuration space')
    parser.add_argument('-nd', '--num_discretize', type=int, default=100, help='Number of grids the environment is divided into along each dimension')
    parser.add_argument('-ndd', '--num_diff_discretize', type=int, default=1, help='Number of different grid discretization {100, 50, 200}')
    parser.add_argument('-plt', '--plot', action='store_true', help='Visualize and save the plot')
    parser.add_argument('-unw', '--unwrapped_angle', action='store_true', help='Use unwrapped angular difference function')

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()

    if args.epsilon_num == 1:
        epsilons = [args.epsilon, ]
    elif args.epsilon_num == 2:
        epsilons = [1.0, 10.0, ]
    elif args.epsilon_num == 3:
        epsilons = [1.0, 10.0, 20.0, ]

    if args.eta_num == 1:
        etas = [args.eta, ]
    elif args.eta_num == 2:
        etas = [1.0, 0.5, ]
    elif args.eta_num == 3:
        etas = [1.0, 0.5, 0.25]

    if args.bias_num == 1:
        biases = [args.bias, ]
    elif args.bias_num == 2:
        biases = [0.05, 0.20, ]
    elif args.bias_num == 3:
        biases = [0.05, 0.10, 0.20, ]

    if args.num_diff_discretize == 1:
        discretizations = [args.num_discretize, ]
    elif args.num_diff_discretize == 2:
        discretizations = [50, 100, ]
    elif args.num_diff_discretize == 3:
        discretizations = [50, 100, 200, ]

    if args.num_obstacles == 0:
        obstacles = []
        obstacle_ids = [0, ]
    elif args.num_obstacles == 1:
        obstacles = [([-0.3, 0, 1.3], [0.25, 0.25, 0.25])]
        obstacle_ids = [0, 1, ]
    elif args.num_obstacles == 2:
        obstacles = [([0.3, 0, 0.6], [0.25, 0.25, 0.25]),
                    ([-0.3, 0, 1.3], [0.25, 0.25, 0.25]),]
        obstacle_ids = [0, 1, 2, ]

    output_directory = f"plots_{args.planner}"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    stat_filename = f"{output_directory}/statistics"

    # Clear the content of statistics.txt before reuse
    with open(stat_filename + ".txt", "w") as file:
        pass 

    for obstacle_id in obstacle_ids:
        print(f"\n#obstacles: {obstacle_id}")
        with open(stat_filename + ".txt", "a") as file:
            file.write(f"#obstacles = {obstacle_id}\n")

        if args.planner in ['astar', 'dynamic']:
            for epsilon in epsilons:
                print(f"epsilon: {epsilon}")
                with open(stat_filename + ".txt", "a") as file:
                    file.write(f"epsilon = {epsilon}\n")

                for discretization in discretizations:
                    print(f"#discr: {discretization}")
                    if args.scene == '2dof_robot_arm':
                        urdf_file = "urdf/2dof_planar_robot.urdf"
                        start_pos = (0, 0)
                        goal_pos = [0, 0, 2.0]
                        env = ArmEnvironment(urdf_file, start_pos, goal_pos, obstacles, vis_plan=args.visualize, num_discretize=discretization)
                    elif args.scene == '3dof_robot_arm':
                        urdf_file = "urdf/3dof_planar_robot.urdf"
                        start_pos = (0, 0, 0)
                        goal_pos = [0, 0, 2.0]
                        env = ArmEnvironment(urdf_file, start_pos, goal_pos, obstacles, vis_plan=args.visualize, num_discretize=discretization)
                    
                    if args.planner == 'astar':
                        from Astar import AstarPlanner
                        # Instantiate the A* algorithm
                        # A* is determenistic, no need for np.random.seed()
                        planner = AstarPlanner(env, epsilon, output_directory)

                        # Get the path from the planner
                        path = planner.Plan(env.start, env.goal)

                        if args.scene == '2dof_robot_arm':
                            cost, num_states_expnd, time_cost = planner.plot()

                        if path is not None:              
                            env.follow_path(path)
                        else:
                            print("No plan returned")

                    elif args.planner == 'dynamic':
                        from DynamicPathPlanner import DynamicPathPlanner
                        # Instantiate the D*/LPA* algorithm
                        planner = DynamicPathPlanner(env, epsilon, output_directory)
                        # IMPORTANT: This variable needs to be set to True only for dynamic path planning
                        env.change_env = True
                        # The below variable decides how often you want to replan. Change it if required
                        env.change_env_step_threshold = 5    

                        max_steps = 100
                        planner.find_path(start=env.start_joint_state, goal=env.goal_joint_state, max_steps=max_steps)
                    
                    # with open(stat_filename + ".txt", "a") as file:
                    #     file.write(f"#discr = {discretization:3d}; Cost = {cost:3d}; #States = {num_states_expnd:5d}; Time = {time_cost:6.4f}")

                    # Free memory
                    del planner

        elif args.planner.lower() in ['rrt', 'nonholrrt', 'rrtstar']:
            for eta in etas:
                print(f"eta: {eta}")
                with open(stat_filename + ".txt", "a") as file:
                    file.write(f"eta = {eta}\n")
                
                for bias in biases:
                    print(f"bias: {bias}")
                    with open(stat_filename + ".txt", "a") as file:
                        file.write(f"bias = {bias*100}%\n")

                    for discretization in discretizations:
                        print(f"#discr: {discretization}")

                        path_costs = []
                        time_costs = []
                        rand_seed = args.seed
                        while rand_seed < (args.seed + args.rand_seed):
                            print(f"seed: {rand_seed}")
                            # Keep the environment the same for different random seed planners (RRT/RRT*)
                            random.seed(rand_seed)
                            np.random.seed(rand_seed)

                            if args.scene == 'car':
                                assert args.planner == 'nonholrrt'
                                start_pos = np.array([ 40, 100, 4.71]).reshape((3, 1))
                                goal_pos  = np.array([350, 150, 1.57]).reshape((3, 1))
                                args.map = 'envs/car_map.txt'
                                env = CarEnvironment(args.map, start_pos, goal_pos, unwrapped=args.unwrapped_angle)
                                # env.init_visualizer()
                            elif args.scene == '2dof_robot_arm':
                                urdf_file = "urdf/2dof_planar_robot.urdf"
                                start_pos = (0, 0)
                                goal_pos = [0, 0, 2.0]
                                env = ArmEnvironment(urdf_file, start_pos, goal_pos, obstacles, vis_plan=args.visualize, num_discretize=discretization)
                            elif args.scene == '3dof_robot_arm':
                                urdf_file = "urdf/3dof_planar_robot.urdf"
                                start_pos = (0, 0, 0)
                                goal_pos = [0, 0, 2.0]
                                env = ArmEnvironment(urdf_file, start_pos, goal_pos, obstacles, vis_plan=args.visualize, num_discretize=discretization)
                            
                            if args.planner.lower() == 'rrt':
                                from RRTPlanner import RRTPlanner
                                # Instantiate the RRT algorithm
                                planner = RRTPlanner(env, output_directory, rand_seed, bias=bias, eta=eta)
                            elif args.planner.lower() == 'rrtstar':
                                from RRTStarPlanner import RRTStarPlanner
                                # Instantiate the RRT Star algorithm
                                planner = RRTStarPlanner(env, output_directory, rand_seed, bias=bias, eta=eta)
                            elif args.planner.lower() == 'nonholrrt':
                                from RRTPlannerNonholonomic import RRTPlannerNonholonomic
                                # Instantiate the RRT algorithm
                                planner = RRTPlannerNonholonomic(env, output_directory, rand_seed, bias=bias)

                            rand_seed += 1

                            # Get the path from the planner
                            path = planner.Plan(env.start, env.goal)

                            if args.scene == 'car':
                                # Visualize the final path.
                                tree = None
                                visited = None
                                if args.planner != 'astar':
                                    tree = planner.tree
                                    path_costs.append(tree.path_cost)
                                    time_costs.append(tree.time_cost)
                                else:
                                    visited = planner.visited
                                print(f"cost: {tree.path_cost:8.4f}\ntime cost: {tree.time_cost:8.4f}\n")
                                # print(f"Path:\n{path}\nTree: {tree}")
                                env.visualize_plan(path, tree, visited, plot=args.plot, unwrapped=args.unwrapped_angle)
                            elif (args.scene == '2dof_robot_arm') or (args.scene == '3dof_robot_arm'):
                                if args.visualize:
                                    tree = None
                                    visited = None
                                    tree = planner.tree
                                    path_costs.append(tree.path_cost)
                                    time_costs.append(tree.time_cost)
                                    print(f"cost: {tree.path_cost:8.4f}\ntime cost: {tree.time_cost:8.4f}")
                                    print(f"Path:\n{path}\nTree: {tree}")
                                    env.visualize_plan(path, tree, visited, plot=args.plot)

                                if path is not None:              
                                    env.follow_path(path)
                                else:
                                    print("No plan returned")
                            
                            # Free memory
                            del planner

                        with open(stat_filename + ".txt", "a") as file:
                            file.write(f"#discr = {discretization:3d}; Cost = [{np.mean(path_costs):8.4f}, {np.std(path_costs):8.4f}]; Time = [{np.mean(time_costs):8.4f}, {np.std(time_costs):8.4f}]\n")
         