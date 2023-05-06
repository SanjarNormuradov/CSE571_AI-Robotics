""" Written by Brian Hou for CSE571: Probabilistic Robotics (Winter 2019)
    Modified by Wentao Yuan for CSE571: Probabilistic Robotics (Spring 2022)
    Modified by Aaron Walsman and Zoey Chen for CSEP590A: Robotics (Spring 2023)
"""

import time
import math
import argparse
import numpy as np
import matplotlib.pyplot as plt

from utils import minimized_angle
from soccer_field import Field
import policies
from ekf import ExtendedKalmanFilter
from pf import ParticleFilter
from PIL import Image


def localize(
    mean_filename,
    env,
    policy,
    filt,
    x0,
    num_steps,
    plot=False,
    step_pause=0.,
    step_breakpoint=False,
):
    
    # Collect data from an entire rollout
    (states_noisefree,
     states_real,
     action_noisefree,
     obs_noisefree,
     obs_real) = env.rollout(x0, policy, num_steps)
    states_filter = np.zeros(states_real.shape)
    states_filter[0, :] = x0.ravel()

    errors = np.zeros((num_steps, 3))
    position_errors = np.zeros(num_steps)
    mahalanobis_errors = np.zeros(num_steps)

    for i in range(num_steps):
        x_real = states_real[i+1, :].reshape((-1, 1))
        u_noisefree = action_noisefree[i, :].reshape((-1, 1))
        z_real = obs_real[i, :].reshape((-1, 1))
        marker_id = env.get_marker_id(i)

        if filt is None:
            mean, cov = x_real, np.eye(3)
        else:
            # filters only know the action and observation
            mean, cov = filt.update(env, u_noisefree, z_real, marker_id)
        states_filter[i+1, :] = mean.ravel()
        
        with open(mean_filename + ".txt", "a") as file:
            x_real_line = '\t'.join([f"{x:8.4f}" for x in x_real.ravel()])
            mean_line = '\t'.join([f"{x:8.4f}" for x in mean.ravel()])
            line = f"\n{x_real_line}\t\t{mean_line}"
            file.write(line)

        if plot:
            # move the robot
            env.move_robot(x_real)
            
            # plot observation
            env.plot_observation(x_real, z_real, marker_id)
            
            # plot actual trajectory
            x_real_previous = states_real[i, :].reshape((-1, 1))
            env.plot_path_step(x_real_previous, x_real, [0,0,1])
            
            # plot noisefree trajectory
            noisefree_previous = states_noisefree[i]
            noisefree_current = states_noisefree[i+1]
            env.plot_path_step(noisefree_previous, noisefree_current, [0,1,0])
            
            # plot estimated trajectory
            if filt is not None:
                filter_previous = states_filter[i]
                filter_current = states_filter[i+1]
                env.plot_path_step(filter_previous, filter_current, [1,0,0])
            
            # plot particles
            if args.filter_type == 'pf':
                env.plot_particles(filt.particles, filt.weights)
        
        # pause/breakpoint
        if step_pause:
            time.sleep(step_pause)
        if step_breakpoint:
            breakpoint()
        
        errors[i, :] = (mean - x_real).ravel()
        errors[i, 2] = minimized_angle(errors[i, 2])
        position_errors[i] = np.linalg.norm(errors[i, :2])

        cond_number = np.linalg.cond(cov)
        if cond_number > 1e12:
            print('Badly conditioned cov (setting to identity):', cond_number)
            print(cov)
            cov = np.eye(3)
        mahalanobis_errors[i] = \
            errors[i:i+1, :].dot(np.linalg.inv(cov)).dot(errors[i:i+1, :].T)

    mean_position_error = position_errors.mean()
    mean_mahalanobis_error = mahalanobis_errors.mean()
    anees = mean_mahalanobis_error / 3

    if filt is not None:
        print('-' * 80)
        print('Mean position error:', mean_position_error)
        print('Mean Mahalanobis error:', mean_mahalanobis_error)
        print('ANEES:', anees)
    
    # if plot:
    #     while True:
    #         env.p.stepSimulation()
    
    return mean_position_error, anees


def setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'filter_type', choices=('none', 'ekf', 'pf'),
        help='filter to use for localization')
    parser.add_argument(
        '--plot', action='store_true',
        help='turn on plotting')
    parser.add_argument(
        '--seed', type=int,
        help='random seed')
    parser.add_argument(
        '--num-steps', type=int, default=200,
        help='timesteps to simulate')
    parser.add_argument(
        '--num-trials', type=int, default=10,
        help='number of trials for given filter-type and data/filter factors')

    # Noise scaling factors
    parser.add_argument(
        '--data-factor', type=float, default=1,
        help='scaling factor for motion and observation noise (data)')
    parser.add_argument(
        '--filter-factor', type=float, default=1,
        help='scaling factor for motion and observation noise (filter)')
    parser.add_argument(
        '--num-particles', type=int, default=100,
        help='number of particles (particle filter only)')
    
    # Learned Observation Model
    parser.add_argument(
        '--use-learned-observation-model', type=str, default=False,
        help='checkpoint for a learned observation model')
    parser.add_argument(
        '--supervision-mode', type=str, default='',
        help='phi|xy')
    parser.add_argument(
        '--device', type=str, default='cpu',
        help='device for the learned observation model')
    
    # Debugging arguments
    parser.add_argument(
        '--step-pause', type=float, default=0.,
        help='slows down the rollout to make it easier to visualize')
    parser.add_argument(
        '--step-breakpoint', action='store_true',
        help='adds a breakpoint to each step for debugging purposes')
    
    return parser


if __name__ == '__main__':
    args = setup_parser().parse_args()

    # if args.seed is not None:
    #     np.random.seed(args.seed)

    alphas = np.array([0.05**2, 0.005**2, 0.1**2, 0.01**2])
    beta = np.diag([np.deg2rad(5)**2])
    
    if args.use_learned_observation_model:
        assert args.supervision_mode in ('xy', 'phi')

    policy = policies.OpenLoopRectanglePolicy()

    filename =  f"{args.filter_type}_errors_df{int(args.data_factor)}_ff{int(args.filter_factor)}"
    if args.seed is None:
        filename += '_seedConst'
    else:
        filename += '_seedRand'

    for filter_factor_exp in range(-int(math.log(args.filter_factor, 4)), int(math.log(args.filter_factor, 4)) + 1): 
        filter_factor = 4**filter_factor_exp
        if args.data_factor == 1:
            data_factor = args.data_factor
        else:
            data_factor = filter_factor
        print('\nData factor: ', data_factor)
        print('Filter factor: ', filter_factor)

        mean_filename = f"{args.filter_type}_mean"

        if filter_factor_exp == -int(math.log(args.filter_factor, 4)):
            with open(filename + ".txt", "a") as file:
                file.write(f"Data factor: {data_factor}\nFilter factor: {filter_factor}")
        else:
            with open(filename + ".txt", "a") as file:
                file.write(f"\n\nData factor: {data_factor}\nFilter factor: {filter_factor}")

        for trial_n in range(args.num_trials):
            # imagename = f"screenshot_{args.filter_type}_df{data_factor}_ff{filter_factor}_#{trial_n}"
            # if args.seed is None:
            #     imagename += '_seedConst'
            # else:
            #     imagename += '_seedRand'

            initial_mean = np.array([180, 50, 0], dtype=float).reshape((-1, 1))
            initial_cov = np.diag(np.array([10, 10, 1], dtype=float))

            if args.seed is None: rand_seed = trial_n
            else: rand_seed = (filter_factor_exp + int(math.log(args.filter_factor, 4))) * args.num_trials + trial_n
            np.random.seed(rand_seed)

            if trial_n == 0:
                subtitle = f"Random seed: {rand_seed}"
            else:
                subtitle = f"\n\nRandom seed: {rand_seed}"
            with open(mean_filename + ".txt", "a") as file:
                x_real_line = '\t'.join([f"{x:8.4f}" for x in initial_mean.ravel()])
                mean_line = '\t'.join([f"{x:8.4f}" for x in initial_mean.ravel()])
                subtitle += f"\n{x_real_line}\t\t{mean_line}"
                file.write(subtitle)

            env = Field(
                data_factor * alphas,
                data_factor * beta,
                gui=args.plot,
                use_learned_observation_model=args.use_learned_observation_model,
                supervision_mode=args.supervision_mode,
                device=args.device
            )
            if args.filter_type == 'none':
                filt = None
            elif args.filter_type == 'ekf':
                filt = ExtendedKalmanFilter(
                    initial_mean,
                    initial_cov,
                    filter_factor * alphas,
                    filter_factor * beta
                )
            elif args.filter_type == 'pf':
                filt = ParticleFilter(
                    initial_mean,
                    initial_cov,
                    args.num_particles,
                    filter_factor * alphas,
                    filter_factor * beta
                )
                
            # You may want to edit this line to run multiple localization experiments.
            errors = "\n" + '\t\t'.join([f"{x:14.8f}" for x in list(localize(mean_filename, env, policy, filt, initial_mean, args.num_steps, args.plot, args.step_pause, args.step_breakpoint))])
            errors += f"\t\t{rand_seed:2d}"
            with open(filename + ".txt", "a") as file:
                file.write(errors)
                if (filter_factor_exp == int(math.log(args.filter_factor, 4))) and (trial_n == 9):
                    file.write("\n\n")

            if args.plot:
                # # Render the camera image
                # _,_,rgbaBuffer, depthBuffer, segmMaskBuffer = env.p.getCameraImage(
                #     *env.image_size, 
                #     env.view_matrix, 
                #     env.projection_matrix, 
                #     renderer=env.p.ER_BULLET_HARDWARE_OPENGL)
                # # Convert the rgbaBuffer to a NumPy array and reshape it to the proper (width, height, channels)
                # # rgba_np = np.array(rgbaBuffer, dtype=np.uint8).reshape(*env.image_size, 4)
                # # Remove alpha channel as it saves memory, + JPEG format doesn't support alpha channel
                # rgb_np = rgbaBuffer[:, :, :3]
                # img = Image.fromarray(rgb_np, 'RGB')
                # img.save(imagename + '.png')
                # print("\nPress Enter to disconnect from the simulation...")
                # input()
                print("\nTake screenshot, and press Enter to restart the simulation with different data/filter factors...")
                input()
                env.p.disconnect()

        # Open txt file with list of num-steps * args.num_trials (num_trials) estimated and real mean (x,y,theta) for given filter-type, data/filter factors
        rand_seed_list = []
        x_real_list = []
        mean_list = []
        x_axis = [x for x in range(1, args.num_steps + 2)]
        with open(mean_filename + ".txt", "r") as file:
            for line in file:
                # Skip "\n" lines
                if len(line.rstrip()) != 0:
                    if "Random" in line:
                        rand_seed_list.append(int(line.rstrip().replace("Random seed: ", "")))
                    else:
                        means_list = [float(x) for x in line.split()]
                        x_real_list.append(list(means_list[:3]))
                        mean_list.append(list(means_list[3:]))
        
        # Clear the content of mean_filename.txt for further reuse
        with open(mean_filename + ".txt", "w") as file:
            pass

        # Plot estimated and real mean (x,y,theta) for given filter-type, data/filter factors and args.num_trials trials into 3 line graphs 
        fig, (x_plot, y_plot, theta_plot) = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(20, 10))
        fig.supxlabel('number of steps')
        fig.suptitle(f"Estimated and real mean plots for data factor: {data_factor}, filter factors: {filter_factor}")
        x_plot.set_title(label="x", loc='left')
        y_plot.set_title(label="y", loc='left')
        theta_plot.set_title(label="theta", loc='left')
        color_list = np.random.rand(2 * args.num_trials, 3)
        for i in range(args.num_trials):
            x_plot.plot(x_axis, [row[0] for row in x_real_list[args.num_steps * i : args.num_steps * (i + 1) + 1]], color=color_list[2*i],
                    label=f"Real mean, seed: {rand_seed_list[i]}", linestyle='--', linewidth=1, marker='o', markersize=1)
            y_plot.plot(x_axis, [row[1] for row in x_real_list[args.num_steps * i : args.num_steps * (i + 1) + 1]], 
                    color=color_list[2*i], linestyle='--', linewidth=1, marker='o', markersize=1)
            theta_plot.plot(x_axis, [row[2] for row in x_real_list[args.num_steps * i : args.num_steps * (i + 1) + 1]], 
                    color=color_list[2*i], linestyle='--', linewidth=1, marker='o', markersize=1)

            line, = x_plot.plot(x_axis, [row[0] for row in mean_list[args.num_steps * i : args.num_steps * (i + 1) + 1]], color=color_list[2*i+1],
                    label=f"Est. mean, seed: {rand_seed_list[i]}", linestyle='-', linewidth=1, marker='s', markersize=1)
            y_plot.plot(x_axis, [row[1] for row in mean_list[args.num_steps * i : args.num_steps * (i + 1) + 1]], 
                    color=color_list[2*i+1], linestyle='-', linewidth=1, marker='s', markersize=1)
            theta_plot.plot(x_axis, [row[2] for row in mean_list[args.num_steps * i : args.num_steps * (i + 1) + 1]], 
                    color=color_list[2*i+1], linestyle='-', linewidth=1, marker='s', markersize=1)

        fig.legend()
        if args.seed is None:
            plt.savefig(mean_filename + f"_df{data_factor}_ff{filter_factor}_seedConst.png", dpi=300, bbox_inches='tight')
        else:
            plt.savefig(mean_filename + f"_df{data_factor}_ff{filter_factor}_seedRand.png", dpi=300, bbox_inches='tight')

    # Open txt file with position error, ANEES and random.seed (for each trial and data/filter factors) for a given filter-type
    factor_dict = {'data': [], 'filter': []}
    mean_position_error_list = []
    # mean_mahalanobis_error_list = []
    anees_list = []
    rand_seed_list = []
    x_axis = [x for x in range(1, 11)]
    with open(filename + ".txt", "r") as file:
        for line in file:
            # Skip "\n" lines
            if len(line.rstrip()) != 0:
                if "Data" in line:
                    factor_dict['data'].append(float(line.rstrip().replace("Data factor: ", "")))
                elif "Filter" in line:
                    factor_dict['filter'].append(float(line.rstrip().replace("Filter factor: ", "")))
                else:
                    error_list = [float(x) for x in line.split()]
                    mean_position_error_list.append(error_list[0])
                    # mean_mahalanobis_error_list.append(error_list[1])
                    anees_list.append(error_list[1])
                    rand_seed_list.append(int(error_list[2]))

    # Plot the errors into one graph
    fig, error_plot = plt.subplots(figsize=(16, 8))
    fig.supxlabel('trial number (different random.seed for each trial and factor)')
    fig.supylabel('errors')
    fig.suptitle('Position error and ANEES plots for different data/filter factors (df/ff)')
    factor_n = len(factor_dict['filter'])
    for i in range(factor_n):
        error_plot.plot(x_axis, mean_position_error_list[args.num_trials * i : args.num_trials * (i + 1)], 
                 label=f"Position df: {factor_dict['data'][i]}; ff: {factor_dict['filter'][i]}; seed: {rand_seed_list[args.num_trials*i]}-{rand_seed_list[args.num_trials*(i+1)-1]}", 
                 linestyle='-', linewidth=2, marker='s', markersize=5)
        # error_plot.plot(x_axis, mean_mahalanobis_error_list[args.num_trials * i : args.num_trials * (i + 1)], 
        #          label=f"Mahalanobis df: {factor_dict['data'][i]}; ff: {factor_dict['filter'][i]}; seed: {rand_seed_list[args.num_trials*i]}-{rand_seed_list[args.num_trials*(i+1)-1]}", 
        #          linestyle='--', linewidth=2, marker=',', markersize=5)
        error_plot.plot(x_axis, anees_list[args.num_trials * i : args.num_trials * (i + 1)], 
                 label=f"ANNES df: {factor_dict['data'][i]}; ff: {factor_dict['filter'][i]}; seed: {rand_seed_list[args.num_trials*i]}-{rand_seed_list[args.num_trials*(i+1)-1]}",
                 linestyle='-.', linewidth=2, marker='o', markersize=5)
    fig.legend()
    plt.savefig(filename + ".png", dpi=300, bbox_inches='tight')

    print("\nPress Enter to disconnect from the simulation...")
    input()
