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


def localize(
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
    
    if plot:
        print('-' * 80)
        print("Take screenshot of the current simulation,\nand press Enter to restart the simulation with different data/filter factor...")
        input()
        env.p.disconnect()
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

    if args.seed is not None:
        np.random.seed(args.seed)

    alphas = np.array([0.05**2, 0.005**2, 0.1**2, 0.01**2])
    beta = np.diag([np.deg2rad(5)**2])
    
    if args.use_learned_observation_model:
        assert args.supervision_mode in ('xy', 'phi')

    policy = policies.OpenLoopRectanglePolicy()
    filename = args.filter_type + "_errors_df" + str(int(args.data_factor)) + "_ff" + str(int(args.filter_factor))

    for filter_factor_exp in range(-int(math.log(args.filter_factor, 4)), int(math.log(args.filter_factor, 4)) + 1): 
        filter_factor = 4**filter_factor_exp
        if args.data_factor == 1:
            data_factor = args.data_factor
        else:
            data_factor = filter_factor
        print('\nData factor: ', data_factor)
        print('Filter factor: ', filter_factor)

        if filter_factor_exp == -int(math.log(args.filter_factor, 4)):
            with open(filename + ".txt", "a") as file:
                file.write("Data factor: " + str(data_factor) + "\nFilter factor: " + str(filter_factor))
        else:
            with open(filename + ".txt", "a") as file:
                file.write("\n\nData factor: " + str(data_factor) + "\nFilter factor: " + str(filter_factor))            

        for trial_n in range(10):
            initial_mean = np.array([180, 50, 0], dtype=float).reshape((-1, 1))
            initial_cov = np.diag(np.array([10, 10, 1], dtype=float))

            if args.seed is None:
                np.random.seed((filter_factor_exp + int(math.log(args.filter_factor, 4)) + 1) * 100 + trial_n)

            env = Field(
                data_factor * alphas,
                data_factor * beta,
                gui=args.plot,
                use_learned_observation_model=args.use_learned_observation_model,
                supervision_mode=args.supervision_mode,
                device=args.device,
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
            errors = "\n" + '\t'.join([f"{x:13.8f}" for x in list(localize(env, policy, filt, initial_mean, args.num_steps, args.plot, args.step_pause, args.step_breakpoint))])
            with open(filename + ".txt", "a") as file:
                file.write(errors)
                if (filter_factor_exp == int(math.log(args.filter_factor, 4))) and (trial_n == 9):
                    file.write("\n\n")
    
    factor_dict = {'data': [], 'filter': []}
    mean_position_error_list = []
    # mean_mahalanobis_error_list = []
    anees_list = []
    x_axis = [x for x in range(1, 11)]
    with open(filename + ".txt", "r") as file:
        for line in file:
            if len(line.rstrip()) != 0:
                if "Data" in line:
                    factor_dict['data'].append(float(line.rstrip().replace("Data factor: ", "")))
                elif "Filter" in line:
                    factor_dict['filter'].append(float(line.rstrip().replace("Filter factor: ", "")))
                else:
                    error_list = [float(x) for x in line.split()]
                    mean_position_error_list.append(error_list[0])
                    anees_list.append(error_list[1])
                    # mean_mahalanobis_error_list.append(error_list[1])

    fig, error_plot = plt.subplots(figsize=(16, 8))
    fig.supxlabel('trial number (different random.seed for each trial and factor)')
    fig.supylabel('errors')
    fig.suptitle('Position error and ANEES plots for different data/filter factors (df/ff)')
    factor_n = len(factor_dict['filter'])
    for i in range(factor_n):
        error_plot.plot(x_axis, mean_position_error_list[10 * i : 10 * (i + 1)], 
                 label="Position df: " + str(factor_dict['data'][i]) + "; ff: " + str(factor_dict['filter'][i]), 
                 linestyle='-', linewidth=2, marker='s', markersize=5)
        # error_plot.plot(x_axis, mean_mahalanobis_error_list[10 * i : 10 * (i + 1)], 
        #          label="Mahalanobis df: " + str(factor_dict['data'][i]) + "; ff: " + str(factor_dict['filter'][i]), 
        #          linestyle='--', linewidth=2, marker=',', markersize=5)
        error_plot.plot(x_axis, anees_list[10 * i : 10 * (i + 1)], 
                 label="ANNES df: " + str(factor_dict['data'][i]) + "; ff: " + str(factor_dict['filter'][i]), 
                 linestyle='-.', linewidth=2, marker='o', markersize=5)
    fig.legend()
    plt.savefig(filename + ".png", dpi=300, bbox_inches='tight')

    print("\nPress Enter to disconnect from the simulation...")
    input()
