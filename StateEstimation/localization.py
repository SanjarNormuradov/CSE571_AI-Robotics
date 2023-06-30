import time
import os
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
        print('-' * 40)
        print(f"Mean Position Error:    {mean_position_error:12.4f}")
        print(f"Mean Mahalanobis Error: {mean_mahalanobis_error:12.4f}")
        print(f"ANEES:                  {anees:12.4f}")
    
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
    
    learned_obs_model = ""
    if args.use_learned_observation_model:
        assert args.supervision_mode in ('xy', 'phi')
        learned_obs_model = f"_cnn_{args.supervision_mode}"  

    policy = policies.OpenLoopRectanglePolicy()

    output_directory = f"plots_{args.filter_type}"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # File path to save statistics (mean, std) of position (optionally Mahalanobis) error and ANEES (Avarage Normalized Estimation Error Squared)
    # for different data/filter factors depending on filter-type, #trials and initial numpy.random.seed 
    stat_filename = f"{output_directory}/statistics"
    # if "--seed i" then for each different data/filter factor simulations 
    #   numpy.random.seed would be set with the same value for each trial ("--num-trial n", default=10) starting from i with 1 step increment,
    #   i.e. data-factor=1, filter-factor=1: trial#1 (np.random.seed=i), trial#2 (np.random.seed=i+1), ... trial#n (np.random.seed=i+n-1)
    #        data-factor=1, filter-factor=4: trial#1 (np.random.seed=i), trial#2 (np.random.seed=i+1), ... trial#n (np.random.seed=i+n-1)
    # else ("--seed i" not specified), then 
    #        data-factor=1, filter-factor=1: trial#1 (np.random.seed=0), trial#2 (np.random.seed=1), ... trial#n (np.random.seed=n-1)
    #        data-factor=1, filter-factor=4: trial#1 (np.random.seed=n), trial#2 (np.random.seed=n+1), ... trial#n (np.random.seed=2n-1)
    stat_filename += f"_#trail{args.num_trials}"
    stat_filename += f"_#particle_{args.num_particles}" if args.filter_type == 'pf' else ''
    stat_filename += (('_initSeed0' if args.seed is None else f"_initSeed{args.seed}") + \
                      ('_diffFactor' if args.data_factor == 1 else '_sameFactor')) if args.filter_type == 'ekf' else ''
    stat_filename += f"{learned_obs_model}"

    if args.seed is not None:
        assert args.seed >= 0, f"random.seed couldn't be negative, {args.seed}"
    rand_start = 0 if args.seed is None else args.seed

    with open(stat_filename + ".txt", "w") as file:
        file.write(f"[data_factor, filter_factor]: [position_error_mean, position_error_std]; [anees_mean, anees_std]\n")
        for filter_factor_exp in range(-int(math.log(args.filter_factor, 4)), int(math.log(args.filter_factor, 4)) + 1): 
            filter_factor = 4**filter_factor_exp
            data_factor = args.data_factor if args.data_factor == 1 else filter_factor
            print(f"\nData Factor: {data_factor:9.6f}\nFilter Factor: {filter_factor:9.6f}")
            pos_errors = []
            anees = []
            for rand_seed in range(rand_start, args.num_trials + rand_start):
                np.random.seed(rand_seed)
                # File path to save screenshots of the car's path. 
                # FAIL: physicsClient.getCameraImage() doesn't see physicsClient.addUserDebugLine() (car's path)
                # imagename = f"screenshot_{args.filter_type}_df{data_factor}_ff{filter_factor}_#{trial_n}"
                # imagename += '_seedRand' if args.seed is None else '_seedConst'
                
                # Initial position (x,y,theta)
                initial_mean = np.array([180, 50, 0], dtype=float).reshape((-1, 1))
                # Initial covariance matrix
                initial_cov = np.diag(np.array([10, 10, 1], dtype=float))

                # Create simulation scene and add the robot
                if rand_seed == rand_start:
                    env = Field(
                        data_factor * alphas,
                        data_factor * beta,
                        gui=args.plot,
                        use_learned_observation_model=args.use_learned_observation_model,
                        supervision_mode=args.supervision_mode,
                        device=args.device
                    )
                else:
                    env = Field(
                        data_factor * alphas,
                        data_factor * beta,
                        gui=False,
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
                
                pos_error, anee = localize(env, policy, filt, initial_mean, args.num_steps, args.plot, args.step_pause, args.step_breakpoint)
                pos_errors.append(pos_error)
                anees.append(anee)

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
                    if rand_seed == rand_start:
                        print("\nTake screenshot, and press Enter to restart the simulation with different data/filter factors...")
                        input()
                    env.p.disconnect()

            pos_errors = np.array(pos_errors)
            anees = np.array(anees)
            file.write(f"[{data_factor:9.6f}, {filter_factor:9.6f}]: [{pos_errors.mean():11.6f}, {pos_errors.std():11.6f}]; [{anees.mean():11.6f}, {anees.std():11.6f}]\n")

    factor_list = []
    mean_dict = {'position':[], 'anees':[]}
    std_dict = {'position':[], 'anees':[]}
    with open(stat_filename + ".txt", "r") as file:
        for line in file:
            # Skip "\n" lines
            if len(line.rstrip()) != 0:
                if 'data_factor' in line:
                    continue
                else:
                    factor_list.append(float(line[line.find(',') + 1 : line.find(',') + 10]))
                    mean_dict['position'].append(float(line[line.find(':') + 3 : line.find(':') + 14]))
                    std_dict['position'].append(float(line[line.find(':') + 16 : line.find(':') + 27]))
                    mean_dict['anees'].append(float(line[line.find(';') + 3 : line.find(';') + 14]))
                    std_dict['anees'].append(float(line[line.find(';') + 16 : line.find(';') + 27]))

    # Plot mean and std for given filter-type and #trials into 2 line graphs 
    fig, (mean_plot, std_plot) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(20, 10))
    factor_label = 'different filter-factors, const data-factor=1' if args.data_factor==1 else 'same data/filter-factors'
    fig.supxlabel(factor_label)
    fig.suptitle(f"Statistics (mean, std) of Position Error (blue) and ANEES (red) over {factor_label}")
    mean_plot.set_title(label="mean", loc='left')
    std_plot.set_title(label="std", loc='left')

    mean_plot.plot(factor_list, mean_dict['position'], label='Position Error mean', 
                   color='blue', linestyle='--', linewidth=1, marker='o', markersize=3)
    mean_plot.plot(factor_list, mean_dict['anees'], label='ANEES mean', 
                   color='red', linestyle='-', linewidth=1, marker='s', markersize=3)
    std_plot.plot(factor_list, std_dict['position'], label='Position Error std', 
                   color='blue', linestyle='--', linewidth=1, marker='o', markersize=3)
    std_plot.plot(factor_list, std_dict['anees'], label='ANEES std', 
                   color='red', linestyle='-', linewidth=1, marker='s', markersize=3)
    fig.legend()
    plt.savefig(stat_filename + ".png", dpi=300, bbox_inches='tight')

    print("\nPress Enter to disconnect from the simulation...")
    input()
