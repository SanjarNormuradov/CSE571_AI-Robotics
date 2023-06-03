import torch
import numpy as np
import torch.optim as optim
from utils import log_density, rollout

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_model(policy, baseline, trajs, policy_optim, baseline_optim, gamma=0.99, baseline_train_batch_size=64, baseline_num_epochs=5, seed=0):
    np.random.seed(seed)
    # Compute the returns on the current batch of trajectories
    # Go through all the trajectories in trajs and compute their return to go: discounted sum of rewards from that timestep to the end. 
    # This is easy to do if you go backwards in time and sum up the reward as a running sum. 
    # Remember that return to go is return = r[t] + gamma*r[t+1] + gamma^2*r[t+2] + ...
    states_all = []
    actions_all = []
    returns_all = []
    for traj in trajs:
        states_singletraj = traj['observations']
        actions_singletraj = traj['actions']
        rewards_singletraj = traj['rewards']
        returns_singletraj = np.zeros_like(rewards_singletraj)
        # Discounted return-to-go
        traj_len = len(rewards_singletraj)
        for i in reversed(range(traj_len)):
            returns_singletraj[i] = rewards_singletraj[i] + (returns_singletraj[i+1] * gamma if i+1 < traj_len else 0)
        states_all.append(states_singletraj)
        actions_all.append(actions_singletraj)
        returns_all.append(returns_singletraj)
    states = np.concatenate(states_all)
    actions = np.concatenate(actions_all)
    returns = np.concatenate(returns_all)
    # print(f"states.shape:\n{states.shape}")
    # print(f"actions.shape:\n{actions.shape}")
    # print(f"returns.shape:\n{returns.shape}")
    # input()
    
    EPS = 1e-9 # Prevent zero-division error in case returns.std() == 0
    # Normalize the returns: zero-center and normalize spread of returns to 1 (normal/Gaussian distribution)
    returns = (returns - returns.mean()) / (returns.std() + EPS)
    
    # Train baseline by regressing onto returns
    # Regress the baseline from each state onto the above computed return to go. You can use similar code to behavior cloning to do so. 
    criterion = torch.nn.MSELoss()
    n = len(states)
    arr = np.arange(n)
    baseline_num_batches = n // baseline_train_batch_size
    states = torch.tensor(states).float().to(device)
    actions = torch.tensor(actions).to(device)
    returns = torch.tensor(returns).float().to(device)
    for epoch in range(baseline_num_epochs):
        np.random.shuffle(arr)
        for i in range(baseline_num_batches):
            batch_index = arr[baseline_train_batch_size * i: baseline_train_batch_size * (i + 1)]
            batch_index = torch.LongTensor(batch_index).to(device)
            input_states = states[batch_index]
            true_returns = returns[batch_index]
            pred_returns = baseline(input_states)

            loss_baseline = criterion(pred_returns, true_returns)
            # Clear gradients of the policy.parameters (weights, bias)
            baseline_optim.zero_grad()
            loss_baseline.backward()
            baseline_optim.step()
            
    # Train policy by optimizing surrogate objective: -log_policy * (return - baseline)
    # Policy gradient = \grad log_policy(a|s) * (return - baseline)
    # Return is computed above, you can compute log_policy using the log_density function imported. 
    # Use standard pytorch machinery to take *one* gradient step on the policy
    mu, std, logstd = policy(states)
    log_policy = log_density(actions, mu, std, logstd)
    # print(f"log_policy.shape:\n{log_policy.shape}")
    baseline_pred = baseline(states)
    # print(f"baseline_pred.shape:\n{baseline_pred.shape}")

    loss_policy = (-log_policy * (returns - baseline_pred)).mean()
    # print(f"loss_policy.shape:\n{loss_policy.shape}")
    # input()

    policy_optim.zero_grad()
    loss_policy.backward()
    policy_optim.step()

    del states, actions, returns, states_all, actions_all, returns_all

    return loss_policy.item()

# Training loop for policy gradient
def simulate_policy_pg(env, policy, baseline, num_epochs=20000, max_path_length=200, pg_batch_size=100, gamma=0.99, 
                       baseline_train_batch_size=64, baseline_num_epochs=5, print_freq=10, render=False, seed=0):
    policy_optim = optim.Adam(policy.parameters())
    baseline_optim = optim.Adam(baseline.parameters())

    losses = []
    for iter_num in range(num_epochs):
        sample_trajs = []

        # Sampling trajectories
        for it in range(pg_batch_size):
            sample_traj = rollout(env=env, agent=policy, episode_length=max_path_length, agent_name='pg', render=render)
            sample_trajs.append(sample_traj)
        
        # Logging returns occasionally
        if iter_num % print_freq == 0:
            rewards_np = np.mean(np.asarray([traj['rewards'].sum() for traj in sample_trajs]))
            path_length = np.max(np.asarray([traj['rewards'].shape[0] for traj in sample_trajs]))
            print(f"Episode: {iter_num:3d};\treward: {rewards_np:6.2f};\t max_path_length: {path_length:3d}")

        # Training model
        policy_loss = train_model(policy, baseline, sample_trajs, policy_optim, baseline_optim, gamma=gamma, 
                                  baseline_train_batch_size=baseline_train_batch_size, baseline_num_epochs=baseline_num_epochs, seed=seed)
        losses.append(policy_loss)
        
    return losses
