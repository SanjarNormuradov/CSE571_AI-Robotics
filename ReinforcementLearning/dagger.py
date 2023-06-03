import torch
import torch.optim as optim
import numpy as np

from utils import rollout, relabel_action

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def simulate_policy_dagger(env, policy, expert_paths, expert_policy=None, num_epochs=500, episode_length=50,
                            batch_size=32, num_dagger_iters=10, num_trajs_per_dagger=10,  data_part=1.0, seed=0):
    np.random.seed(seed)
    # Hint: Loop through num_dagger_iters iterations, at each iteration train a policy on the current dataset.
    # Then rollout the policy, use relabel_action to relabel the actions along the trajectory with "expert_policy" and then add this to current dataset
    # Repeat this so the dataset grows with states drawn from the policy, and relabeled actions using the expert.
    optimizer = optim.Adam(list(policy.parameters()))
    criterion = torch.nn.MSELoss()
    returns = []
    loss_list = []

    trajs = expert_paths
    # Dagger iterations
    for dagger_itr in range(num_dagger_iters):
        idxs = np.array(range(len(trajs)))
        num_batches = int(len(idxs)*episode_length*data_part) // batch_size
        observations = np.array([_dict['observations'] for _dict in trajs]).reshape(-1, 11)
        actions = np.array([_dict['actions'] for _dict in trajs]).reshape(-1, 2)
        data = np.concatenate((observations, actions), axis=1) # new shape = (idxs*50, 13)
        losses = []
        for epoch in range(num_epochs):
            np.random.shuffle(data) # shuffle rows (observation and corresponding action remain in the same row)
            crop_data = data[:int(data.shape[0]*data_part),:]
            running_loss = 0.0
            for i in range(num_batches):
                # Clear gradients of the policy.parameters (weights, bias)
                optimizer.zero_grad()

                if i < (num_batches - 1):
                    obs = crop_data[i*batch_size:(i+1)*batch_size, :11]
                    act = crop_data[i*batch_size:(i+1)*batch_size, 11:]
                else:
                    obs = crop_data[i*batch_size:, :11]
                    act = crop_data[i*batch_size:, 11:]

                obs = torch.tensor(obs).float().to(device)
                act = torch.tensor(act).float().to(device)
                act_pred = policy(obs)
                loss = criterion(act_pred, act)

                # Compute gradients of the loss with respect to the parameters (weights, bias)
                loss.backward()
                # Update the parameters according to the gradients
                optimizer.step()
                running_loss += loss.item()
            # print(f"[{epoch+1}, {(i+1):5d}] loss: {running_loss / 10.:.8f}")
            losses.append(loss.item())
        loss_list.append(losses)
        
        # Collecting more data for dagger
        trajs_recent = []
        for k in range(num_trajs_per_dagger):
            env.reset()
            # Rollout new trajectories using trained policy
            path = rollout(env,
                           policy,
                           agent_name='bc',
                           episode_length=episode_length,
                           render=False)
            # Relabel actions using expert policy
            path = relabel_action(path, expert_policy)
            trajs_recent.append(path)
        # Aggregate dataset
        trajs += trajs_recent
        mean_return = np.mean(np.array([traj['rewards'].sum() for traj in trajs_recent]))
        print(f"[{(dagger_itr+1):2d}] Average DAgger return = {mean_return:10.6f}")
        returns.append(mean_return)

    return loss_list