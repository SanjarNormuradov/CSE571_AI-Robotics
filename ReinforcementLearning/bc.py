import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def simulate_policy_bc(env, policy, expert_data, num_epochs=500, episode_length=50, batch_size=32, data_part=1.0, seed=0):
    np.random.seed(seed)
    optimizer = optim.Adam(list(policy.parameters()))
    # Mean-Square Error Loss function
    criterion = torch.nn.MSELoss()
    idxs = np.array(range(len(expert_data)))
    num_batches = int(len(idxs)*episode_length*data_part) // batch_size
    observations = np.array([_dict['observations'] for _dict in expert_data]).reshape(-1, 11)
    actions = np.array([_dict['actions'] for _dict in expert_data]).reshape(-1, 2)
    data = np.concatenate((observations, actions), axis=1) # new shape = (250, 13)
    # expert_data = [dict1, ...., dict5], 
    # dict = {'observations': np.array(50, 11), 'next_observations': np.array(50, 11), 
    #         'actions': np.array(50, 2), 'rewards': np.array(50, 1), 'dones': np.array(50, 1), 'images': np.array(0,)}
    losses = []
    for epoch in range(num_epochs): 
        np.random.shuffle(data) # shuffle rows (observation and corresponding action remain in the same row)
        crop_data = data[:int(data.shape[0]*data_part),:]
        # print(f"crop_data:\t{crop_data}")
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
            act_pred = policy.forward(obs)
            loss = criterion(act_pred, act)

            # Compute gradients of the loss with respect to the parameters (weights, bias)
            loss.backward()
            # Update the parameters according to the gradients
            optimizer.step()
            running_loss += loss.item()

        if epoch % 50 == 0:
            print('[%4d] loss: %.8f' %
                (epoch, running_loss / 10.))
        losses.append(loss.item())

    return losses
