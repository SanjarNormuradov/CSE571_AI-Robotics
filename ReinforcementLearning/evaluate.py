import torch
import numpy as np

from utils import rollout

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def evaluate(env, policy, agent_name, num_validation_runs=10, episode_length=50, render=False):
    success_count = 0
    rewards_suc = 0
    rewards_all = 0
    for k in range(num_validation_runs):
        o = env.reset()
        path = rollout(
                env,
                policy,
                agent_name=agent_name,
                episode_length=episode_length,
                render=render)
        if agent_name == 'pg':
            success = len(path['dones']) == episode_length
        elif agent_name == 'bc' or agent_name == 'dagger':
            success = np.linalg.norm(env.get_body_com("fingertip") - env.get_body_com("target"))<0.1
        if success:
            success_count += 1
            rewards_suc += np.sum(path['rewards'])
        rewards_all += np.sum(path['rewards'])
        # print(f"test {k:5d}, success {str(success):<5}, reward {np.sum(path['rewards']):8.4f}")
    result_txt = [f"{success_count/num_validation_runs}",
                  f"{(rewards_suc/max(success_count, 1)):10.6f}",
                  f"{(rewards_all/num_validation_runs):10.6f}"]
    print(f"Success rate: {result_txt[0]}")
    print(f"Average reward (success only): {result_txt[1]}")
    print(f"Average reward          (all): {result_txt[2]}")
    return result_txt