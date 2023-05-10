import numpy as np
import matplotlib as plt
import os, time

import tianshou as ts
from tianshou.policy import DDPGPolicy
from tianshou.data import Batch, ReplayBuffer

import torch, numpy as np, torch.nn as nn
from tianshou.utils.net.common import Net
from torch.utils.tensorboard import SummaryWriter
import gymnasium as gym
from SRCEnv import SRCEnv


def main(verbose=False):
    task = 'SRCEnv'
    log_dir = './logs/results'

    # Hyperparameters
    lr, epoch, batch_size = 1e-6, 5, 5
    train_num, test_num = 1, 1
    gamma, n_step, target_freq = 0.9, 3, 320
    buffer_size = 20000
    eps_train, eps_test = 0.1, 0.05
    step_per_epoch, step_per_collect = 2, 10
    logger = ts.utils.TensorboardLogger(SummaryWriter(log_dir)) 

    gym.envs.register(id=task, entry_point=SRCEnv)
    train_envs = ts.env.DummyVectorEnv([lambda: gym.make(task) for _ in range(train_num)])
    test_envs = ts.env.DummyVectorEnv([lambda: gym.make(task) for _ in range(test_num)])

    env = gym.make(task)
    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n

    actor_net = Net(state_shape=state_shape, action_shape=action_shape, hidden_sizes=[128, 128, 128])
    actor_optim = torch.optim.Adam(actor_net.parameters(), lr=lr)

    critic_net = Net(state_shape=state_shape, action_shape=action_shape, hidden_sizes=[128, 128, 128])
    critic_optim = torch.optim.Adam(critic_net.parameters(), lr=lr)

    

    # policy = ts.policy.DDPGPolicy(net, optim, gamma, n_step, target_update_freq=target_freq)
    policy = DDPGPolicy(actor_net, actor_optim, critic_net, critic_optim, gamma=gamma, estimation_step=n_step)

    # train_collector = ts.data.Collector(policy, train_envs, ts.data.VectorReplayBuffer(buffer_size, train_num), exploration_noise=True)
    train_collector = ts.data.Collector(policy, train_envs, exploration_noise=True)  # because DQN uses epsilon-greedy method
    test_collector = ts.data.Collector(policy, test_envs, exploration_noise=True)  # because DQN uses epsilon-greedy method

    # Ref: https://colab.research.google.com/drive/1MhzYXtUEfnRrlAVSB3SR83r0HA5wds2i?usp=sharing#scrollTo=a_mtvbmBZbfs
    print('Start training...')
    result = ts.trainer.offpolicy_trainer(
        policy, 
        train_collector, 
        test_collector, 
        epoch, 
        step_per_epoch, 
        step_per_collect,
        test_num, 
        batch_size, 
        update_per_step=1 / step_per_collect,
        # train_fn=lambda epoch, env_step: policy.set_eps(eps_train),
        # test_fn=lambda epoch, env_step: policy.set_eps(eps_test),
        stop_fn=lambda mean_rewards: mean_rewards >= env.spec.reward_threshold,
        logger=logger)
    print(f'Finished training! Use {result["duration"]}')
    torch.save(policy.state_dict(), 'dqn.pth')

if __name__ == '__main__':
    main(verbose=True)
