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

    policy = DDPGPolicy(actor_net, actor_optim, critic_net, critic_optim, gamma=gamma, estimation_step=n_step)

    buf = ReplayBuffer(size=12)


    # train_collector = ts.data.Collector(policy, train_envs, ts.data.VectorReplayBuffer(buffer_size, train_num), exploration_noise=True)
    train_collector = ts.data.Collector(policy, train_envs, exploration_noise=True)  # because DQN uses epsilon-greedy method
    test_collector = ts.data.Collector(policy, test_envs, exploration_noise=True)  # because DQN uses epsilon-greedy method

    # Ref: https://colab.research.google.com/drive/1MhzYXtUEfnRrlAVSB3SR83r0HA5wds2i?usp=sharing#scrollTo=a_mtvbmBZbfs
    print('Start training...')
    reward_store = []
    obs = env.reset()

    for i in range(epoch):
        print('obs: ', obs)
        act = policy(Batch(obs=obs)).act.item()
        obs_next, rew, done, info = env.step(act)
        # pretend ending at step 3
        done = True if i==2 else False
        info["id"] = i
        buf.add(Batch(obs=obs, act=act, rew=rew, done=done, obs_next=obs_next, info=info))
        obs_next = obs
        policy.update(0, buf, batch_size=batch_size, repeat=6) 


    

    
    

if __name__ == '__main__':
    main(verbose=True)