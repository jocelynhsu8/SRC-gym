import numpy as np
<<<<<<< HEAD
import matplotlib as plt
import os, time

import tianshou as ts
from tianshou.policy import DDPGPolicy
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

    # Ref: https://colab.research.google.com/drive/1qMsEiZZ8mh60ycbfoX-nYy6qMCnLkmZE?usp=sharing#scrollTo=do-xZ-8B7nVH
    print('Start training...')
    reward_store = []

    for i in range(epoch):
        result = train_collector.collect(n_step=step_per_epoch)
        reward_store.append(result['rew'])
    
        evaluation_result = test_collector.collect(n_episode=10)

        if verbose:
            print(f'Epoch #{i}: Train Reward {result["rew"]} | Eval Reward {evaluation_result["rew"]}')
        
        logger.add_scalar('train_reward', result['rew'], i)
        logger.add_scalar('eval_reward', evaluation_result['rew'], i)
        policy.update(0, train_collector.buffer, batch_size=batch_size, repeat=1)
        train_collector.reset_buffer(keep_statistics=True)
        np.save(log_dir+'/reward_store.npy', reward_store)

    reward_store = np.array(reward_store)
    plt.plot(reward_store, label='reward', xlabel='Epoch', ylabel='Reward')

    print(f'Finished training!')
    torch.save(policy.state_dict(), 'dqn.pth')

    # result = ts.trainer.offpolicy_trainer(
    #     policy, 
    #     train_collector,
    #     test_collector, 
    #     epoch, 
    #     step_per_epoch, 
    #     step_per_collect,
    #     test_num, 
    #     batch_size, 
    #     update_per_step=1 / step_per_collect,
    #     # train_fn=lambda epoch, env_step: policy.set_eps(eps_train),
    #     # test_fn=lambda epoch, env_step: policy.set_eps(eps_test),
    #     stop_fn=lambda mean_rewards: mean_rewards >= env.spec.reward_threshold,
    #     logger=logger)
    
    

if __name__ == '__main__':
    main(verbose=True)
=======
import os, time

from stable_baselines3 import DDPG
from stable_baselines3.ddpg.policies import MlpPolicy
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from SRCEnv import SRCEnv


def train(training_env: SRCEnv, log_dir='./logs/results'):
    """ Train a DDPG agent on the SRCEnv environment

    Params:
    -------
    training_env: SRCEnv
        The environment used for training
    log_dir: str
        The directory to save the results
    """

    # Prepare log directory and hyperparameters
    os.makedirs(log_dir, exist_ok=True)
    n_actions = training_env.action_space.shape[0]
    noise_std = 0.2
    action_noise = OrnsteinUhlenbeckActionNoise(
        mean=np.zeros(n_actions),
        sigma=noise_std * np.ones(n_actions)
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=100000,
        save_path="./ddpg_dvrk_tensorboard/"
    )
    callback = CallbackList([checkpoint_callback])  # , eval_callback])

    # Train the agent
    model = DDPG(
      MlpPolicy,
      training_env,
      gamma=0.95,
      verbose=1,
      nb_train_steps=300,
      nb_rollout_steps=150,
      param_noise=param_noise,
      batch_size=128,
      action_noise=action_noise,
      random_exploration=0.05,
      normalize_observations=True,
      tensorboard_log="./ddpg_dvrk_tensorboard/",
      observation_range=(-1.5,
                       1.5),
      critic_l2_reg=0.01
    )
    training_env.reset()
    model.learn(total_timesteps=400000, log_interval=100, callback=callback)
    model.save("ddpg")


def eval(eval_env):
    """ Load a trained model and evaluate it
    
    Params:
    -------
    eval_env: SRCEnv
        The environment used for evaluation, must be wrapped with SRCEnv
    """
    model = DDPG.load('./ddpg', env=eval_env)
    count = 0
    step_num_arr = []
    for _ in range(20):
        number_steps = 0
        obs = eval_env.reset()
        for _ in range(400):
            action, _ = model.predict(obs)
            obs, reward, done, _ = eval_env.step(action)
            number_steps += 1
            # print(obs['achieved_goal'][0:3], obs['desired_goal'][0:3], reward)
            if done:
                step_num_arr.append(number_steps)
                count += 1
                print("----------------It reached terminal state -------------------")
                break
    print(
        "PSM grasped the needle ",
        count,
        " times and the Average step count was ",
        np.average(np.array(step_num_arr))
    )


if __name__ == '__main__':
    ENV_NAME = 'psm/baselink'
    # env_kwargs = {
    #     'action_space_limit': 0.05,
    #     'goal_position_range': 0.05,
    #     'position_error_threshold': 0.01,
    #     'goal_error_margin': 0.0075,
    #     'joint_limits':
    #     {
    #         'lower_limit': np.array([-0.2,
    #                                 -0.2,
    #                                 0.1,
    #                                 -1.5,
    #                                 -1.5,
    #                                 -1.5,
    #                                 -1.5]),
    #         'upper_limit': np.array([0.2,
    #                                 0.2,
    #                                 0.24,
    #                                 1.5,
    #                                 1.5,
    #                                 1.5,
    #                                 1.5])
    #     },
    #     'workspace_limits':
    #     {
    #         'lower_limit': np.array([-0.04, -0.03, -0.2]),
    #         'upper_limit': np.array([0.03, 0.04, -0.091])
    #     },
    #     'enable_step_throttling': False,
    #     'steps_to_print': 10000
    # }
    # Training
    print("Creating training_env")
    src_env = SRCEnv()
    # src_env = SRCEnv(**env_kwargs)
    time.sleep(5)
    # src_env.make(ENV_NAME)
    src_env.reset()

    main(training_env=src_env)  #, eval_env=eval_env)
    src_env._client.clean_up()

    # Evaluate learnt policy
    print("Creating eval_env")
    # eval_env = SRCEnv(**env_kwargs)
    eval_env = SRCEnv()
    time.sleep(5)
    # eval_env.make(ENV_NAME)
    eval_env.reset()

    load_model(eval_env=eval_env)
    eval_env._client.clean_up()
>>>>>>> 7e53c47c524c057c7b586904457ceee746dfe523
