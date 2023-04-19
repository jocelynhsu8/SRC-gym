import numpy as np
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
