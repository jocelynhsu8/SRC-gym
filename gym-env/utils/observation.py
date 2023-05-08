import numpy as np

class Observation:

    def __init__(self, state, achieved_goal, desired_goal):
        self.state = state
        self.achieved_goal = achieved_goal
        self.desired_goal = desired_goal
        # self.dist = 0
        # self.reward = 0.0
        # self.prev_reward = 0.0
        # self.cur_reward = 0.0
        # self.is_done = False
        # self.is_truncated = False
        # self.info = {}
        # self.sim_step_no = 0

    def reset(self, state, achieved_goal):
        self.state = state
        self.achieved_goal = achieved_goal
        # self.dist = 0
        # self.reward = 0.0
        # self.prev_reward = 0.0
        # self.cur_reward = 0.0
        # self.is_done = False
        # self.is_truncated = False
        # self.info = {}
        # self.sim_step_no = 0

    def cur_observation(self):
        return np.array(self.state)
        # return np.array(self.state), np.array(self.reward), np.array(self.is_done), np.array(self.is_truncated), self.info