'''
Base Environment for Surgical Robotics Challenge
'''
import gymnasium as gym
import gymnasium_robotics
from gymnasium import spaces
import numpy as np
import time
import re

from PyKDL import Frame, Rotation, Vector
from gym.spaces.box import Box
from src.scripts.surgical_robotics_challenge.psm_arm import PSM
from src.scripts.surgical_robotics_challenge.ecm_arm import ECM
from src.scripts.surgical_robotics_challenge.scene import Scene
from src.scripts.surgical_robotics_challenge.simulation_manager import SimulationManager
from src.scripts.surgical_robotics_challenge.task_completion_report import TaskCompletionReport
from src.scripts.surgical_robotics_challenge.utils.task3_init import NeedleInitialization
from src.scripts.surgical_robotics_challenge.evaluation.evaluation import Task_2_Evaluation, Task_2_Evaluation_Report
from utils.observation import Observation
from utils.needle_kinematics import NeedleKinematics


def add_break(s):
    time.sleep(s)
    print('-------------')

class SRCEnv(gymnasium_robotics.core.GoalEnv):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        # Define action and observation space
        super(SRCEnv, self).__init__()
        # Limits for psm
        self.action_lims_low = np.array([np.deg2rad(-91.96), np.deg2rad(-60), -0.0, np.deg2rad(-175), np.deg2rad(-90), np.deg2rad(-85), 0])
        self.action_lims_high = np.array([np.deg2rad(91.96), np.deg2rad(60), 0.240, np.deg2rad(175), np.deg2rad(90), np.deg2rad(85), 1])
        self.action_space = spaces.Box(self.action_lims_low, self.action_lims_high)

        self.observation_space = spaces.Box(
            low=np.array([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, 0]), 
            high=np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 1]), 
            shape=(7,))

        # Connect to client using SimulationManager
        self.simulation_manager = SimulationManager('src_client')

        # Initialize simulation environment
        self.world_handle = self.simulation_manager.get_world_handle()
        self.scene = Scene(self.simulation_manager)
        # self.simulation_manager._client.print_summary()
        self.psm1 = PSM(self.simulation_manager, 'psm1')
        self.psm2 = PSM(self.simulation_manager, 'psm2')
        self.ecm = ECM(self.simulation_manager, 'CameraFrame')
        self.needle = NeedleInitialization(self.simulation_manager) # needle obj
        self._needle_kin = NeedleKinematics() # needle movement and positioning

        self.psm1.servo_jp([-0.4, -0.22, 0.139, -1.64, -0.37, -0.11])
        self.init_state = np.array([-0.4, -0.22, 0.139, -1.64, -0.37, -0.11, 0.8])
        self.init_achieved_goal = self.psm1.measured_cp()
        self.dt = 1/120
        self.observation = Observation(self.init_state, self.init_achieved_goal, self.get_needle_mid_in_world())
        self.info = {'calc_dist': True, 
                    'calc_angle': False, 
                    'grasp_completed': False, 
                    'insert_completed': False,
                    'target_completed': False, 
                    'sim_step_no': 0,
                    'max_sim_step': 100}

        # Small sleep to let the handles initialize properly
        add_break(0.5)
        return

    def get_needle_tip_in_world(self):
        """ Get the needle tip pose in world coordinates """
        return self._needle_kin.get_tip_pose()

    def get_needle_mid_in_world(self):
        """ Get the needle mid pose in world coordinates """
        return self._needle_kin.get_mid_pose()

    def _update_observation(self, action):
        """ Update the observation of the environment

        Parameters
        - action: an action provided by the environment
        """
        self.observation.state += action * self.dt
        self.psm1.servo_jp(self.observation.state[:6])
        self.psm1.set_jaw_angle(self.observation.state[6])
        self.observation.achieved_goal = self.psm1.measured_cp()
        self.info['sim_step_no'] += 1

    def reset(self):
        """ Reset the state of the environment to an initial state """
        print('reset func')
        self.world_handle.reset()
        self.psm1.servo_jp([-0.4, -0.22, 0.139, -1.64, -0.37, -0.11])
        self.psm1.set_jaw_angle(0.8)
        self.observation.reset(self.init_state, self.init_achieved_goal)
        add_break(3.0)
        return np.array(self.observation.achieved_goal, dtype=np.float32), self.info
    

    def step(self, action):
        """ Execute one time step within the environment

        Parameters
        - action: an action provided by the environment

        Returns
        - observation: agent's observation of the current environment after the action

        """
        # Limit PSM action to bounds
        print('current state: ', self.observation.achieved_goal)
        print('action: ', action)
        action = np.clip(action, self.action_lims_low, self.action_lims_high)
        self.action = action

        # Move PSM
        self.psm1.servo_jp(action)

        # Update simulation
        self.world_handle.update()
        self._update_observation(action)
        reward = self.compute_reward(self.observation.achieved_goal, self.observation.desired_goal, self.info)
        terminated = self.compute_terminated(self.observation.achieved_goal, self.observation.desired_goal, self.info)
        truncated = self.compute_truncated(self.observation.achieved_goal, self.observation.desired_goal, self.info)
        return self.observation, reward, terminated, truncated, self.info

    def compute_terminated(self, achieved_goal, desired_goal, info):
        """ Compute if the episode is terminated
        
        Parameters
        - obs: an action provided by the environment
        
        Returns
        - terminated: whether the episode is terminated
        """
        # TODO: update for other tasks once done with grasped
        terminated = False
        if info['calc_dist'] and not info['calc_angle']:
            terminated = self.calc_dist(achieved_goal, desired_goal) < 0.01
        self.info['grasp_completed'] = terminated
        return terminated

    def compute_truncated(self, achieved_goal, desired_goal, info):
        """ Compute if the episode is truncated
        
        Parameters
        - obs: an action provided by the environment
        
        Returns
        - truncated: whether the episode is truncated
        """
        return info['sim_step_no'] >= info['max_sim_step']

    def compute_reward(self, achieved_goal, desired_goal, info):
        """ Compute the cumulative reward for the action taken
        
        Parameters
        - obs: an action provided by the environment
        
        Returns
        - reward: the reward for the action
        """
        reward = 0
        if not info['grasp_completed']:
            if info['calc_dist']:
                reward += self.calc_dist(achieved_goal, desired_goal) * -1
            if info['calc_angle']:
                reward += self.calc_angle(achieved_goal, desired_goal) * -1
        # TODO: add reward for insert and target
        return reward

    def calc_angle(self, psm):
        """ Compute dot product of needle mid and specific psm tip
        
        Parameters
        - needle: the needle pose in world coordinates
        - psm: the psm pose in world coordinates
        
        Returns
        - angle: the angle between the needle and psm"""
        needle_R = str(self.get_needle_mid_in_world().M).replace('[', '').replace(']', '').replace('\n', ' ').replace(';', ' ').replace(',', ' ').split()
        needle_R_x = np.array([float(i) for i in needle_R]).reshape(3, 3)[0:3, 0:1].T
        needle_R_y = np.array([float(i) for i in needle_R]).reshape(3, 3)[0:3, 1:2].T
        psm_R_x = np.array(psm.measured_cp()[0:3, 0])
        psm_R_y = np.array(psm.measured_cp()[0:3, 1])
        return np.dot(needle_R_x, psm_R_x) + np.dot(needle_R_y, psm_R_y)

    def calc_dist(self, goal_pose, current_pose):
        """ Compute the distance between the goal pose and current pose

        Parameters
        - goal_pose: the goal pose in world coordinates
        - current_pose: the current pose in world coordinates

        Returns
        - dist: the distance between the goal pose and current pose
        """
        if type(goal_pose) == Frame:
            goal_pose_p = np.array([goal_pose.p.x(), goal_pose.p.y(), goal_pose.p.z()])
        elif type(goal_pose) == np.matrix or type(goal_pose) == np.ndarray:
            goal_pose_p = goal_pose[0:3, 3]

        if type(current_pose) == Frame:
            current_pose_p = np.array([current_pose.p.x(), current_pose.p.y(), current_pose.p.z()])
        elif type(current_pose) == np.matrix or type(current_pose) == np.ndarray:
            current_pose_p = current_pose[0:3, 3]

        dist = np.linalg.norm(goal_pose_p - current_pose_p)
        return dist

    

    def render(self, mode='human', close=False):
        '''
        1. run the simulation (using script or smt else) - init()
        2. use the client to update the robots' positions
        3. done. (ambf client will automatically update the positions of the robots in ambf sim and render it)
        '''
        # Render the environment to the screen
        print("PSM1 End-effector pose in Base Frame", self.psm1.measured_cp())
        print("PSM1 Base pose in World Frame", self.psm1.get_T_b_w())
        print("PSM1 Joint state", self.psm1.measured_jp())
        add_break(1.0)
        print("PSM2 End-effector pose in Base Frame", self.psm2.measured_cp())
        print("PSM2 Base pose in World Frame", self.psm2.get_T_b_w())
        print("PSM2 Joint state", self.psm2.measured_jp())
        add_break(1.0)
        print("Needle pose in Needle Frame", self.needle.get_tip_to_needle_offset())
        add_break(1.0)
        # Things are slightly different for ECM as the `measure_cp` returns pose in the world frame
        print("ECM pose in World", self.ecm.measured_cp())
        add_break(1.0)

        # Scene object poses are all w.r.t World
        print("Entry 1 pose in World", self.scene.entry1_measured_cp())
        print("Exit 4 pose in World", self.scene.exit4_measured_cp())


if __name__ == "__main__":
    env = SRCEnv()
    env.reset()
    print(env.observation.achieved_goal)
    env.step([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    print(env.observation.achieved_goal)
    # env.reset()
    # env.render()
