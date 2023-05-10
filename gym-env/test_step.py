import pandas as pd
from SRCEnv import SRCEnv


psm1_bl = pd.read_csv('./trajectories/walee-attempt2/2023-05-06-17-42-38-psm1-baselink-state.csv')
psm2_bl = pd.read_csv('./trajectories/walee-attempt2/2023-05-06-17-42-38-psm2-baselink-state.csv')
joint = 'field.joint_positions'

# TODO: double check, there are 8 joint states recorded but only 7 DoF
joint_states_psm1 = []
joint_states_psm2 = []
for i in range(len(psm1_bl[joint + '0'])):
    joint_states_psm1.append([psm1_bl[joint + str(num)][i] for num in range(7)])
    joint_states_psm2.append([psm2_bl[joint + str(num)][i] for num in range(7)])

env = SRCEnv()
env.reset()
for i in range(len(psm1_bl[joint + '0'])):
    env.step(joint_states_psm2[i])
    env.render()