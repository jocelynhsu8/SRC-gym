# Open a terminal
1. roscd dvrk_config
2. rosrun dvrk_robot dvrk_console_json -j jhu-daVinci/console-MTML-MTMR.json
# Starts the MTMs control
# If you encounter arms not powerin on. In terminal
3. qlacommand -c close-relays
# If encounter, mismatch between pots and encs. In a new terminal
# Close the control GUI
4. qlacommand -c reset-encoder-preload
# Rerun 3
# Rerun 2

# New terminal
cd ~/surgical_robotics_challenge
5. ./run_environment.sh
# Move windows to the correct screens

# New terminal
cd ~/surgical_robotics_challenge/scripts/surgical_robotics_challenge/teleoperation
7. ./mtm_psm_pair_teleop.sh

# New terminal. To record.
cd rosbag_sr
8. rosbag record /ambf/env/psm1/baselink/Command /ambf/env/psm2/baselink/Command /ambf/env/psm1/Actuator0/Command /ambf/env/psm2/Actuator0/Command
OR
8. rosbag record /ambf/env/psm1/baselink/Command /ambf/env/psm2/baselink/Command /ambf/env/psm1/Actuator0/Command /ambf/env/psm2/Actuator0/Command /ambf/env/psm2/baselink/State /ambf/env/psm1/baselink/State /ambf/env/psm2/Actuator0/State /ambf/env/psm1/Actuator0/State

# Then to replay.
Close 7
# reset the ambf_world. In a new terminal
rostopic pub /ambf/env/World/Command/Reset std_msgs/Empty
rosbag play <whatever ros bag you want>

# convert rosbag files to csv
rostopic echo -b 2023-05-06-17-42-38.bag -p /ambf/env/psm1/baselink/State > 2023-05-06-17-42-38-psm1-baselink-state.csv

