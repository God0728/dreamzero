robot_q 区段	含义	活动度（range / std）	解读
idx 0–6	base_pose（浮动基 pos3+quat4）	range≤0.13，std≤0.04	站立，几乎不动
idx 7–18 = joint[0–11]	双腿 12 关节	range 0.002–0.30，std 0.0002–0.086	几乎静止（站立平衡微调）
idx 19–21 = joint[12–14]	腰 waist 3 关节	range 0.08–0.30	小幅
idx 22–35 = joint[15–28]	双臂 14 关节	range 0.16–1.04，std 0.04–0.29	主运动，任务动作


MCAP 60D Name Sequence

Default eef: inspire
Hand layout: left hand 6 + right hand 6

State 60D definition:
- 0..35: robot_q_current = base(7) + robot joints(29)
- 36..47: hand_state = left hand(6) + right hand(6)
- 48..59: ee_state = current left_xyzrpy + current right_xyzrpy

Action 60D definition:
- 0..35: robot_q_desired = base(7) + robot joints(29)
- 36..47: hand_cmd = left hand(6) + right hand(6)
- 48..59: ee_state = desired left_xyzrpy + desired right_xyzrpy

Shared 0..35 robot layout
0  base_x
1  base_y
2  base_z
3  base_qw
4  base_qx
5  base_qy
6  base_qz
7  left_hip_pitch
8  left_hip_roll
9  left_hip_yaw
10 left_knee
11 left_ankle_pitch
12 left_ankle_roll
13 right_hip_pitch
14 right_hip_roll
15 right_hip_yaw
16 right_knee
17 right_ankle_pitch
18 right_ankle_roll
19 waist_yaw
20 waist_roll
21 waist_pitch
22 left_shoulder_pitch
23 left_shoulder_roll
24 left_shoulder_yaw
25 left_elbow
26 left_wrist_roll
27 left_wrist_pitch
28 left_wrist_yaw
29 right_shoulder_pitch
30 right_shoulder_roll
31 right_shoulder_yaw
32 right_elbow
33 right_wrist_roll
34 right_wrist_pitch
35 right_wrist_yaw

State 60D names
36 left_hand_state_0
37 left_hand_state_1
38 left_hand_state_2
39 left_hand_state_3
40 left_hand_state_4
41 left_hand_state_5
42 right_hand_state_0
43 right_hand_state_1
44 right_hand_state_2
45 right_hand_state_3
46 right_hand_state_4
47 right_hand_state_5
48 current_left_x
49 current_left_y
50 current_left_z
51 current_left_roll
52 current_left_pitch
53 current_left_yaw
54 current_right_x
55 current_right_y
56 current_right_z
57 current_right_roll
58 current_right_pitch
59 current_right_yaw

Action 60D names
36 left_hand_cmd_0
37 left_hand_cmd_1
38 left_hand_cmd_2
39 left_hand_cmd_3
40 left_hand_cmd_4
41 left_hand_cmd_5
42 right_hand_cmd_0
43 right_hand_cmd_1
44 right_hand_cmd_2
45 right_hand_cmd_3
46 right_hand_cmd_4
47 right_hand_cmd_5
48 desired_left_x
49 desired_left_y
50 desired_left_z
51 desired_left_roll
52 desired_left_pitch
53 desired_left_yaw
54 desired_right_x
55 desired_right_y
56 desired_right_z
57 desired_right_roll
58 desired_right_pitch
59 desired_right_yaw




base xyz:       [0.       0.       0.783826]
base quat wxyz: [1. 0. 0. 0.]

robot joints 29:
[ 0.053084  0.041477 -0.024735 -0.084403  0.019161 -0.024773 -0.000060
 -0.051107 -0.151302 -0.087667  0.083529  0.026442  0.009589 -0.035873
  0.149153  0.244958  0.436969  0.046846  0.142564 -0.314454 -0.309600
 -0.026713  0.170044 -0.470573  0.074865  0.130676  0.313363 -0.209472
  0.139604]

hand 12:
[0.    0.801 0.    0.    0.    0.    0.    0.799 0.    0.    0.    0.   ]

ee state 12:
[ 0.186809  0.265457  0.031517  0.127760  0.192463  0.108076
  0.209327 -0.231697  0.043041 -0.157459  0.280306  0.136455]