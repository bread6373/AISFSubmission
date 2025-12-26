# AISFSubmission

1. Description and Goal of the Project
The primary goal of this project is to train a Bipedal Walker to be able to walk a short distance to the right. To make it harder, obstacles and holes may be added to increase the difficulty. This uses Reinforcment Learning (RL) and a Proximal Policy Optimization (PPO) to train the walker. The algorithm is implemente with a custom reward function and the stable-baselines3 library in Python.

2. Directory Tree
├── FinalRecordings
│   ├── Hardcore
│   │   ├── best_reach_end.mov
│   │   ├── fail_hole.mov
│   │   ├── hardcore_fail.mov
│   │   ├── knee.mov
│   │   ├── lidar.mov
│   │   └── nolift.mov
│   └── Normal
│       └── easy_mode.mov
├── README.md
└── src
    ├── Hardcore
    │   ├── hardcode_tuning_knee.py
    │   ├── hardcode_tuning_knee_eval.py
    │   ├── hardcode_tuning_lidar.py
    │   ├── hardcode_tuning_lidar_eval.py
    │   ├── hardcode_tuning_nolift.py
    │   ├── hardcode_tuning_nolift_eval.py
    │   ├── hardcore_fail.py
    │   ├── hardcore_tuning4.py
    │   └── hardcore_tuning4_eval.py
    └── Normal
        ├── bipedal.gif
        ├── bipedal.py
        ├── bipedal_tuning1.gif
        ├── bipedal_tuning1.py
        └── tempCodeRunnerFile.py

3. Features & Reward Design
The main reward design is 