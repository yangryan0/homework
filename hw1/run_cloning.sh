#!/bin/bash

python beh_cloning.py ./rollouts/Hopper-v2-5.pkl Hopper-v2 5
python beh_cloning.py ./rollouts/Hopper-v2-10.pkl Hopper-v2 10
python beh_cloning.py ./rollouts/Hopper-v2-20.pkl Hopper-v2 20
python beh_cloning.py ./rollouts/Hopper-v2-40.pkl Hopper-v2 40
python beh_cloning.py ./rollouts/Hopper-v2-80.pkl Hopper-v2 80

python beh_cloning.py ./rollouts/Humanoid-v2-5.pkl Humanoid-v2 5
python beh_cloning.py ./rollouts/Humanoid-v2-10.pkl Humanoid-v2 10
python beh_cloning.py ./rollouts/Humanoid-v2-20.pkl Humanoid-v2 20
python beh_cloning.py ./rollouts/Humanoid-v2-40.pkl Humanoid-v2 40
python beh_cloning.py ./rollouts/Humanoid-v2-80.pkl Humanoid-v2 80