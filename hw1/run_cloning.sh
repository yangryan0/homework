#!/bin/bash

python beh_cloning.py ./rollouts/Hopper-v2-5.pkl Hopper-v2 5
python beh_cloning.py ./rollouts/Hopper-v2-10.pkl Hopper-v2 10
python beh_cloning.py ./rollouts/Hopper-v2-20.pkl Hopper-v2 20
python beh_cloning.py ./rollouts/Hopper-v2-40.pkl Hopper-v2 40
python beh_cloning.py ./rollouts/Hopper-v2-80.pkl Hopper-v2 80

python beh_cloning.py ./rollouts/Humanoid-v2-80.pkl Humanoid-v2 80
python beh_cloning.py ./rollouts/Ant-v2-80.pkl Ant-v2 80
python beh_cloning.py ./rollouts/HalfCheetah-v2-80.pkl HalfCheetah-v2 80
python beh_cloning.py ./rollouts/Reacher-v2-80.pkl Reacher-v2 80
python beh_cloning.py ./rollouts/Walker2d-v2-80.pkl Walker2d-v2 80

python dagger.py ./experts/Hopper-v2.pkl ./rollouts/Hopper-v2-5.pkl Hopper-v2 5 20

python deep_BC.py ./rollouts/Hopper-v2-5.pkl Hopper-v2 5
python deep_BC.py ./rollouts/Hopper-v2-10.pkl Hopper-v2 10
python deep_BC.py ./rollouts/Hopper-v2-20.pkl Hopper-v2 20
python deep_BC.py ./rollouts/Hopper-v2-40.pkl Hopper-v2 40
python deep_BC.py ./rollouts/Hopper-v2-80.pkl Hopper-v2 80