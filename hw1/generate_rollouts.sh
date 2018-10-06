#!/bin/bash

python run_expert.py experts/Hopper-v2.pkl Hopper-v2 --num_rollouts 5
python run_expert.py experts/Hopper-v2.pkl Hopper-v2 --num_rollouts 10
python run_expert.py experts/Hopper-v2.pkl Hopper-v2 --num_rollouts 20
python run_expert.py experts/Hopper-v2.pkl Hopper-v2 --num_rollouts 40
python run_expert.py experts/Hopper-v2.pkl Hopper-v2 --num_rollouts 80

python run_expert.py experts/Humanoid-v2.pkl Humanoid-v2 --num_rollouts 80
python run_expert.py experts/Ant-v2.pkl Ant-v2 --num_rollouts 80
python run_expert.py experts/HalfCheetah-v2.pkl HalfCheetah-v2 --num_rollouts 80
python run_expert.py experts/Reacher-v2.pkl Reacher-v2 --num_rollouts 80
python run_expert.py experts/Walker2d-v2.pkl Walker2d-v2 --num_rollouts 80