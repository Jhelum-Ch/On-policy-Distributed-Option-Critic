#!/usr/bin/env python3
USE_TEAMGRID = False

import argparse
import gym
import time

if USE_TEAMGRID:
    import teamgrid
else:
    import gym_minigrid


import utils

# Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("--storage_dir", required=True,
                    help="name of the storage folder in which the model is saved (REQUIRED)")
parser.add_argument("--experiment_dir", type=int, default=1,
                    help="number of the experiment folder inside in which the model is saved")
parser.add_argument("--seed_dir", type=int, default=1,
                    help="number of the seed folder inside in which the model is saved")
parser.add_argument("--seed", type=int, default=0,
                    help="random seed (default: 0) for the evaluation run")
parser.add_argument("--env", default=None,
                    help="name of the environment to be run"
                         "if None, env will be taken from saved args.json"
                         "which is the env on which the model was trained")
parser.add_argument("--shift", type=int, default=0,
                    help="number of times the environment is reset at the beginning (default: 0)")
parser.add_argument("--argmax", action="store_true", default=False,
                    help="select the action with highest probability")
parser.add_argument("--pause", type=float, default=0.1,
                    help="pause duration between two consequent actions of the agent")
args = parser.parse_args()

# Set seed for all randomness sources

utils.seed(args.seed)

# Creates directory manager

dir_manager = utils.DirectoryManager(args.storage_dir, args.seed_dir, args.experiment_dir)

# Generate environment

train_args = utils.load_config_from_json(filename=dir_manager.seed_dir/"args.json")

if args.env is None:
    args.env = train_args.env

env = gym.make(args.env)
env.seed(args.seed)
for _ in range(args.shift):
    env.reset()

# Define agent

agent = utils.Agent(args.env, env.observation_space, dir_manager.seed_dir, args.argmax)

# Run the agent

done = True

while True:
    if done:
        obs = env.reset()

    time.sleep(args.pause)
    renderer = env.render()

    action = agent.get_action(obs)
    obs, reward, done, _ = env.step(action)
    agent.analyze_feedback(reward, done)

    if renderer.window is None:
        break