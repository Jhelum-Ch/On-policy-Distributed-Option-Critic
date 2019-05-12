#!/usr/bin/env python3
USE_TEAMGRID = True

import argparse
import gym
import time
from utils.config import parse_bool
import imageio

import matplotlib.pyplot as plt

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
                         "if None, env will be taken from saved config.json"
                         "which is the env on which the model was trained")
parser.add_argument("--num_episodes", type=int, default=5,
                    help="number of episodes to show")
parser.add_argument("--shift", type=int, default=0,
                    help="number of times the environment is reset at the beginning (default: 0)")
parser.add_argument("--argmax", action="store_true", default=False,
                    help="select the action with highest probability")
parser.add_argument("--fps", default=10, type=int,
                    help="speed at which frames are displayed")
parser.add_argument("--save_gifs", type=parse_bool, default=False,
                        help="Saves gif of each episode into model directory")
args = parser.parse_args()

# Set seed for all randomness sources

utils.seed(args.seed)

# Creates directory manager

dir_manager = utils.DirectoryManager(args.storage_dir, args.seed_dir, args.experiment_dir)

# Generate environment

train_config = utils.load_config_from_json(filename=dir_manager.seed_dir / "config.json")

if args.env is None:
    args.env = train_config.env

env = gym.make(args.env, num_agents=train_config.num_agents)
env.seed(args.seed)
for _ in range(args.shift):
    env.reset()

# Rendering parameters

frames = []
ifi = 1. / args.fps

# Define agent

agent = utils.Agent(args.env, env.observation_space, dir_manager.seed_dir, train_config.num_agents, args.argmax)

# Run the agent

done = True

i = 0
ep = 0
while True:
    calc_start = time.time()

    i += 1
    if done:
        ep += 1
        if ep > args.num_episodes: break
        print(f"Episode {ep}")

        obss = env.reset()

    # renders the environment

    if args.save_gifs:
        frames.append(env.render('rgb_array'))

    else:
        # enforces the fps args
        elapsed = time.time() - calc_start
        if elapsed < ifi:
            time.sleep(ifi - elapsed)

        renderer = env.render('human')

    # action selection

    actions = agent.get_action(obss)

    # environment step

    obss, rewards, done, _ = env.step(actions)

    for j, reward in enumerate(rewards):
        agent.analyze_feedback(reward, done)

env.close()

# Saves the collected gifs

if args.save_gifs:
    gif_path = dir_manager.storage_dir / 'gifs'
    gif_path.mkdir(exist_ok=True)

    print("Saving gif...")

    gif_num = 0
    while (gif_path / f"{args.env}__experiment{args.experiment_dir}_seed{args.seed_dir}_{gif_num}.gif").exists():
        gif_num += 1
    imageio.mimsave(str(
        gif_path / f"{args.env}__experiment{args.experiment_dir}_seed{args.seed_dir}_{gif_num}.gif"), frames, duration=ifi)
