# #!/usr/bin/env python3
# USE_TEAMGRID = True
#
# import argparse
# import gym
# import time
# from utils.config import parse_bool
# import imageio
#
# import matplotlib.pyplot as plt
#
# if USE_TEAMGRID:
#     import teamgrid
# else:
#     # import gym_minigrid
#     import multiagent
#     from make_env import make_env
#     from multiagent.environment import MultiAgentEnv
#     import multiagent.scenarios as scenarios
#
#
# import utils
#

import pdb
import argparse
import gym
import time
import logging
import torch
import torch_rl
import sys
import os
from pathlib import Path
from tqdm import tqdm
from utils.plots import *
import numpy as np
from utils.general import round_to_two
import teamgrid
import multiagent
from make_env import make_env

import utils
from utils import parse_bool
from model import ACModel
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
parser.add_argument("--num_episodes", type=int, default=30,
                    help="number of episodes to show")
parser.add_argument("--shift", type=int, default=0,
                    help="number of times the environment is reset at the beginning (default: 0)")
parser.add_argument("--argmax", action="store_true", default=False,
                    help="select the action with highest probability")
parser.add_argument("--fps", default=1, type=int,
                    help="speed at which frames are displayed")
parser.add_argument("--save_gifs", type=parse_bool, default=False,
                        help="Saves gif of each episode into model directory")

# arguments to replace flag
parser.add_argument("--use_teamgrid", type=parse_bool, default=True)
parser.add_argument("--use_switch", type=parse_bool, default=True)
parser.add_argument("--use_central_critic", type=parse_bool, default=True)
parser.add_argument("--use_always_broadcast", type=parse_bool, default=True)
parser.add_argument("--shared_rewards", type=parse_bool, default=True)
args = parser.parse_args()

# Set seed for all randomness sources

utils.seed(args.seed)

# Creates directory manager

dir_manager = utils.DirectoryManager(args.storage_dir, args.seed_dir, args.experiment_dir)

# Generate environment

train_config = utils.load_config_from_json(filename=dir_manager.seed_dir / "config.json")

if args.env is None:
    args.env = train_config.env


envs =[]
for i in range(train_config.procs):
    if args.use_teamgrid:
        env = gym.make(args.env, shared_rewards=args.shared_rewards)
    else:
        env = make_env(train_config.scenario, train_config.benchmark)
    env.seed(args.seed)
    envs.append(env)

for _ in range(args.shift):
    env.reset()

# Rendering parameters

frames = []
#num_frames = []
ifi = 1. / args.fps

# Define agent

agent = utils.Agent(args.env, env.observation_space, dir_manager.seed_dir, train_config.num_agents, args.argmax)
acmodel = agent.acmodel
preprocess_obss = agent.preprocess_obss
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

        renderer = env.render('human', close=True) # close=False for visualization
    # action selection

    actions = agent.get_action(obss)
    print('actions', actions)

    # environment step

    obss, rewards, done, _ = env.step(actions)
    #print('done', done)

    for j, reward in enumerate(rewards):
        agent.analyze_feedback(reward, done)


    if train_config.algo == "a2c":
        algo = torch_rl.A2CAlgo(num_agents=train_config.num_agents, envs=envs, acmodel=acmodel, replay_buffer=train_config.replay_buffer, \
                                num_frames_per_proc=train_config.frames_per_proc, discount=train_config.discount, lr=train_config.lr, \
                                gae_lambda=train_config.gae_lambda,
                                entropy_coef=train_config.entropy_coef, value_loss_coef=train_config.value_loss_coef, \
                                max_grad_norm=train_config.max_grad_norm, recurrence=train_config.recurrence,
                                rmsprop_alpha=train_config.optim_alpha, rmsprop_eps=train_config.optim_eps, \
                                preprocess_obss=preprocess_obss, num_options=train_config.num_options)
    elif train_config.algo == "ppo":
        algo = torch_rl.PPOAlgo(num_agents=train_config.num_agents, envs=envs, acmodel=acmodel, replay_buffer=train_config.replay_buffer, \
                                num_frames_per_proc=train_config.frames_per_proc, discount=train_config.discount, lr=train_config.lr, \
                                gae_lambda=train_config.gae_lambda,
                                entropy_coef=train_config.entropy_coef, value_loss_coef=train_config.value_loss_coef, max_grad_norm=train_config.max_grad_norm, \
                                recurrence=train_config.recurrence,
                                adam_eps=train_config.optim_eps, clip_eps=train_config.clip_eps, epochs=train_config.epochs, batch_size=train_config.batch_size, \
                                preprocess_obss=preprocess_obss)
    elif train_config.algo == "oc":
        algo = torch_rl.OCAlgo(num_agents=train_config.num_agents, envs=envs, acmodel=acmodel, replay_buffer=train_config.replay_buffer, \
                               num_frames_per_proc=train_config.frames_per_proc, discount=train_config.discount, lr=train_config.lr, \
                               gae_lambda=train_config.gae_lambda,
                               entropy_coef=train_config.entropy_coef, value_loss_coef=train_config.value_loss_coef, \
                               max_grad_norm=train_config.max_grad_norm, recurrence=train_config.recurrence,
                               rmsprop_alpha=train_config.optim_alpha, rmsprop_eps=train_config.optim_eps, preprocess_obss=preprocess_obss,
                               num_options=ctrain_config.num_options, termination_loss_coef=train_config.termination_loss_coef, \
                               termination_reg=train_config.termination_reg)
    elif train_config.algo == "doc":
        #config.recurrence = 2
        algo = torch_rl.DOCAlgo(num_agents=train_config.num_agents, envs=envs, acmodel=acmodel, replay_buffer=train_config.replay_buffer, \
                                num_frames_per_proc=train_config.frames_per_proc, discount=train_config.discount, lr=train_config.lr, \
                                gae_lambda=train_config.gae_lambda,
                               entropy_coef=train_config.entropy_coef, value_loss_coef=train_config.value_loss_coef, max_grad_norm=train_config.max_grad_norm, \
                                recurrence=train_config.recurrence,
                               rmsprop_alpha=train_config.optim_alpha, rmsprop_eps=train_config.optim_eps, preprocess_obss=preprocess_obss,
                               num_options=train_config.num_options, termination_loss_coef=train_config.termination_loss_coef, \
                                termination_reg=train_config.termination_reg)

    elif train_config.algo == "maddpg":
        algo = torch_rl.MADDPGAlgo(num_agents=train_config.num_agents, envs=envs, acmodel=acmodel, replay_buffer=train_config.replay_buffer, \
                                   tau = train_config.tau, num_frames_per_proc=train_config.frames_per_proc, discount=train_config.discount, lr=train_config.lr, \
                                   gae_lambda=train_config.gae_lambda,
                 entropy_coef=config.entropy_coef, value_loss_coef=train_config.value_loss_coef, max_grad_norm=train_config.max_grad_norm, \
                                   recurrence=train_config.recurrence,
                 adam_eps=train_config.optim_eps, clip_eps=config.clip_eps, epochs=train_config.epochs, er_batch_size=train_config.er_batch_size, \
                                   preprocess_obss=preprocess_obss)

    #save data

    graph_data = {
        "num_frames": [],
        "return_with_broadcast_penalties_mean": [],
        "return_with_broadcast_penalties_std": [],
        "mean_agent_return_with_broadcast_penalties_mean": [],
        "mean_agent_return_with_broadcast_penalties_std": [],
        "entropy": [],
        "broadcast_entropy": [],
        "policy_loss": [],
        "broadcast_loss": [],
        "value_loss": [],
        "options": [],
        "actions": [],
        "broadcasts": []
    }

    logs = algo.update_parameters()
    #num_frames.append(i)
    return_per_episode = utils.synthesize(logs["return_per_episode"])
    return_per_episode_with_broadcast_penalties = utils.synthesize(logs["return_per_episode_with_broadcast_penalties"])
    mean_agent_return_per_episode_with_broadcast_penalties = utils.synthesize(
        logs["mean_agent_return_with_broadcast_penalties"])
    num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])
    options = logs["options"]
    actions = logs["actions"]
    broadcasts = logs["broadcasts"]

    #status = {"num_frames": num_frames, "update": update}

    # Saving data for graphs

    graph_data["num_frames"].append(i)
    #graph_data["return_mean"].append(return_per_episode['mean'])
    #graph_data["return_std"].append(return_per_episode['std'])
    graph_data["return_with_broadcast_penalties_mean"].append(return_per_episode_with_broadcast_penalties['mean'])
    graph_data["return_with_broadcast_penalties_std"].append(return_per_episode_with_broadcast_penalties['std'])
    graph_data["mean_agent_return_with_broadcast_penalties_mean"].append(
        mean_agent_return_per_episode_with_broadcast_penalties['mean'])
    graph_data["mean_agent_return_with_broadcast_penalties_std"].append(
        mean_agent_return_per_episode_with_broadcast_penalties['std'])
    #graph_data["episode_length_mean"].append(num_frames_per_episode['mean'])
    #graph_data["episode_length_std"].append(num_frames_per_episode['std'])
    graph_data["entropy"].append(logs["entropy"])
    graph_data["broadcast_entropy"].append(logs["broadcast_entropy"])
    graph_data["policy_loss"].append(logs["policy_loss"])
    graph_data["broadcast_loss"].append(logs["broadcast_loss"])
    graph_data["value_loss"].append(logs["value_loss"])
    #graph_data["grad_norm"].append(logs["grad_norm"])
    graph_data["options"].append(options)
    graph_data["actions"].append(actions)
    graph_data["broadcasts"].append(broadcasts)
    #print("brd", graph_data["broadcasts"])

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
