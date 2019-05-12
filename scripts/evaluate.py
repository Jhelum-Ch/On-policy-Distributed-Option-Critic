#!/usr/bin/env python3
USE_TEAMGRID = True

import argparse
import gym
import time
import torch
from torch_rl.utils.penv import ParallelEnv

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
parser.add_argument("--episodes", type=int, default=100,
                    help="number of episodes of evaluation (default: 100)")
parser.add_argument("--procs", type=int, default=16,
                    help="number of processes (default: 16)")
parser.add_argument("--argmax", action="store_true", default=False,
                    help="action with highest probability is selected")
parser.add_argument("--worst-episodes-to-show", type=int, default=10,
                    help="The number of worse episodes to show")
args = parser.parse_args()

# Set seed for all randomness sources

utils.seed(args.seed)

# Creates directory manager

dir_manager = utils.DirectoryManager(args.storage_dir, args.seed_dir, args.experiment_dir)

# Generate environment

train_config = utils.load_config_from_json(filename=dir_manager.seed_dir / "config.json")

if args.env is None:
    args.env = train_config.env

envs = []
for i in range(args.procs):
    env = gym.make(args.env, num_agents=train_config.num_agents)
    env.seed(args.seed + 10000*i)
    envs.append(env)
env = ParallelEnv(envs)

# Define agent

agent = utils.Agent(args.env, env.observation_space, dir_manager.seed_dir, args.argmax, args.procs)
print("CUDA available: {}\n".format(torch.cuda.is_available()))

# Initialize logs

logs = {"num_frames_per_episode": [], "return_per_episode": []}

# Run the agent

start_time = time.time()

obss = env.reset()

log_done_counter = 0
log_episode_return = torch.zeros(args.procs, device=agent.device)
log_episode_num_frames = torch.zeros(args.procs, device=agent.device)

while log_done_counter < args.episodes:
    actions = agent.get_actions(obss)
    obss, rewards, dones, _ = env.step(actions)
    agent.analyze_feedbacks(rewards, dones)

    log_episode_return += torch.tensor(rewards, device=agent.device, dtype=torch.float)
    log_episode_num_frames += torch.ones(args.procs, device=agent.device)

    for i, done in enumerate(dones):
        if done:
            log_done_counter += 1
            logs["return_per_episode"].append(log_episode_return[i].item())
            logs["num_frames_per_episode"].append(log_episode_num_frames[i].item())

    mask = 1 - torch.tensor(dones, device=agent.device, dtype=torch.float)
    log_episode_return *= mask
    log_episode_num_frames *= mask

end_time = time.time()

# Print logs

num_frames = sum(logs["num_frames_per_episode"])
fps = num_frames/(end_time - start_time)
duration = int(end_time - start_time)
return_per_episode = utils.synthesize(logs["return_per_episode"])
num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])

print("F {} | FPS {:.0f} | D {} | R:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | F:μσmM {:.1f} {:.1f} {} {}"
      .format(num_frames, fps, duration,
              *return_per_episode.values(),
              *num_frames_per_episode.values()))

# Print worst episodes

n = args.worst_episodes_to_show
if n > 0:
    print("\n{} worst episodes:".format(n))

    indexes = sorted(range(len(logs["return_per_episode"])), key=lambda k: logs["return_per_episode"][k])
    for i in indexes[:n]:
        print("- episode {}: R={}, F={}".format(i, logs["return_per_episode"][i], logs["num_frames_per_episode"][i]))