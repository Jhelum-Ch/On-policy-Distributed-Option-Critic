#!/usr/bin/env python3
USE_TEAMGRID = False

import argparse
import gym
import time
import datetime
import torch
import torch_rl
import sys
import os
from pathlib import Path
import numpy as np

if USE_TEAMGRID:
    import teamgrid
else:
    import gym_minigrid

import utils
from utils import parse_bool
from model import ACModel

# Parse arguments

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("--algo", default='oc', #required=True,
                    help="algorithm to use: a2c | ppo | oc (REQUIRED)")
parser.add_argument("--env", default='MiniGrid-DoorKey-5x5-v0', #required=True,
                    help="name of the environment to train on (REQUIRED)")
parser.add_argument("--desc", default="",
                    help="string added as suffix to git_hash to explain the experiments in this folder")
parser.add_argument("--experiment_dir", type=int, default=None,
                    help="the experiment number (inside storage_dir folder)"
                         "if that experiment already exists, you will be offered to resume training")
parser.add_argument("--seed", type=int, default=1,
                    help="random seed (default: 1)")
parser.add_argument("--procs", type=int, default=16,
                    help="number of processes (default: 16)")
parser.add_argument("--frames", type=int, default=10**7,
                    help="number of frames of training (default: 10e7)")
parser.add_argument("--log-interval", type=int, default=1,
                    help="number of updates between two logs (default: 1)")
parser.add_argument("--save-interval", type=int, default=10,
                    help="number of updates between two saves (default: 0, 0 means no saving)")
parser.add_argument("--tb", type=parse_bool, default=True,
                    help="log into Tensorboard")
parser.add_argument("--frames-per-proc", type=int, default=None,
                    help="number of frames per process before update (default: 5 for A2C and 128 for PPO)")
parser.add_argument("--discount", type=float, default=0.99,
                    help="discount factor (default: 0.99)")
parser.add_argument("--lr", type=float, default=7e-4,
                    help="learning rate for optimizers (default: 7e-4)")
parser.add_argument("--gae-lambda", type=float, default=0.95,
                    help="lambda coefficient in GAE formula (default: 0.95, 1 means no gae)")
parser.add_argument("--entropy-coef", type=float, default=0.01,
                    help="entropy term coefficient (default: 0.01)")
parser.add_argument("--value-loss-coef", type=float, default=0.5,
                    help="value loss term coefficient (default: 0.5)")
parser.add_argument("--max-grad-norm", type=float, default=0.5,
                    help="maximum norm of gradient (default: 0.5)")
parser.add_argument("--optim-eps", type=float, default=1e-5,
                    help="Adam and RMSprop optimizer epsilon (default: 1e-5)")
parser.add_argument("--optim-alpha", type=float, default=0.99,
                    help="RMSprop optimizer apha (default: 0.99)")
parser.add_argument("--clip-eps", type=float, default=0.2,
                    help="clipping epsilon for PPO (default: 0.2)")
parser.add_argument("--epochs", type=int, default=4,
                    help="number of epochs for PPO (default: 4)")
parser.add_argument("--batch-size", type=int, default=256,
                    help="batch size for PPO (default: 256)")
parser.add_argument("--recurrence", type=int, default=1,
                    help="number of timesteps gradient is backpropagated (default: 1)\nIf > 1, a LSTM is added to the model to have memory")
parser.add_argument("--text", action="store_true", default=False,
                    help="add a GRU to the model to handle text input")
parser.add_argument("--auto-resume", action="store_true", default=False,
                    help="whether to automatically resume training when lauching the script on existing model")
parser.add_argument("--num-options", type=int, default=None,
                    help="number of options (default: 1, 1 means no options)")
parser.add_argument("--termination-loss-coef", type=float, default=0.5,
                    help="termination loss term coefficient (default: 0.5)")
parser.add_argument("--termination-reg", type=float, default=0.01,
                    help="termination regularization constant (default: 0.01)")
args = parser.parse_args()
args.mem = args.recurrence > 1

if args.algo in ['a2c', 'ppo']: assert args.num_options is None

# Define run dir

git_hash = "{0}_{1}".format(utils.get_git_hash(path='.'),
                            utils.get_git_hash(path=str(os.path.dirname(gym_minigrid.__file__))))
storage_dir = f"{git_hash}_{args.desc}"
dir_manager = utils.DirectoryManager(storage_dir, args.seed, args.experiment_dir)
dir_manager.create_directories()

# Define logger, CSV writer and Tensorboard writer

logger = utils.create_logger(save_dir=dir_manager.seed_dir)
csv_file, csv_writer = utils.get_csv_writer(save_dir=dir_manager.seed_dir)
if args.tb:
    from tensorboardX import SummaryWriter
    tb_writer = SummaryWriter(str(dir_manager.seed_dir))

# Log command and all script arguments

logger.info("{}\n".format(" ".join(sys.argv)))
logger.info("{}\n".format(args))

# Set seed for all randomness sources

utils.seed(args.seed)

# Generate environments

envs = []
for i in range(args.procs):
    env = gym.make(args.env)
    env.seed(args.seed + 10000*i)
    envs.append(env)

# Define obss preprocessor

obs_space, preprocess_obss = utils.get_obss_preprocessor(args.env, envs[0].observation_space, dir_manager.seed_dir)

# Load training status

try:
    status = utils.load_status(save_dir=dir_manager.seed_dir)
except OSError:
    status = {"num_frames": 0, "update": 0}

# Define actor-critic model

if Path(utils.get_model_path(save_dir=dir_manager.seed_dir)).exists():
    if args.auto_resume or \
            input(f'Model in "{dir_manager.seed_dir}" already exists. Resume training? [y or n]').lower() in ['y', 'yes']:
        acmodel = utils.load_model(save_dir=dir_manager.seed_dir)
        logger.info("Model successfully loaded\n")
    else:
        print("Aborting...")
        sys.exit()
else:
    acmodel = ACModel(obs_space, envs[0].action_space, args.mem, args.text, args.num_options)
    logger.info("Model successfully created\n")
logger.info("{}\n".format(acmodel))
logger.info(f"Numer of params: {acmodel.get_number_of_params()}")

if torch.cuda.is_available():
    acmodel.cuda()
logger.info("CUDA available: {}\n".format(torch.cuda.is_available()))

# Saves config

utils.save_config_to_json(args, filename=Path(dir_manager.seed_dir)/"args.json")

# Define actor-critic algo

if args.algo == "a2c":
    algo = torch_rl.A2CAlgo(envs, acmodel, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                            args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                            args.optim_alpha, args.optim_eps, preprocess_obss)
elif args.algo == "ppo":
    algo = torch_rl.PPOAlgo(envs, acmodel, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                            args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                            args.optim_eps, args.clip_eps, args.epochs, args.batch_size, preprocess_obss)
elif args.algo == "oc":
    algo = torch_rl.OCAlgo(envs, acmodel, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                            args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                            args.optim_alpha, args.optim_eps, preprocess_obss,
                            args.num_options, args.termination_loss_coef, args.termination_reg)
else:
    raise ValueError("Incorrect algorithm name: {}".format(args.algo))

# Train model

num_frames = status["num_frames"]
total_start_time = time.time()
update = status["update"]

while num_frames < args.frames:
    # Update model parameters

    update_start_time = time.time()
    logs = algo.update_parameters()
    update_end_time = time.time()

    num_frames += logs["num_frames"]
    update += 1

    # Print logs

    if update % args.log_interval == 0:
        fps = logs["num_frames"]/(update_end_time - update_start_time)
        duration = int(time.time() - total_start_time)
        return_per_episode = utils.synthesize(logs["return_per_episode"])
        rreturn_per_episode = utils.synthesize(logs["reshaped_return_per_episode"])
        num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])

        header = ["update", "frames", "FPS", "duration"]
        data = [update, num_frames, fps, duration]
        header += ["rreturn_" + key for key in rreturn_per_episode.keys()]
        data += rreturn_per_episode.values()
        header += ["num_frames_" + key for key in num_frames_per_episode.keys()]
        data += num_frames_per_episode.values()
        header += ["entropy", "value", "policy_loss", "value_loss", "grad_norm"]
        data += [logs["entropy"], logs["value"], logs["policy_loss"], logs["value_loss"], logs["grad_norm"]]

        logger.info(
            "U {} | F {:06} | FPS {:04.0f} | D {} | rR:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | F:μσmM {:.1f} {:.1f} {} {} | H {:.3f} | V {:.3f} | pL {:.3f} | vL {:.3f} | ∇ {:.3f}"
            .format(*data))

        header += ["return_" + key for key in return_per_episode.keys()]
        data += return_per_episode.values()

        if status["num_frames"] == 0:
            csv_writer.writerow(header)
        csv_writer.writerow(data)
        csv_file.flush()

        if args.tb:
            for field, value in zip(header, data):
                tb_writer.add_scalar(field, value, num_frames)

        status = {"num_frames": num_frames, "update": update}

    # Save vocabulary and model

    if args.save_interval > 0 and update % args.save_interval == 0:
        preprocess_obss.vocab.save()

        if torch.cuda.is_available():
            acmodel.cpu()
        utils.save_model(acmodel, save_dir=dir_manager.seed_dir)
        logger.info("Model successfully saved")
        if torch.cuda.is_available():
            acmodel.cuda()

        utils.save_status(status, save_dir=dir_manager.seed_dir)