#!/usr/bin/env python3
USE_TEAMGRID = True

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

if USE_TEAMGRID:
    import teamgrid
else:
    import gym_minigrid

import utils
from utils import parse_bool
from model import ACModel

# Parse arguments

def get_training_args(overwritten_args=None):
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--algo", default='doc', choices=['doc', 'oc', 'a2c', 'ppo'],#required=True,
                        help="algorithm to use: a2c | ppo | oc (REQUIRED)")
    parser.add_argument("--env", default='TEAMGrid-FourRooms-v0', #required=True,
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
    parser.add_argument("--log_interval", type=int, default=1,
                        help="number of updates between two logs (default: 1)")
    parser.add_argument("--save_interval", type=int, default=100,
                        help="number of updates between two saves (default: 0, 0 means no saving)")
    parser.add_argument("--tb", type=parse_bool, default=True,
                        help="log into Tensorboard")
    parser.add_argument("--frames_per_proc", type=int, default=None,
                        help="number of frames per process before update (default: 5 for A2C and 128 for PPO)")
    parser.add_argument("--discount", type=float, default=0.99,
                        help="discount factor (default: 0.99)")
    parser.add_argument("--lr", type=float, default=7e-4,
                        help="learning rate for optimizers (default: 7e-4)")
    parser.add_argument("--gae_lambda", type=float, default=0.95,
                        help="lambda coefficient in GAE formula (default: 0.95, 1 means no gae)")
    parser.add_argument("--entropy_coef", type=float, default=0.01,
                        help="entropy term coefficient (default: 0.01)")
    parser.add_argument("--value_loss_coef", type=float, default=0.5,
                        help="value loss term coefficient (default: 0.5)")
    parser.add_argument("--max_grad_norm", type=float, default=0.5,
                        help="maximum norm of gradient (default: 0.5)")
    parser.add_argument("--optim_eps", type=float, default=1e-5,
                        help="Adam and RMSprop optimizer epsilon (default: 1e-5)")
    parser.add_argument("--optim_alpha", type=float, default=0.99,
                        help="RMSprop optimizer apha (default: 0.99)")
    parser.add_argument("--clip_eps", type=float, default=0.2,
                        help="clipping epsilon for PPO (default: 0.2)")
    parser.add_argument("--epochs", type=int, default=4,
                        help="number of epochs for PPO (default: 4)")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="batch size for PPO (default: 256)")
    parser.add_argument("--recurrence", type=int, default=1,
                        help="number of timesteps gradient is backpropagated (default: 1)\nIf > 1, a LSTM is added to the model to have memory")
    parser.add_argument("--text", action="store_true", default=False,
                        help="add a GRU to the model to handle text input")
    parser.add_argument("--auto_resume", action="store_true", default=False,
                        help="whether to automatically resume training when lauching the script on existing model")
    # Option-Critic configs
    parser.add_argument("--num_options", type=int, default=3,
                        help="number of options (default: 1, 1 means no options)")
    parser.add_argument("--termination_loss_coef", type=float, default=0.5,
                        help="termination loss term coefficient (default: 0.5)")
    parser.add_argument("--termination_reg", type=float, default=0.01,
                        help="termination regularization constant (default: 0.01)")
    # Multi-Agent configs
    parser.add_argument("--num_agents", type=int, default=2,
                        help="number of trainable agents interacting with the teamgrid environment")

    return parser.parse_args(overwritten_args)


def train(config, dir_manager=None, logger=None, pbar="default_pbar"):

    config.mem = config.recurrence > 1
    # In the multi-agent setup, for DOC, different agents really are empty slots in which options are executed
    # Therefore, the number of options (policies) needs to be greater or equal to the number of agents (slots)
    if config.algo == 'doc':
        assert config.num_options >= config.num_agents
    # In the multi-agent setup, for baseline algorithms, each agent has its own policy
    # However, for implementation uniformity, we will consider them as if they were different options
    # (but each agent will always keep the same "option")
    elif config.algo in ['a2c', 'ppo']:
        config.num_options = config.num_agents

    if dir_manager is None:

        # Define save dir

        git_hash = "{0}_{1}".format(utils.get_git_hash(path='.'), utils.get_git_hash(path=str(os.path.dirname(teamgrid.__file__))))
        storage_dir = f"{git_hash}_{config.desc}"
        dir_manager = utils.DirectoryManager(storage_dir, config.seed, config.experiment_dir)
        dir_manager.create_directories()

    # Define logger, CSV writer and Tensorboard writer

    if logger is None:
        logger = utils.create_logger(name="", loglevel=logging.DEBUG,
                                     logfile=dir_manager.seed_dir / "log.txt", streamHandle=False)

    csv_file, csv_writer = utils.get_csv_writer(save_dir=dir_manager.seed_dir)
    if config.tb:
        from tensorboardX import SummaryWriter
        tb_writer = SummaryWriter(str(dir_manager.seed_dir))

    # Log command and all script arguments

    logger.debug("{}\n".format(" ".join(sys.argv)))
    logger.debug("{}\n".format(config))

    # Set seed for all randomness sources

    utils.seed(config.seed)

    # Generate environments

    envs = []
    for i in range(config.procs):
        env = gym.make(config.env, num_agents=config.num_agents)
        env.seed(config.seed + 10000*i)
        envs.append(env)

    # Define obss preprocessor

    obs_space, preprocess_obss = utils.get_obss_preprocessor(config.env, envs[0].observation_space, dir_manager.seed_dir)

    # Load training status

    try:
        status = utils.load_status(save_dir=dir_manager.seed_dir)
    except OSError:
        status = {"num_frames": 0, "update": 0}

    # Define actor-critic model

    if Path(utils.get_model_path(save_dir=dir_manager.seed_dir)).exists():
        if config.auto_resume or \
                input(f'Model in "{dir_manager.seed_dir}" already exists. Resume training? [y or n]').lower() in ['y', 'yes']:
            acmodel = utils.load_model(save_dir=dir_manager.seed_dir)
            logger.debug("Model successfully loaded\n")
            config = utils.load_config_from_json(filename=Path(dir_manager.seed_dir) / "config.json")

        else:
            print("Aborting...")
            sys.exit()
    else:
        acmodel = ACModel(obs_space=obs_space,
                          action_space=envs[0].action_space,
                          use_memory=config.mem,
                          use_text=config.text,
                          num_agents=config.num_agents,
                          num_options=config.num_options,
                          use_act_values=True if config.algo in ["oc", "doc"] else False,
                          use_term_fn=True if config.algo in ["oc", "doc"] else False,
                          use_central_critic=True if config.algo == "doc" else False,
                          use_broadcasting=True if config.algo == "doc" else False,
                          )

        logger.debug("Model successfully created\n")
        utils.save_config_to_json(config, filename=Path(dir_manager.seed_dir) / "config.json")

    # Print info on model

    logger.debug("{}\n".format(acmodel))
    logger.debug(f"Numer of params: {acmodel.get_number_of_params()}")

    if torch.cuda.is_available():
        acmodel.cuda()
    logger.debug("CUDA available: {}\n".format(torch.cuda.is_available()))

    # Define actor-critic algo

    if config.algo == "a2c":
        algo = torch_rl.A2CAlgo(config.num_agents, envs, acmodel, config.frames_per_proc, config.discount, config.lr, config.gae_lambda,
                                config.entropy_coef, config.value_loss_coef, config.max_grad_norm, config.recurrence,
                                config.optim_alpha, config.optim_eps, preprocess_obss, config.num_options)
    elif config.algo == "ppo":
        algo = torch_rl.PPOAlgo(config.num_agents, envs, acmodel, config.frames_per_proc, config.discount, config.lr, config.gae_lambda,
                                config.entropy_coef, config.value_loss_coef, config.max_grad_norm, config.recurrence,
                                config.optim_eps, config.clip_eps, config.epochs, config.batch_size, preprocess_obss, config.num_options)
    elif config.algo == "oc":
        algo = torch_rl.OCAlgo(config.num_agents, envs, acmodel, config.frames_per_proc, config.discount, config.lr, config.gae_lambda,
                               config.entropy_coef, config.value_loss_coef, config.max_grad_norm, config.recurrence,
                               config.optim_alpha, config.optim_eps, preprocess_obss,
                               config.num_options, config.termination_loss_coef, config.termination_reg)
    elif config.algo == "doc":
        algo = torch_rl.DOCAlgo(config.num_agents, envs, acmodel, config.frames_per_proc, config.discount, config.lr, config.gae_lambda,
                               config.entropy_coef, config.value_loss_coef, config.max_grad_norm, config.recurrence,
                               config.optim_alpha, config.optim_eps, preprocess_obss,
                               config.num_options, config.termination_loss_coef, config.termination_reg)

    else:
        raise ValueError("Incorrect algorithm name: {}".format(config.algo))



    # Creates a progress-bar

    if type(pbar) is str:
        if pbar == "default_pbar":
            pbar = tqdm()

    if pbar is not None:
        pbar.n = status["num_frames"]
        pbar.total = config.frames
        pbar.desc = f'{dir_manager.storage_dir.name}/{dir_manager.experiment_dir.name}/{dir_manager.seed_dir.name}'

    # Train model

    num_frames = status["num_frames"]
    total_start_time = time.time()
    update = status["update"]

    # Initialize some containers

    graph_data = {
        "num_frames": [],
        "return_mean": [],
        "return_std": [],
        "entropy": [],
        "policy_loss": [],
        "value_loss": [],
        "grad_norm": []
    }
    graph_data["agent_colors"] = [envs[0].agents[j].color for j in range(config.num_agents)]

    while num_frames < config.frames:
        # Update model parameters

        update_start_time = time.time()
        logs = algo.update_parameters()
        update_end_time = time.time()

        num_frames += logs["num_frames"][0]
        if pbar is not None:
            pbar.update(logs["num_frames"][0])
        update += 1

        # Print logs

        n_updates = config.frames // (algo.num_frames_per_proc * config.procs)
        if update == 1 or (num_frames != config.frames and update % (n_updates // 3) == 0):
            logger.info(f"Frames {num_frames}/{config.frames}, speed={round_to_two(logs['num_frames'][0]/(update_end_time - update_start_time))}fps")

        if update % config.log_interval == 0:
            fps = logs["num_frames"][0]/(update_end_time - update_start_time)

            duration = int(time.time() - total_start_time)
            return_per_episode = utils.synthesize(logs["return_per_episode"])
            rreturn_per_episode = utils.synthesize(logs["reshaped_return_per_episode"])
            num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])

            status = {"num_frames": num_frames, "update": update}

            # Saving data for graphs

            graph_data["num_frames"].append(num_frames)
            graph_data["return_mean"].append(return_per_episode['mean'])
            graph_data["return_std"].append(return_per_episode['std'])
            graph_data["entropy"].append(logs["entropy"])
            graph_data["policy_loss"].append(logs["policy_loss"])
            graph_data["value_loss"].append(logs["value_loss"])
            graph_data["grad_norm"].append(logs["grad_norm"])

            utils.save_graph_data(graph_data, save_dir=dir_manager.seed_dir)

        # Save vocabulary and model

        if config.save_interval > 0 and update % config.save_interval == 0:
            if hasattr(preprocess_obss, "vocab"):
                preprocess_obss.vocab.save()

            if torch.cuda.is_available():
                acmodel.cpu()
            utils.save_model(acmodel, save_dir=dir_manager.seed_dir)
            logger.debug("Model successfully saved")
            if torch.cuda.is_available():
                acmodel.cuda()

            utils.save_status(status, save_dir=dir_manager.seed_dir)

            # Saving graphs

                # Losses
            fig, axes = create_fig((2,2))
            plot_curve(axes[0,0], graph_data["num_frames"], np.array(graph_data["policy_loss"]).T, labels=[f"agent {i}" for i in range(config.num_agents)], colors=[envs[0].agents[j].color for j in range(config.num_agents)], xlabel="frames", title="Policy Loss")
            plot_curve(axes[0,1], graph_data["num_frames"], np.array(graph_data["value_loss"]).T, labels=[f"agent {i}" for i in range(config.num_agents)], colors=[envs[0].agents[j].color for j in range(config.num_agents)], xlabel="frames", title="Value Loss")
            plot_curve(axes[1,0], graph_data["num_frames"], np.array(graph_data["entropy"]).T, labels=[f"agent {i}" for i in range(config.num_agents)], colors=[envs[0].agents[j].color for j in range(config.num_agents)], xlabel="frames", title="Entropy")
            plot_curve(axes[1,1], graph_data["num_frames"], np.array(graph_data["grad_norm"]).T, labels=[f"agent {i}" for i in range(config.num_agents)], colors=[envs[0].agents[j].color for j in range(config.num_agents)], xlabel="frames", title="Gradient Norm")
            fig.savefig(str(dir_manager.seed_dir / 'curves.png'))
            plt.close(fig)

                # Return
            fig, ax = create_fig((1, 1))
            plot_curve(ax, graph_data["num_frames"],
                       np.array(graph_data["return_mean"]).T,
                       stds=np.array(graph_data["return_std"]).T,
                       colors=[envs[0].agents[j].color for j in range(config.num_agents)],
                       labels=[f"agent {i}" for i in range(config.num_agents)],
                       xlabel="frames", title="Average Return")
            fig.savefig(str(dir_manager.seed_dir / 'return.png'))
            plt.close(fig)

if __name__ == "__main__":
    config = get_training_args()
    train(config)
