# USAGE
# This file needs to be called locally (from the serial directory itself)
# All the files in the "serial" folder act on the experiment folders contained in serial/models/...
import matplotlib
matplotlib.use('Agg')
import itertools
import argparse
from scripts.train import get_training_args
from utils.directory_tree import DirectoryManager
from utils.config import save_dict_to_json, load_dict_from_json, save_config_to_json, config_to_str
from utils.plots import plot_sampled_hyperparams
from utils.save import get_git_hash
import matplotlib.pyplot as plt
import os
import sys

USE_TEAMGRID = False
if USE_TEAMGRID:
    import teamgrid
else:
    import gym_minigrid


def get_prepare_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--desc', type=str, required=True)
    parser.add_argument('--search_type', type=str, default='grid', choices=['grid', 'random'])
    parser.add_argument('--n_experiments', type=int, default=15)
    return parser.parse_args()


if __name__ == '__main__':
    serial_args = get_prepare_args()

    # Adds code versions git-hash for current project and environment package as prefixes to storage_name
    git_hash = "{0}_{1}".format(get_git_hash(path='.'),
                                get_git_hash(path=str(os.path.dirname(gym_minigrid.__file__))))
    storage_dir = f"{git_hash}_{serial_args.desc}"

    # Creates dictionary pointer-access to a training config object initialized by default
    config = get_training_args(overwritten_args="")
    config.desc = serial_args.desc
    config_dict = vars(config)

    # GRID SEARCH
    if serial_args.search_type == 'grid':
        from scripts.schedule_grid import VARIATIONS
        VARIATIONS = dict(VARIATIONS)

        # Exclude seeds from VARIATIONS (they are treated separately as a special case)
        SEED_VARIATIONS = VARIATIONS['seed']
        del VARIATIONS['seed']

        # Transforms our dictionary of lists (key: list of values) into a list of lists of tuples (key, single_value)
        VARIATIONS_LISTS = []
        sorted_keys = sorted(VARIATIONS.keys(), key=lambda item: (len(VARIATIONS[item]), item), reverse=True)
        for key in sorted_keys:
            VARIATIONS_LISTS.append([(key, VARIATIONS[key][j]) for j in range(len(VARIATIONS[key]))])

        # Creates a list of combinations of hyperparams given in VARIATIONS (to grid-search over)
        experiments = list(itertools.product(*VARIATIONS_LISTS))

        # Checks which hyperparameter are actually varied
        varied_params = [k for k in VARIATIONS.keys() if len(VARIATIONS[k]) > 1]

        print('\nPreparing a GRID search')
        print(f"Variations:")
        for key in VARIATIONS.keys(): print(f"    {key}: {VARIATIONS[key]}")

    # RANDOM SEARCH
    elif serial_args.search_type == 'random':
        # Samples all experiments' hyperparameters
        from scripts.schedule_random import sample_experiment, SEED_VARIATIONS
        experiments = [sample_experiment() for _ in range(serial_args.n_experiments)]

        # Makes sure no param was included twice in the schedule file
        param_names = [param_name for param_name, param_value in experiments[0]]
        assert all([param_names.count(param_names[i]) == 1 for i in range(len(param_names))]), \
            f'A hyperparam has been entered twice in the schedule file:\n param_names="{param_names}"'

        # Checks which hyperparameter are actually varied
        param_samples = {param_name:[] for param_name, param_value in experiments[0]}
        for experiment in experiments:
            for param_tuple in experiment:
                param_name, param_value = param_tuple
                param_samples[param_name].append(param_value)

        non_varied_params = []
        for param_name in param_samples.keys():
            if all([x == param_samples[param_name][0] for x in param_samples[param_name]]):
                non_varied_params.append(param_name)

        for param_name in non_varied_params:
            del param_samples[param_name]
        varied_params = param_samples.keys()

        print(f'\nPreparing a RANDOM search over {serial_args.n_experiments} experiments and {len(SEED_VARIATIONS)} seeds')

    else:
        raise NotImplementedError

    # Printing info
    print(f"\nDefault {config_to_str(config)}")

    print(f"\n\nAbout to create {len(experiments)} experiment directories")
    print(f"    maoc git-hash: {git_hash.split('_')[0]}\n    environment git-hash: {git_hash.split('_')[1]}\n")
    answer = input("Should we proceed? [y or n]")
    if answer.lower() not in ['y', 'yes']:
        print("Aborting...")
        sys.exit()

    # For every experiment defined, creates an experiment directory to be retrieved and run later
    for param_list in experiments:

        # Modifies the config for this particular experiment
        for param_tuple in param_list:
            param_name, param_value = param_tuple
            if param_name not in config_dict.keys():
                raise ValueError(f"{param_name} from schedule.VARIATIONS is not a valid training hyperparameter")
            else:
                config_dict[param_name] = param_value

        experiment_num = int(DirectoryManager(storage_name=storage_dir, seed=1).experiment_dir.stem.strip('experiment'))

        for seed in SEED_VARIATIONS:
            config.seed = seed

            # Creates the experiment directory
            dir_manager = DirectoryManager(storage_name=storage_dir,
                                           experiment_num=experiment_num,
                                           seed=config.seed)
            dir_manager.create_directories()

            config_unique_dict = {k: v for k, v in param_list if k in varied_params}
            config_unique_dict['seed'] = seed

            # Saves the set of unique variations to a json file (to easily identify the uniqueness of this experiment)
            save_dict_to_json(config_unique_dict, filename=str(dir_manager.seed_dir / 'config_unique.json'))

            # Saves the config to a json file (for later use)
            save_config_to_json(config, filename=str(dir_manager.seed_dir / 'config.json'))

            # Creates empty file UNHATCHED meaning that the experiment is ready to be run
            open(str(dir_manager.seed_dir/'UNHATCHED'), 'w+').close()

    # Printing summary
    first_experiment_created = int(dir_manager.current_experiment.strip('experiment')) - len(experiments) + 1
    last_experiment_created = first_experiment_created + len(experiments) - 1

    if serial_args.search_type == 'grid':
        # Saves VARIATIONS in the directory
        key = f'{first_experiment_created}-{last_experiment_created}'
        VARIATIONS['seed'] = SEED_VARIATIONS  # reincorporates seed_variations
        if (dir_manager.storage_dir / 'variations.json').exists():
            variations_dict = load_dict_from_json(filename=str(dir_manager.storage_dir / 'variations.json'))
            assert key not in variations_dict.keys()
            variations_dict[key] = VARIATIONS
        else:
            variations_dict = {key: VARIATIONS}
        save_dict_to_json(variations_dict, filename=str(dir_manager.storage_dir / 'variations.json'))
        open(str(dir_manager.storage_dir / 'GRID_SEARCH'), 'w+').close()

    elif serial_args.search_type == 'random':
        fig, axes = plt.subplots(len(param_samples), 1, figsize=(6, 2 * len(param_samples)))
        plot_sampled_hyperparams(axes, param_samples)
        fig.savefig(str(dir_manager.storage_dir / 'variations.png'))
        plt.close(fig)
        open(str(dir_manager.storage_dir / 'RANDOM_SEARCH'), 'w+').close()

    print(f'\nDONE\nCreated directories '
          f'{str(dir_manager.storage_dir)}/experiment{first_experiment_created}-{last_experiment_created}')
    print(f"Each of these experiments contain directories for the following seeds: {SEED_VARIATIONS}")
