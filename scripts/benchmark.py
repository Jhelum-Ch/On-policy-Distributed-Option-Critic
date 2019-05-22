import sys
sys.path.append('..')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils.directory_tree import DirectoryManager
from utils.plots import plot_curve, create_fig
from utils.config import load_dict_from_json, save_dict_to_json
from utils.save import load_graph_data, create_logger
from collections import OrderedDict
import logging
import seaborn as sns
sns.set()

AGENT_I = 0
STORAGE_DIRS = [
    # "6c23331_3e9171e_A2C",
    "6c23331_3e9171e_OC",
    "6c23331_3e9171e_DOC",
    # "6c23331_3e9171e_PPO",
]


def compare_models_by_learning_curve(storage_dirs, agent_i, num_key=False, n_labels=None, logger=None):
    if num_key:
        assert len(storage_dirs) == 1

    if logger is None:
        logger = create_logger("", logging.INFO, logfile=None, streamHandle=True)

    logger.info(f'\n{"benchmark_learning".upper()}:')

    curves_mean = OrderedDict()
    eval_curves_means = OrderedDict()
    eval_curves_stds = OrderedDict()
    num_frames = OrderedDict()

    for storage_dir in storage_dirs:

        # Get all experiment directories
        all_experiments = DirectoryManager.get_all_experiments(storage_dir=DirectoryManager.root / storage_dir)

        for experiment_dir in all_experiments:

            # For that experiment, get all seed directories
            experiment_seeds = DirectoryManager.get_all_seeds(experiment_dir=experiment_dir)

            for i, seed_dir in enumerate(experiment_seeds):

                logger.info(f"{seed_dir}")
                config_dict = load_dict_from_json(str(seed_dir/"config.json"))
                env = config_dict["env"]
                experiment_num = seed_dir.parent.stem.strip('experiment')

                graph_data = load_graph_data(save_dir=seed_dir)

                if num_key:
                    key = f"exp{experiment_num}"
                else:
                    key = storage_dir

                if env not in curves_mean.keys():
                    curves_mean[env] = OrderedDict()
                    num_frames[env] = OrderedDict()
                    eval_curves_means[env] = []
                    eval_curves_stds[env] = []
                if key not in curves_mean[env].keys():
                    curves_mean[env][key] = []
                    num_frames[env][key] = []

                curves_mean[env][key].append(np.array(graph_data['return_mean']).T[agent_i])
                num_frames[env][key] = graph_data['num_frames']

    if len(curves_mean.keys()) > 1:
        axes_shape = (2, int(np.ceil(len(curves_mean.keys())) / 2.))
    else:
        axes_shape = (1, 1)
    fig, axes = create_fig(axes_shape)

    for i, env in enumerate(curves_mean.keys()):

        labels = []
        for j, key in enumerate(curves_mean[env].keys()):

            eval_curves_means[env].append(np.stack(curves_mean[env][key]).mean(axis=0))
            eval_curves_stds[env].append(np.stack(curves_mean[env][key]).std(axis=0))
            labels.append(key.split("_")[-1])

        if n_labels is not None:
            curves_means = np.array([array.mean() for array in eval_curves_means[env]])
            n_max_idxs = (-curves_means).argsort()[:n_labels]

            for i in range(len(labels)):
                if i in n_max_idxs:
                    continue
                else:
                    labels[i] = None

        if axes_shape == (1, 1):
            current_ax = axes
        elif any(np.array(axes_shape) == 1):
            current_ax = axes[i]
        else:
            current_ax = axes[i // axes_shape[1], i % axes_shape[1]]

        plot_curve(current_ax,
                    xs=num_frames[env].values(),
                    ys=eval_curves_means[env],
                    stds=eval_curves_stds[env],
                    labels=labels,
                    xlabel="Frames", title=env)

    for storage_dir in storage_dirs:
        fig.savefig(DirectoryManager.root / storage_dir / 'benchmark_learning.png')
        save_dict_to_json(storage_dirs, DirectoryManager.root / storage_dir / 'benchmark_learning_sources.json')

    plt.close(fig)

if __name__ == '__main__':
    logger = create_logger(name="", loglevel=logging.INFO)
    compare_models_by_learning_curve(storage_dirs=STORAGE_DIRS, agent_i=AGENT_I, logger=logger)
