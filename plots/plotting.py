import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from os import path
import argparse
import os
import sys
import utils
from utils.plots import *


def get_plotting_args(overwritten_args=None):

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument("--json_file", default='plots/mean_returns_switch.json',
                        help="the json file that specifies what experiments to plot")
    parser.add_argument("--document_name", default="return", help="the name of the created png file")
    parser.add_argument("--title", default='Average Returns', help='Title of the plot created')
    parser.add_argument("--xlabel", default="Frames", help="Label of the x-axis")
    parser.add_argument("--ylabel", default="reward", help="Label on the y-axis")

    return parser.parse_args(overwritten_args)

def check_experiment_length(xs):
    '''
    Function to check if the x-axis of all of the experiments are of the same length

    :param xs: list of list, where each sublist is the x-axis of that experiment
    :return: False if they don't match. If they do match just return the common x-axis for all of the experiments (assuming same length means they have same elements)
    '''

    first_length = len(xs[0])
    for x in xs:
        if len(x) != first_length:
            print("The length of the experiments don't match. One experiment has length {0} and another has length {1}.".format(first_length,len(x)))
            return False

    return xs[0]

def match_length(ys, std_results, xs):
    '''
    Function to match the length of the different experiments. Idea is to loop through x-axis of smallest and match these elements with the x-axis of the others.
    This deletes the elements that aren't common to both for the ys, std_dev and xs.

    :param ys: List of list. Each sublist is the ys of that experiment
    :param std_results: List of list. Each sublist is the std_devs of that experiment
    :param xs: List of list. Each sublist is the x-axis of that experiment
    :return: modified ys, std, xs such that they have the same length
    '''

    # Find Smallest
    min_length = sys.maxsize
    min_index = -1
    for i in range(len(xs)):
        if min_length > len(xs[i]):
            min_length = len(xs[i])
            min_index = i

    # Loop through all others, and remove extra elements
    for indx, x in enumerate(xs):

        if indx != min_index:

            counter = 0 # keep track of where we are at for the x[min_index]
            x_to_remove = [] # list of xs that are extra, will be removed
            y_indx_to_remove = [] # list of index positions for y's to be removed
            for x_indx, element in enumerate(x):

                # element of x not equal to x of min
                if (element != xs[min_index][counter]):

                    # keep track of element to be removed from x-axis (this assumes each entry is unique there append elements and not index)
                    x_to_remove.append(element)
                    # keep track of index to be removed from ys
                    y_indx_to_remove.append(x_indx)

                else:
                    counter += 1

            # remove all elements that where in the list of elements to be removed
            xs[indx] = [e for e in x if e not in x_to_remove]

            # remove extra ys
            ys[indx] = np.delete(ys[indx], y_indx_to_remove)
            ys[indx] = np.reshape(ys[indx], (ys[indx].shape[0], 1)) #reshape to be compatible

            # same thing with std dev
            std_results[indx] = np.delete(std_results[indx], y_indx_to_remove)
            std_results[indx] = np.reshape(std_results[indx], (std_results[indx].shape[0], 1))

    # at the end make sure x[min] isn't too big, delete all the ones that were never met.
    if counter < len(xs[min_index]):
        # pdb.set_trace()
        extras = []
        y_extras = []
        for i in range(counter, len(xs[min_index])):
            extras.append(xs[min_index][i])
            y_extras.append(i)

        xs[min_index] = [w for w in xs[min_index] if w not in extras]
        ys[min_index] = np.delete(ys[min_index], y_extras)
        ys[min_index] = np.reshape(ys[min_index], (ys[min_index].shape[0], 1))

        std_results[min_index] = np.delete(std_results[min_index], y_extras)
        std_results[min_index] = np.reshape(std_results[min_index], (std_results[min_index].shape[0], 1))

    return ys, std_results, xs

def plot(config, metric = "mean_agent_return_with_broadcast_penalties_mean"):

    '''
    Function to plot results on one plot given a json file specifing the directories and names we would like to use

    :param json_file: Json file of experiment directories and names
    :param home_directory:  Where we are running the plot script from, this modifies the path to storage directory (We assume everything is being stored in "storage" directory
    :param x_label: self explanatory
    :param metric: metric used for the plots compatible with the data inside the pickle file of results
    :return: saves plot
    '''
    json_file = config.json_file
    home_directory = "."
    x_label = config.xlabel

    # load json as dict
    loaded_dict = utils.load_dict_from_json(json_file)

    # get experiments and names
    experiments = loaded_dict['experiment_folders']
    names = loaded_dict['names']

    # make sure they are the same length
    if len(experiments) != len(names):
        print("Error in number of arguments for experiments or names, make sure there are the same number of experiment folders as there are names")
        exit(1)

    # create file path to the experiment directories
    dir = Path(home_directory+'/storage')

    # keep track of xs, mean_results and std dev to be plotted
    xs = []
    mean_results = []
    std_results = []
    # For each experiment, loops through all seeds as aggregate info
    for exp in experiments:

        exp_results = []
        exp_xs = []

        seeds_dir = os.listdir(dir/exp)
        for seed in seeds_dir:
            path = os.path.join(dir/exp, seed)
            data = utils.load_graph_data(path)[metric]
            frames = utils.load_graph_data(path)['num_frames']
            exp_xs.append(frames)
            exp_results.append(data)

        # make sure each run has same length
        exp_xs = check_experiment_length(exp_xs)
        xs.append(exp_xs)

        mean_exp_results = np.mean(exp_results, axis=0)
        std_exp_results = np.std(exp_results, axis=0)

        mean_results.append(mean_exp_results)
        std_results.append(std_exp_results)


    if (check_experiment_length(xs) == False):
        mean_results, std_results, xs = match_length(mean_results, std_results, xs)

    ys = np.concatenate(mean_results, axis= 1).T
    stds = np.concatenate(std_results, axis=1).T

    #plot everything
    fig, ax = create_fig((1, 1))
    plot_curve(ax, xs,
               ys,
               stds=stds,
               labels=names,
               xlabel=x_label, title=config.title)
    fig.savefig(config.document_name)
    plt.close(fig)


if __name__=="__main__":

    #home_directory = "."
    #json_file = "plots/mean_returns_switch_ppo.json"

    config = get_plotting_args()

    plot(config)

