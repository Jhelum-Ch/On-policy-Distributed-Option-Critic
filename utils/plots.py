from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
import numpy as np

def plot_curve(ax, x, ys, colors=None, labels=None, xlabel="", ylabel="", title="", stds=None):
    if colors is None:
        cm = plt.cm.get_cmap('viridis')
        colors = [np.array(cm(float(i) / float(len(ys)))[:3]) for i in range(len(ys))]

    if labels is None:
        labels = [f'curve {i}' for i in range(len(ys))]

    # Plots losses and smoothed losses for every agent
    for i, y in enumerate(ys):
        if stds is None:
            ax.plot(x, y, color=colors[i], alpha=0.3)
            ax.plot(x, smooth(y), color=colors[i], label=labels[i])
        else:
            ax.plot(x, y, color=colors[i], label=labels[i])
            ax.fill_between(x, y-stds[i], y+stds[i], color=colors[i], alpha=0.1)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc='lower right')

def plot_sampled_hyperparams(ax, param_samples):
    cm = plt.cm.get_cmap('viridis')
    for i, param in enumerate(param_samples.keys()):
        args = param_samples[param], np.zeros_like(param_samples[param])
        kwargs = {'linestyle':'', 'marker':'o', 'label':param, 'alpha':0.2, 'color':cm(float(i)/float(len(param_samples)))}
        if param == 'lr':
            ax[i].semilogx(*args, **kwargs)
        else:
            ax[i].plot(*args, **kwargs)
            ax[i].xaxis.set_major_locator(MaxNLocator(integer=True))

        ax[i].get_yaxis().set_ticks([])
        ax[i].legend(loc='upper right')

def smooth(data_serie, smooth_factor=0.8):
    assert smooth_factor > 0. and smooth_factor < 1.
    mean = data_serie[0]
    new_serie = []
    for value in data_serie:
        mean = smooth_factor * mean + (1 - smooth_factor) * value
        new_serie.append(mean)

    return new_serie

def create_fig(axes_shape):
    figsize = (8 * axes_shape[1], 5 * axes_shape[0])
    fig, axes = plt.subplots(axes_shape[0], axes_shape[1], figsize=figsize)
    return fig, axes
