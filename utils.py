#!/usr/bin/env python3

# (c) 2016 Chris Hodapp, chodapp3@gatech.edu

import os

import numpy
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def get_single_csv(dirname):
    # Since Spark saves the CSV data by partition, we must still find the
    # single file in that path (we coalesced to one partition, but it's
    # still not a known filename):
    files = [d for d in os.listdir(dirname) if d.lower().endswith(".csv")]
    if len(files) != 1:
        raise Exception("Can't find single CSV file in %s" % (dirname,))
    fname = os.path.join(dirname, files[0])
    return fname


def plot_weights(weights, variances = None):
    if variances is not None:
        stds = numpy.sqrt(variances)
    fig = plt.figure(figsize = (20,20))
    outer_grid = gridspec.GridSpec(10, 10, wspace=0.0, hspace=0.0)
    for i in range(100):
        ax = plt.Subplot(fig, outer_grid[i])
        ax.plot(numpy.arange(0, weights.shape[0]), weights[:,i])
        if variances is not None:
            ax.fill_between(numpy.arange(0, weights.shape[0]),
                            weights[:,i] - stds[:,i],
                            weights[:,i] + stds[:,i],
                            color = "blue",
                            alpha = 0.2)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.add_subplot(ax)
    plt.tight_layout()
