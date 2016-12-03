#!/usr/bin/env python3

# (c) 2016 Chris Hodapp, chodapp3@gatech.edu
# utils.py: Utility functions used elsewhere

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

def sample_patches(groups, patch_multiple, patch_length, labelsDict):
    # The number of contiguous patches in the time-series in 'groups'
    # depends on the desired patch length.  So, assign a weight to
    # each time-series based on how many patches are in it (in effect,
    # make each patch equally likely):
    num_patches = numpy.array([len(ts[1])-patch_length for ts in groups])
    weights = num_patches / float(num_patches.sum())

    # And then select 'patch_count' indices, each one for a particular
    # time-series:
    patch_count = len(groups) * patch_multiple
    random_idx = numpy.random.choice(len(groups), patch_count, p = weights)

    # Make an array to hold all this:
    x_data = numpy.zeros((patch_count, patch_length * 2))
    # Ordinarily I would put mean and variance in a 3rd dimension, but
    # Keras doesn't seem to actually allow multidimensional inputs in
    # 'Dense' despite saying that it does. Whatever.

    # x_labels has the respective labels for the rows in x_data:
    x_labels = []

    # Fill with (uniformly) random patches from the selected time-series:
    for (i, ts_idx) in enumerate(random_idx):
        # ts_idx is an index of a group (i.e. time-series) itself:
        ts = groups[ts_idx][1]
        # Within this group we also need to pick the patch:
        start_idx = numpy.random.randint(num_patches[ts_idx])
        end_idx = start_idx + patch_length
        # First half is mean:
        x_data[i, :patch_length] = ts["MEAN"].iloc[start_idx:end_idx]
        # Second half is variance:
        x_data[i, patch_length:] = ts["VARIANCE"].iloc[start_idx:end_idx]
        # and assign the respective label:
        hadm, _, _ = groups[ts_idx][0]
        x_labels.append(labelsDict[hadm])
    return (x_data, x_labels)

def standardize(series):
    """Convert the given series or NumPy array to mean 0 and standard
    deviation 1."""
    series = series - series.mean()
    series_std = series.std()
    if (series_std > 1e-20):
        return series / series_std
    else:
        return series
