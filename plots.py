#!/usr/bin/env python

# (c) 2016 Chris Hodapp, chodapp3@gatech.edu

import utils

import os
import pandas
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy
import math


def gpr_plot(raw_subdf, gpr_subdf, ax = None, labels = True, legend = True):
    """Plot the results from a "raw" time-series sub-dataframe, and from a
    Gaussian Process Regression interpolated sub-dataframe.  They are
    assumed to be from the same group, but are not checked.
    """
    ax1 = raw_subdf.plot.line(x = "CHARTTIME", y = "VALUENUM", ax = ax, label = "Raw", color = "Black", alpha=0.7)
    ax2 = raw_subdf.plot.line(x = "CHARTTIME_warped", y = "VALUENUM", ax=ax1, label = "Warped", color = "Red")
    # Add error bars to GPR:
    gpr_subdf2 = gpr_subdf.copy()
    gpr_subdf2["MEAN+"] = gpr_subdf2["MEAN"] + numpy.sqrt(gpr_subdf2["VARIANCE"])
    gpr_subdf2["MEAN-"] = gpr_subdf2["MEAN"] - numpy.sqrt(gpr_subdf2["VARIANCE"])
    ax3 = gpr_subdf2.plot.line(x = "CHARTTIME_warped", y = "MEAN", ax=ax2, label = "GPR", color = "Blue")
    # This should work, but for whatever reason probably related to
    # the train-wreckedness of Pandas and matplotlib plotting does
    # not:
    #ax3.fill_between(gpr_subdf2.index,
    #                 gpr_subdf2["MEAN-"].values,
    #                 gpr_subdf2["MEAN+"].values
    #                 color = "Blue",
    #                 alpha = 0.2)
    #
    # So I'm doing this instead:
    ax4 = gpr_subdf2.plot.line(x = "CHARTTIME_warped", y = "MEAN+", ax=ax3, color = "Blue", alpha = 0.4, style = "--")
    ax5 = gpr_subdf2.plot.line(x = "CHARTTIME_warped", y = "MEAN-", ax=ax4, color = "Blue", alpha = 0.4, style = "--")
    if not legend:
        ax1.legend_.remove()
    else:
        # and this hackish crap:
        lines, labels = ax1.get_legend_handles_labels()
        ax1.legend(lines[:3], labels[:3], loc='best')
    if labels:
        plt.xlabel("Relative time (days)")
        # TODO: Get LOINC code maybe?
        plt.ylabel("Value (%s)" % (raw_subdf["VALUEUOM"].iloc[0],))
    else:
        plt.xlabel("")
        plt.ylabel("")
        plt.setp(ax1.get_xticklabels(), visible=False)
        plt.setp(ax1.get_yticklabels(), visible=False)
    return ax5

def gpr_plot_grid(fig, raw_groups, gpr_groups, rows = 8, cols = 8, random = True):
    groups = list(raw_groups.groups.keys())
    groups.sort()
    outer_grid = gridspec.GridSpec(rows, cols, wspace=0.0, hspace=0.0)
    for i in range(rows * cols):
        ax = plt.Subplot(fig, outer_grid[i])
        if random:
            group_id = groups[numpy.random.randint(len(groups))]
        else:
            group_id = groups[i]
        try:
            ax2 = gpr_plot(raw_groups.get_group(group_id), gpr_groups.get_group(group_id), ax, False, False)
        except KeyError:
            print("Can't find %s" % (group_id,))
        except IndexError:
            print("IndexError on %s?" % (group_id,))
        ax.set_xticks([])
        ax.set_yticks([])
        fig.add_subplot(ax)
    plt.tight_layout()

def plots():
    numpy.random.seed(0x123456)
    
    #######################################################################
    # Loading data
    #######################################################################

    #suffix = "276_427_50820"
    data_dir = "./data/"
    #suffix = "cohort_518_584_11558-4"
    suffix = "cohort_428_584_1742-6"

    ts_raw = pandas.read_csv(
        utils.get_single_csv("%s/%s_train.csv" % (data_dir, suffix)))
    ts_raw.fillna("", inplace = True)
    ts_raw_groups = ts_raw.groupby((ts_raw["HADM_ID"], ts_raw["ITEMID"], ts_raw["VALUEUOM"]))

    ts_gpr = pandas.read_csv(
        utils.get_single_csv("%s/%s_predict.csv" % (data_dir, suffix)))
    ts_gpr.fillna("", inplace = True)
    ts_gpr_groups = ts_gpr.groupby((ts_gpr["HADM_ID"], ts_gpr["ITEMID"], ts_gpr["VALUEUOM"]))

    #######################################################################
    # Plotting
    #######################################################################
    fig = plt.figure(figsize = (40,40))
    gpr_plot_grid(fig, ts_raw_groups, ts_gpr_groups, 8, 8)
    plt.savefig("%s/%s_timeseries.png" % (data_dir, suffix), bbox_inches='tight')
    plt.savefig("%s/%s_timeseries.eps" % (data_dir, suffix), bbox_inches='tight')
    plt.close()

    # Just pick something:
    l = list(ts_raw_groups.groups.keys())
    l.sort()
    idx = numpy.random.randint(len(l))
    print(idx)
    group_id = l[idx]
    gpr_plot(ts_raw_groups.get_group(group_id), ts_gpr_groups.get_group(group_id))
    plt.title(str(group_id))
    plt.savefig("%s/%s_timeseries_single.png" % (data_dir, suffix), bbox_inches='tight')
    plt.savefig("%s/%s_timeseries_single.eps" % (data_dir, suffix), bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    plots()
