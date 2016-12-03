#!/usr/bin/env python

# (c) 2016 Chris Hodapp, chodapp3@gatech.edu
# timeseries_plots.py: Producing plots of original, warped, and
# GPR-interpolated time-series data from a prior run of the Spark
# code.

import utils

import argparse

import pandas
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy

def main():
    parser = argparse.ArgumentParser(
        description="Plot time-series in its original, warped, and GPR interpolated form")
    parser.add_argument("-d", "--data_dir", required=True,
                        help="Input directory for data files from Spark")
    parser.add_argument("-o", "--output_dir", required=True,
                        help="Output directory for plots and models")
    parser.add_argument("--icd9a", required=True,
                        help="First ICD9 code (for locating saved data)")
    parser.add_argument("--icd9b", required=True,
                        help="Second ICD9 code (for locating saved data)")
    parser.add_argument("-l", "--loinc", required=True,
                        help="LOINC code (for locating saved data)")
    parser.add_argument("-g", "--grid", default=(4,4), nargs=2,
                        help="Number of plots horizontally and vertically (default 4 4)")
    args = parser.parse_args()

    nx, ny = args.grid
    # TODO: Set this from date or something?
    numpy.random.seed(0x123456)
    
    #######################################################################
    # Loading data
    #######################################################################
    
    suffix = "cohort_%s_%s_%s" % (args.icd9a, args.icd9b, args.loinc)
    csvname = "%s/%s_train.csv" % (args.data_dir, suffix)

    print("Trying to load: %s" % (csvname,))
    ts_raw = pandas.read_csv(utils.get_single_csv(csvname))
    ts_raw.fillna("", inplace = True)
    ts_raw_groups = ts_raw.groupby((ts_raw["HADM_ID"], ts_raw["ITEMID"], ts_raw["VALUEUOM"]))

    csvname = "%s/%s_predict.csv" % (args.data_dir, suffix)

    print("Trying to load: %s" % (csvname,))
    ts_gpr = pandas.read_csv(utils.get_single_csv(csvname))
    ts_gpr.fillna("", inplace = True)
    ts_gpr_groups = ts_gpr.groupby((ts_gpr["HADM_ID"], ts_gpr["ITEMID"], ts_gpr["VALUEUOM"]))

    #######################################################################
    # Plotting
    #######################################################################
    fig = plt.figure() #figsize = (40,40))
    # Why is there a 0-1 axis on both X and Y?
    plt.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')
    plt.tick_params(axis='y', which='both', left='off', right='off', labelleft='off')
    gpr_plot_grid(fig, ts_raw_groups, ts_gpr_groups, nx, ny)
    pngname = "%s/%s_timeseries.png" % (args.output_dir, suffix)
    epsname = "%s/%s_timeseries.eps" % (args.output_dir, suffix)
    print("Saving %s..." % (pngname,))
    plt.savefig(pngname, bbox_inches='tight')
    print("Saving %s..." % (epsname,))
    plt.savefig(epsname, bbox_inches='tight')
    plt.close()

    # Just pick something:
    l = list(ts_raw_groups.groups.keys())
    l.sort()
    idx = numpy.random.randint(len(l))
    group_id = l[idx]
    gpr_plot(ts_raw_groups.get_group(group_id), ts_gpr_groups.get_group(group_id))
    plt.title(str(group_id))
    pngname = "%s/%s_timeseries_single.png" % (args.output_dir, suffix)
    epsname = "%s/%s_timeseries_single.eps" % (args.output_dir, suffix)
    print("Saving %s..." % (pngname,))
    plt.savefig(pngname, bbox_inches='tight')
    print("Saving %s..." % (epsname,))
    plt.savefig(epsname, bbox_inches='tight')
    plt.close()
    
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
    plt.xlabel("")
    plt.ylabel("")
    plt.tight_layout()

if __name__ == "__main__":
    main()
