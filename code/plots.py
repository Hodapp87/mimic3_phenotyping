#!/usr/bin/env python3

# (c) 2016 Chris Hodapp, chodapp3@gatech.edu

import utils

import os
import pandas
import matplotlib.pyplot as plt
import numpy
import math


#######################################################################
# Loading data
#######################################################################

ts_raw = pandas.read_csv(
    utils.get_single_csv("../data-temp/labs_cohort_train_518_584.csv"))
ts_raw_groups = ts_raw.groupby((ts_raw["HADM_ID"], ts_raw["ITEMID"], ts_raw["VALUEUOM"]))

ts_gpr = pandas.read_csv(
    utils.get_single_csv("../data-temp/labs_cohort_predict_518_584.csv"))
ts_gpr_groups = ts_gpr.groupby((ts_gpr["HADM_ID"], ts_gpr["ITEMID"], ts_gpr["VALUEUOM"]))

def gpr_plot(raw_subdf, gpr_subdf):
    """Plot the results from a "raw" time-series sub-dataframe, and from a
    Gaussian Process Regression interpolated sub-dataframe.  They are
    assumed to be from the same group, but are not checked.
    """
    ax1 = raw_subdf.plot.line(x = "CHARTTIME", y = "VALUENUM", label = "Raw", color = "Black")
    ax2 = raw_subdf.plot.line(x = "CHARTTIME_warped", y = "VALUENUM", ax=ax1, label = "Warped", color = "Red")
    plt.xlabel("Relative time (days)")
    # TODO: Get LOINC code maybe?
    plt.ylabel("Value (%s)" % (raw_subdf["VALUEUOM"].iloc[0],))
    ax3 = gpr_subdf.plot.line(x = "CHARTTIME_warped", y = "MEAN", ax=ax2, label = "GPR", color = "Blue")
    # Add error bars to GPR:
    gpr_subdf2 = gpr_subdf.copy()
    gpr_subdf2["MEAN+"] = gpr_subdf2["MEAN"] + numpy.sqrt(gpr_subdf2["VARIANCE"])
    gpr_subdf2["MEAN-"] = gpr_subdf2["MEAN"] - numpy.sqrt(gpr_subdf2["VARIANCE"])
    ax4 = gpr_subdf2.plot.line(x = "CHARTTIME_warped", y = "MEAN+", ax=ax3, color = "Blue", style = "--")
    ax5 = gpr_subdf2.plot.line(x = "CHARTTIME_warped", y = "MEAN-", ax=ax4, color = "Blue", style = "--")
    return ax5

# Just pick something:
group_id = list(ts_raw_groups.groups.keys())[100]
plt.close()
gpr_plot(ts_raw_groups.get_group(group_id), ts_gpr_groups.get_group(group_id))
plt.show()
