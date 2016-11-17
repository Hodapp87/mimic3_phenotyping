#!/usr/bin/env python

import os
import pandas
import matplotlib.pyplot as plt
import numpy
import math

from keras.layers import Input, Dense
from keras.models import Model

input_path = "../data-temp/labs_cohort_predict_518_584.csv"

#######################################################################
# Loading data
#######################################################################

# Since Spark saves the CSV data by partition, we must still find the
# single file in that path (we coalesced to one partition, but it's
# still not a known filename):
files = [d for d in os.listdir(input_path) if d.endswith(".csv")]
if len(files) != 1:
    raise Exception("Can't find single CSV file in %s" % (input_path,))

fname = os.path.join(input_path, files[0])
print("Loading from %s..." % (fname,))
df = pandas.read_csv(fname)
gr = list(df.groupby((df["HADM_ID"], df["ITEMID"], df["VALUEUOM"])))
print("Got %d points (%d admissions)." % (len(df), len(gr)))

# 'gr' then is a list of: ((HADM_ID, ITEMID, VALUEUOM), time-series dataframe)

#######################################################################
# Gathering patches
#######################################################################

# Next, we need the actual number of contiguous patches in the
# time-series.  The length of these contiguous patches is set to twice
# the padding before and after in order to ensure every series can
# supply a patch.
padding = 7.5
interval = 0.5
minPatch = int(math.ceil(2 * padding / interval))

# So, assign a weight to each time-series based on how many patches
# are in it (in effect, make each patch equally likely):
numPatches = numpy.array([len(ts[1])-minPatch for ts in gr])
weights = numPatches / numPatches.sum()

# And then select 'patchCount' indices, each one for a particular
# time-series:
patchCount = len(gr) * 3
randomIdx = numpy.random.choice(len(gr), patchCount, p = weights)

# Make an array to hold all this (final dimension is for mean &
# variance):
xTrain = numpy.zeros((patchCount, minPatch, 2))
# Fill with (uniformly) random patches from the selected time-series:
for (i,tsIdx) in enumerate(randomIdx):
    ts = gr[tsIdx][1]
    startIdx = numpy.random.randint(numPatches[tsIdx])
    endIdx = startIdx + minPatch
    xTrain[i,:, 0] = ts["MEAN"].iloc[startIdx:endIdx]
    xTrain[i,:, 1] = ts["VARIANCE"].iloc[startIdx:endIdx]
