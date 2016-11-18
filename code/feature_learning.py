#!/usr/bin/env python

import os
import pandas
import matplotlib.pyplot as plt
import numpy
import math

from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras.regularizers import activity_l1, l2, activity_l2
from keras import backend as K
from keras import objectives

#######################################################################
# Loading data
#######################################################################

input_path = "../data-temp/labs_cohort_predict_518_584.csv"

# Since Spark saves the CSV data by partition, we must still find the
# single file in that path (we coalesced to one partition, but it's
# still not a known filename):
files = [d for d in os.listdir(input_path) if d.endswith(".csv")]
if len(files) != 1:
    raise Exception("Can't find single CSV file in %s" % (input_path,))

fname = os.path.join(input_path, files[0])
print("Training: Loading from %s..." % (fname,))
df = pandas.read_csv(fname)
gr = list(df.groupby((df["HADM_ID"], df["ITEMID"], df["VALUEUOM"])))
print("Training: Got %d points (%d admissions)." % (len(df), len(gr)))

# 'gr' then is a list of: ((HADM_ID, ITEMID, VALUEUOM), time-series dataframe)

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
patch_length = int(math.ceil(2 * padding / interval))

# So, assign a weight to each time-series based on how many patches
# are in it (in effect, make each patch equally likely):
num_patches = numpy.array([len(ts[1])-patch_length for ts in gr])
weights = num_patches / num_patches.sum()

# And then select 'patch_count' indices, each one for a particular
# time-series:
patch_count = len(gr) * 3
random_idx = numpy.random.choice(len(gr), patch_count, p = weights)

# Make an array to hold all this:
x_train = numpy.zeros((patch_count, patch_length * 2))
# Fill with (uniformly) random patches from the selected time-series:
for (i,ts_idx) in enumerate(random_idx):
    ts = gr[ts_idx][1]
    start_idx = numpy.random.randint(num_patches[ts_idx])
    end_idx = start_idx + patch_length
    # First half is mean:
    x_train[i, :patch_length] = ts["MEAN"].iloc[start_idx:end_idx]
    # Second half is variance:
    x_train[i, patch_length:] = ts["VARIANCE"].iloc[start_idx:end_idx]

#######################################################################
# Autoencoder
#######################################################################

# Size of hidden layers:
hidden1 = 100
hidden2 = 100

input_ts = Input(shape=(patch_length * 2,))
encoded1 = Dense(hidden1,
                 activation='sigmoid',
                 activity_regularizer=activity_l1(0.1),
                 W_regularizer=l2(0.01))(input_ts)
encoded2 = Dense(hidden2,
                 activation='sigmoid',
                 activity_regularizer=activity_l1(0.1),
                 W_regularizer=l2(0.01))(encoded1)

decoded1 = Dense(hidden1, activation='linear')(encoded2)
decoded2 = Dense(patch_length * 2, activation='linear')(decoded1)

autoencoder = Model(input=input_ts, output=decoded2)
autoencoder.compile(optimizer='adadelta', loss='mse')

autoencoder.fit(x_train, x_train,
                nb_epoch=50,
                batch_size=256,
                shuffle=True,
                #validation_data=(x_test, x_test)
                )
