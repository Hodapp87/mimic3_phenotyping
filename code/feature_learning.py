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

# Standardize input to mean 0, variance 1 (to confuse matters a
# little, the data that we're standardizing is itself mean and
# variance of the Gaussian process)
df["MEAN"]     = df["MEAN"]     - df["MEAN"].mean()
df["VARIANCE"] = df["VARIANCE"] - df["VARIANCE"].mean()
df["MEAN"]     = df["MEAN"]     / df["MEAN"].std()
df["VARIANCE"] = df["VARIANCE"] / df["VARIANCE"].std()

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
x_data = numpy.zeros((patch_count, patch_length * 2))
# Ordinarily I would put mean and variance in a 3rd dimension, but
# Keras doesn't seem to actually allow multidimensional inputs in
# 'Dense' despite saying that it does. Whatever.

# Fill with (uniformly) random patches from the selected time-series:
for (i,ts_idx) in enumerate(random_idx):
    ts = gr[ts_idx][1]
    start_idx = numpy.random.randint(num_patches[ts_idx])
    end_idx = start_idx + patch_length
    # First half is mean:
    x_data[i, :patch_length] = ts["MEAN"].iloc[start_idx:end_idx]
    # Second half is variance:
    x_data[i, patch_length:] = ts["VARIANCE"].iloc[start_idx:end_idx]

#######################################################################
# Training/validation split
#######################################################################
# What ratio of the data to leave behind for validation
validation_ratio = 0.4
numpy.random.shuffle(x_data)
split_idx = int(patch_count * validation_ratio)
x_val, x_train = x_data[:split_idx,:], x_data[split_idx:,:]

# TODO: Get labels

#######################################################################
# Stacked Autoencoder
#######################################################################

# Size of hidden layers:
hidden1 = 100
hidden2 = 100

# Input is size of one patch, with another dimension of size 2 (for
# mean & variance)
ts_shape = (patch_length * 2,)

# Make the first autoencoder and train it:
input_ts = Input(shape=ts_shape)
encoded = Dense(hidden1,
                activation='sigmoid',
                activity_regularizer=activity_l1(0.0001),
                W_regularizer=l2(0.0001)
                )(input_ts)
decoded = Dense(output_dim=patch_length * 2,
                activation='linear')(encoded)

autoencoder = Model(input=input_ts, output=decoded)
autoencoder.compile(optimizer='adadelta', loss='mse')

autoencoder.fit(x_train, x_train,
                nb_epoch=150,
                batch_size=256,
                shuffle=True,
                validation_data=(x_val, x_val)
                )

# Learn primary features (from hidden layer of autoencoder):
encoder1 = Model(input=input_ts, output=encoded)
hidden_features_train = encoder1.predict(x_train)
# Is this right?
hidden_features_val = encoder1.predict(x_val)

# Make the 2nd autoencoder:
input2 = Input(shape=(hidden1,))
encoded2 = Dense(hidden2,
                 activation='sigmoid',
                 activity_regularizer=activity_l1(0.0001),
                 W_regularizer=l2(0.0001))(input2)
decoded2 = Dense(output_dim=patch_length * 2,
                 activation='linear')(encoded2)

autoencoder2 = Model(input=input2, output=decoded2)
autoencoder2.compile(optimizer='adadelta', loss='mse')

# Now, for input of the primary features (not raw data), it should
# produce the training data:
autoencoder2.fit(hidden_features_train, x_train,
                 nb_epoch=100,
                 batch_size=256,
                 shuffle=True,
                 validation_data=(hidden_features_val, x_val)
                 )

# TODO: Fine-tune to train entire network?
