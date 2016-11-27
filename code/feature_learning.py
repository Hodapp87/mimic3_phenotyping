#!/usr/bin/env python3

# (c) 2016 Chris Hodapp, chodapp3@gatech.edu

import utils

import os
import pandas
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy
import math

from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras.regularizers import activity_l1, l2, activity_l2
from keras import backend as K
from keras import objectives

# Make sure pydot-ng is installed for the below
from keras.utils.visualize_util import plot

#######################################################################
# Loading data
#######################################################################

input_path = "../data-temp/labs_cohort_predict_518_584.csv"

fname = utils.get_single_csv(input_path)
print("Training: Loading from %s..." % (fname,))
df = pandas.read_csv(fname)
df_groups = df.groupby((df["HADM_ID"], df["ITEMID"], df["VALUEUOM"]))
gr = list(df_groups)
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

print("Sampled %d patches of data" % (len(x_data),))

#######################################################################
# Training/validation split
#######################################################################
# What ratio of the data to leave behind for validation
validation_ratio = 0.2
numpy.random.shuffle(x_data)
split_idx = int(patch_count * validation_ratio)
x_val, x_train = x_data[:split_idx,:], x_data[split_idx:,:]
print("Split r=%g: %d patches for training, %d for validation" %
      (validation_ratio, len(x_val), len(x_train)))

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

raw_input_tensor = Input(shape=ts_shape, name="raw_input")
encode1_layer = Dense(hidden1,
                      activation='sigmoid',
                      activity_regularizer=activity_l1(0.0001),
                      W_regularizer=l2(0.0001),
                      name="encode1")
encode1_tensor = encode1_layer(raw_input_tensor)

decode1_layer = Dense(output_dim=patch_length * 2,
                      activation='linear',
                      name="decode1")
decode1_tensor = decode1_layer(encode1_tensor)

# First model is input -> encode1 -> decode1:
autoencoder1 = Model(input=raw_input_tensor, output=decode1_tensor)
autoencoder1.compile(optimizer='adadelta', loss='mse')
plot(autoencoder1, to_file='keras_autoencoder1.png', show_shapes=True)

# Train first autoencoder on raw input.
autoencoder1.fit(x_train, x_train,
                 nb_epoch=600,
                 batch_size=256,
                 shuffle=True,
                 validation_data=(x_val, x_val))

# Stack the 2nd autoencoder (connecting to encode1):
encode2_layer = Dense(hidden2,
                      activation='sigmoid',
                      activity_regularizer=activity_l1(0.0001),
                      W_regularizer=l2(0.00001),
                      name="encode2")
encode2_tensor = encode2_layer(encode1_tensor)

decode2_layer = Dense(output_dim=patch_length * 2,
                      activation='linear',
                      name="decode2")
decode2_tensor = decode2_layer(encode2_tensor)

# Then we need raw_input -> encode1 -> encode2 -> decode2, but with
# some layers left out of training:
autoencoder2 = Model(input=raw_input_tensor, output=decode2_tensor)
encode1_layer.trainable = False
# Probably superfluous:
decode1_layer.trainable = False
autoencoder2.compile(optimizer='adadelta', loss='mse')
plot(autoencoder2, to_file='keras_autoencoder2.png', show_shapes=True)

# Train second autoencoder.  We're basically training it on primary
# hidden features, not raw input, because we're keeping the first
# encoder's weights constant (hence 'trainable = False' above) and so
# it is simply feeding encoded input in.
autoencoder2.fit(x_train, x_train,
                 nb_epoch=200,
                 batch_size=256,
                 shuffle=True,
                 validation_data=(x_val, x_val)
                 )

# Finally, fine-tune stacked autoencoder on raw inputs:
encode1_layer.trainable = True
sae = Model(input=raw_input_tensor, output=decode2_tensor)
sae.compile(optimizer='adadelta', loss='mse')
plot(sae, to_file='keras_sae.png', show_shapes=True)
sae.fit(x_train, x_train,
        nb_epoch=200,
        batch_size=256,
        shuffle=True,
        validation_data=(x_val, x_val))

# Then, here is our model which provides 2nd hidden layer features:
stacked_encoder = Model(input=raw_input_tensor, output=encode2_tensor)
plot(stacked_encoder, to_file='keras_stacked_encoder.png', show_shapes=True)

def plot_weights(weights, name, variances = None):
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
    # Why didn't I do this?
    #plt.savefig("%s.eps" % (name,), bbox_inches='tight')
    plt.savefig("%s.pdf" % (name,), bbox_inches='tight')
    plt.savefig("%s.png" % (name,), bbox_inches='tight')
    plt.tight_layout()
    plt.close()
    
# Get means from 1st-layer weights:
plot_weights(encode1_layer.get_weights()[0][:30,:],
             "keras_1st_layer",
             None)
             #encode1_layer.get_weights()[0][30:,:])
