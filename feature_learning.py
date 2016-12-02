#!/usr/bin/env python

# (c) 2016 Chris Hodapp, chodapp3@gatech.edu

import utils

import argparse
import pandas
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy
import math

import sklearn.manifold
import sklearn.preprocessing

#######################################################################
# Argument parsing
#######################################################################

parser = argparse.ArgumentParser(
    description="Perform unsupervised feature learning on processed data")
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
parser.add_argument("-n", "--net_plot",
                    help="Plot neural network structure (requires pydot-ng)",
                    action="store_true")
parser.add_argument("-t", "--tsne",
                    help="Perform t-SNE on learned features and produce plots",
                    action="store_true")
parser.add_argument("-w", "--weight_l2",
                    help="Set L2 weight regularization (default 0.0003)",
                    default=0.0003,
                    action="store_true")
parser.add_argument("-a", "--activity_l1",
                    help="Set L1 activity regularization (default 0.00003)",
                    default=0.00003,
                    action="store_true")
parser.add_argument("-p", "--patch_length",
                    help="Set patch length for neural network training",
                    default=20,
                    action="store_true")

args = parser.parse_args()
print(args)

# These can be slower, so do them after argument parsing:
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras.regularizers import activity_l1, l2, activity_l2
from keras import backend as K
from keras import objectives

# Make sure pydot-ng is installed for the below
try:
    from keras.utils.visualize_util import plot
except:
    print("Couldn't import keras.utils.visualize_util")

#######################################################################
# Loading data
#######################################################################

suffix = "cohort_%s_%s_%s" % (args.icd9a, args.icd9b, args.loinc)
csvname = "%s/%s_predict.csv" % (args.data_dir, suffix)
print("Trying to load: %s" % (csvname,))

# Load interpolated time-series, and group together (they're in
# flattened form in the CSV):
df = pandas.read_csv(utils.get_single_csv(csvname))
df.fillna("", inplace = True)

# 'gr' then is a list of: ((HADM_ID, ITEMID, VALUEUOM), time-series dataframe)

# Standardize input to mean 0, variance 1 (to confuse matters a
# little, the data that we're standardizing is itself mean and
# variance of the Gaussian process)
df["MEAN"]     = df["MEAN"]     - df["MEAN"].mean()
df["VARIANCE"] = df["VARIANCE"] - df["VARIANCE"].mean()
# Checking for near-zero standard deviation is probably pointless
# since the data will likely be useless, but I do it anyhow
mean_std = df["MEAN"].std()
if (mean_std > 1e-20):
    df["MEAN"] = df["MEAN"] / mean_std
var_std = df["VARIANCE"].std()
if (var_std > 1e-20):
    df["VARIANCE"] = df["VARIANCE"] / var_std

df_groups = df.groupby((df["HADM_ID"], df["ITEMID"], df["VALUEUOM"]))
gr = list(df_groups)
print("Got %d points (%d admissions)." % (len(df), len(gr)))

# 'gr' then is a list of: ((HADM_ID, ITEMID, VALUEUOM), time-series dataframe)

csvname = "%s/%s_predict.csv" % (args.data_dir, suffix)
print("Trying to load: %s" % (csvname,))

# Also load the labels; the CSV has (HADM_ID, ICD-9 category):
labels_df = pandas.read_csv(utils.get_single_csv(csvname))
# Turn it to a dictionary with HADM_ID -> category:
labels = dict(zip(labels_df["HADM_ID"], labels_df["ICD9_CATEGORY"]))

#######################################################################
# Gathering patches
#######################################################################

# Next, we need the actual number of contiguous patches in the
# time-series, which depends on the desired patch length:
patch_length = args.patch_length

# So, assign a weight to each time-series based on how many patches
# are in it (in effect, make each patch equally likely):
num_patches = numpy.array([len(ts[1])-patch_length for ts in gr])
weights = num_patches / float(num_patches.sum())

# And then select 'patch_count' indices, each one for a particular
# time-series:
patch_count = len(gr) * 3
random_idx = numpy.random.choice(len(gr), patch_count, p = weights)

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
    ts = gr[ts_idx][1]
    # Within this group we also need to pick the patch:
    start_idx = numpy.random.randint(num_patches[ts_idx])
    end_idx = start_idx + patch_length
    # First half is mean:
    x_data[i, :patch_length] = ts["MEAN"].iloc[start_idx:end_idx]
    # Second half is variance:
    x_data[i, patch_length:] = ts["VARIANCE"].iloc[start_idx:end_idx]
    # and assign the respective label:
    hadm, _, _ = gr[ts_idx][0]
    x_labels.append(labels[hadm])

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

# We don't split the labels because, intentionally, we never use them
# for feature learning; it's unsupervised learning.

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
                      activity_regularizer=activity_l1(args.activity_l1),
                      W_regularizer=l2(args.weight_l2),
                      name="encode1")
encode1_tensor = encode1_layer(raw_input_tensor)

decode1_layer = Dense(output_dim=patch_length * 2,
                      activation='linear',
                      name="decode1")
decode1_tensor = decode1_layer(encode1_tensor)

# First model is input -> encode1 -> decode1:
autoencoder1 = Model(input=raw_input_tensor, output=decode1_tensor)
autoencoder1.compile(optimizer='adadelta', loss='mse')
if args.net_plot:
    pngname = '%s/keras_autoencoder1.png' % (args.output_dir,)
    epsname = '%s/keras_autoencoder1.eps' % (args.output_dir,)
    print("Saving %s..." % (pngname,))
    plot(autoencoder1, to_file=pngname, show_shapes=True)
    print("Saving %s..." % (epsname,))
    plot(autoencoder1, to_file=epsname, show_shapes=True)

# Train first autoencoder on raw input.
autoencoder1.fit(x_train, x_train,
                 nb_epoch=200,
                 batch_size=256,
                 shuffle=True,
                 validation_data=(x_val, x_val))

# Stack the 2nd autoencoder (connecting to encode1):
encode2_layer = Dense(hidden2,
                      activation='sigmoid',
                      activity_regularizer=activity_l1(args.activity_l1),
                      W_regularizer=l2(args.weight_l2),
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
if args.net_plot:
    pngname = '%s/keras_autoencoder2.png' % (args.output_dir,)
    epsname = '%s/keras_autoencoder2.eps' % (args.output_dir,)
    print("Saving %s..." % (pngname,))
    plot(autoencoder2, to_file=pngname, show_shapes=True)
    print("Saving %s..." % (epsname,))
    plot(autoencoder2, to_file=epsname, show_shapes=True)

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
if args.net_plot:
    pngname = '%s/keras_sae.png' % (args.output_dir,)
    epsname = '%s/keras_sae.eps' % (args.output_dir,)
    print("Saving %s..." % (pngname,))
    plot(sae, to_file=pngname, show_shapes=True)
    print("Saving %s..." % (epsname,))
    plot(sae, to_file=epsname, show_shapes=True)
sae.fit(x_train, x_train,
        nb_epoch=200,
        batch_size=256,
        shuffle=True,
        validation_data=(x_val, x_val))

# Then, here is our model which provides 2nd hidden layer features:
stacked_encoder = Model(input=raw_input_tensor, output=encode2_tensor)
if args.net_plot:
    pngname = '%s/keras_stacked_encoder.png' % (args.output_dir,)
    epsname = '%s/keras_stacked_encoder.eps' % (args.output_dir,)
    print("Saving %s..." % (pngname,))
    plot(stacked_encoder, to_file=pngname, show_shapes=True)
    print("Saving %s..." % (epsname,))
    plot(stacked_encoder, to_file=epsname, show_shapes=True)

print("Plotting 1st-layer weights...")
# Get means from 1st-layer weights:
utils.plot_weights(encode1_layer.get_weights()[0][:30,:], None)
epsname = "%s/%s_keras_layer1.eps" % (args.output_dir, suffix)
pngname = "%s/%s_keras_layer1.png" % (args.output_dir, suffix)
print("Saving %s..." % (pngname,))
plt.savefig(pngname, bbox_inches='tight')
print("Saving %s..." % (epsname,))
plt.savefig(epsname, bbox_inches='tight')
plt.close()

#######################################################################
# t-SNE
#######################################################################

if args.tsne:
    code1, code2 = labels_df["ICD9_CATEGORY"].unique()
    def category_to_color(c):
        if c == code1:
            return "red"
        elif c == code2:
            return "blue"
        else:
            raise Exception("Unknown category: %s" % (c,))
    colors = [category_to_color(i) for i in x_labels]

    # Build model for 1st-layer features:
    encoder1 = Model(input=raw_input_tensor, output=encode1_tensor)

    features_raw1 = encoder1.predict(x_data)
    ss = sklearn.preprocessing.StandardScaler()
    features1 = ss.fit_transform(features_raw1)

    print("t-SNE on 1st-layer features...")
    tsne1 = sklearn.manifold.TSNE(random_state = 0)
    Y_tsne1 = tsne1.fit_transform(features1)

    plt.scatter(Y_tsne1[:,0], Y_tsne1[:, 1], color = colors, s=2)
    epsname = "%s/%s_tsne_layer1.eps" % (args.output_dir, suffix)
    pngname = "%s/%s_tsne_layer1.png" % (args.output_dir, suffix)
    print("Saving %s..." % (pngname,))
    plt.savefig(pngname, bbox_inches='tight')
    print("Saving %s..." % (epsname,))
    plt.savefig(epsname, bbox_inches='tight')
    plt.close()

    print("t-SNE on 2nd-layer features...")
    features_raw2 = stacked_encoder.predict(x_data)
    ss = sklearn.preprocessing.StandardScaler()
    features2 = ss.fit_transform(features_raw2)

    tsne2 = sklearn.manifold.TSNE(random_state = 0)
    Y_tsne2 = tsne2.fit_transform(features2)

    plt.scatter(Y_tsne2[:,0], Y_tsne2[:, 1], color = colors, s=2)
    epsname = "%s/%s_tsne_layer2.eps" % (args.output_dir, suffix)
    pngname = "%s/%s_tsne_layer2.png" % (args.output_dir, suffix)
    print("Saving %s..." % (pngname,))
    plt.savefig(pngname, bbox_inches='tight')
    print("Saving %s..." % (epsname,))
    plt.savefig(epsname, bbox_inches='tight')
    plt.close()
