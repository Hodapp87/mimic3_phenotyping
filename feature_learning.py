#!/usr/bin/env python

# (c) 2016 Chris Hodapp, chodapp3@gatech.edu
# feature_learning.py: Tool for reading in data from a prior Spark
# run, conditioning it for input to an autoencoder, greedy layerwise
# training an autoencoder on it (or loading prior trained results),
# producing some plots of neural network weights, and using the
# learned features for t-SNE visualization and a logistic regression
# classifier.

import utils

import argparse
import math

import pandas
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy
import sklearn.manifold
import sklearn.preprocessing
import sklearn.linear_model # LogisticRegression
import sklearn.metrics

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
parser.add_argument("-m", "--load_model",
                    help="Load trained weights from given HDF5 (.h5) file rather than training",
                    default="")
parser.add_argument("-s", "--save_model",
                    help="Save trained weights as an HDF5 (.h5) file",
                    default="")
parser.add_argument("-w", "--weight_l2",
                    help="Set L2 weight regularization (default 0.0003); ignored for -m",
                    default=0.0003)
parser.add_argument("-a", "--activity_l1",
                    help="Set L1 activity regularization (default 0.00003); ignored for -m",
                    default=0.00003)
parser.add_argument("-p", "--patch_length",
                    help="Set patch length for neural network training and testing",
                    default=20)
parser.add_argument("-r", "--logistic_regression",
                    help="Train and run logistic regression classifier",
                    action="store_true")

args = parser.parse_args()

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

# Training data: load interpolated time-series, and group together
# (they're in flattened form in the CSV):
csvname = "%s/%s_predict.csv" % (args.data_dir, suffix)
print("Trying to load: %s" % (csvname,))
train = pandas.read_csv(utils.get_single_csv(csvname))
train.fillna("", inplace = True)

# Standardize input to mean 0, variance 1 (to confuse matters a
# little, the data that we're standardizing is itself mean and
# variance of the Gaussian process)
train["MEAN"] = utils.standardize(train["MEAN"])
train["VARIANCE"] = utils.standardize(train["VARIANCE"])

train_groups_ = train.groupby((train["HADM_ID"], train["ITEMID"], train["VALUEUOM"]))
train_groups = list(train_groups_)
print("Got %d train points (%d admissions)." % (len(train), len(train_groups)))

# Then do likewise for testing data:
csvname = "%s/%s_predict_test.csv" % (args.data_dir, suffix)
print("Trying to load: %s" % (csvname,))
test = pandas.read_csv(utils.get_single_csv(csvname))
test.fillna("", inplace = True)

# Standardize input to mean 0, variance 1 (to confuse matters a
# little, the data that we're standardizing is itself mean and
# variance of the Gaussian process)
test["MEAN"] = utils.standardize(test["MEAN"])
test["VARIANCE"] = utils.standardize(test["VARIANCE"])

test_groups_ = test.groupby((test["HADM_ID"], test["ITEMID"], test["VALUEUOM"]))
test_groups = list(test_groups_)
print("Got %d testing points (%d admissions)." % (len(test), len(test_groups)))

# 'train_groups' and 'test_groups' then are lists of:
# ((HADM_ID, ITEMID, VALUEUOM), time-series dataframe)

# Also load the labels; the CSV has (HADM_ID, ICD-9 category):
csvname = "%s/%s_categories.csv" % (args.data_dir, suffix)
print("Trying to load: %s" % (csvname,))
labels_df = pandas.read_csv(utils.get_single_csv(csvname))
# Turn it to a dictionary with HADM_ID -> category:
labels = dict(zip(labels_df["HADM_ID"], labels_df["ICD9_CATEGORY"]))

labels_hist = {}
for (hadm_id,_,_),_ in train_groups:
    label = labels[hadm_id]
    if label not in labels_hist:
        labels_hist[label] = 0
    labels_hist[label] += 1

print("Counts in training data:")
print(labels_hist)

#######################################################################
# Gathering patches
#######################################################################

x_data, x_labels = utils.sample_patches(
    train_groups, 3, args.patch_length, labels)
print("Sampled %d patches of data (training)" % (len(x_data),))

x_data_test, x_labels_test = utils.sample_patches(
    test_groups, 3, args.patch_length, labels)
print("Sampled %d patches of data (testing)" % (len(x_data_test),))

#######################################################################
# Training/validation split
#######################################################################
# This is a bit confusing.  'Training' here refers just to the
# training of the neural network, where we set aside part of the
# training data for actual training and part for validation.

# What ratio of the data to leave behind for validation
validation_ratio = 0.2

# Produce [0, 1, 2...N-1], and shuffle these (rather than shuffling
# x_data):
idxs = numpy.arange(len(x_data))
numpy.random.shuffle(idxs)
split_idx = int(len(x_data) * validation_ratio)
# and use shuffled indices to split training/validation:
x_val, x_train = x_data[idxs[split_idx:],:], x_data[idxs[:split_idx],:]
print("Split r=%g: %d patches for training, %d for validation" %
      (validation_ratio, len(x_val), len(x_train)))

#######################################################################
# Stacked Autoencoder
#######################################################################

# Should we train this net? (versus loading from file)
train_net = args.load_model == ""

# Size of hidden layers:
hidden1 = 100
hidden2 = 100

# Input is size of one patch, with another dimension of size 2 (for
# mean & variance)
ts_shape = (args.patch_length * 2,)

raw_input_tensor = Input(shape=ts_shape, name="raw_input")
encode1_layer = Dense(hidden1,
                      activation='sigmoid',
                      activity_regularizer=activity_l1(args.activity_l1),
                      W_regularizer=l2(args.weight_l2),
                      name="encode1")
encode1_tensor = encode1_layer(raw_input_tensor)

decode1_layer = Dense(output_dim=args.patch_length * 2,
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

if train_net:
    # Train first autoencoder on raw input.
    autoencoder1.fit(x_train, x_train,
                     nb_epoch=500,
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

decode2_layer = Dense(output_dim=args.patch_length * 2,
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

if train_net:
    # Train second autoencoder.  We're basically training it on
    # primary hidden features, not raw input, because we're keeping
    # the first encoder's weights constant (hence 'trainable = False'
    # above) and so it is simply feeding encoded input in.
    autoencoder2.fit(x_train, x_train,
                     nb_epoch=500,
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
if train_net:
    sae.fit(x_train, x_train,
            nb_epoch=500,
            batch_size=256,
            shuffle=True,
            validation_data=(x_val, x_val))

# Load weights from a pre-trained model if requested:
if args.load_model != "":
    print("Loading weights from %s..." % (args.load_model,))
    sae.load_weights(args.load_model)
    # TODO: Check if this propagates to stacked_encoder properly

# And save the model weights if requested:
if args.save_model != "":
    print("Saving weights to %s..." % (args.save_model,))
    sae.save_weights(args.save_model)

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

# Get 1st-layer features:
encoder1 = Model(input=raw_input_tensor, output=encode1_tensor)
features_raw1 = encoder1.predict(x_data)
features_raw1_test = encoder1.predict(x_data_test)
# Standardize training, and apply same transform to test:
ss = sklearn.preprocessing.StandardScaler()
features1 = ss.fit_transform(features_raw1)
features1_test = ss.transform(features_raw1_test)

# Get 2nd-layer features:
features_raw2 = stacked_encoder.predict(x_data)
features_raw2_test = stacked_encoder.predict(x_data_test)
ss = sklearn.preprocessing.StandardScaler()
# Likewise, standardize training, and apply same transform to test:
features2 = ss.fit_transform(features_raw2)
features2_test = ss.transform(features_raw2_test)

# Prepare labels:
code1, code2 = labels_df["ICD9_CATEGORY"].unique()
x_labels_num = numpy.array([1 * (c == code1) for c in x_labels])
x_labels_num_test = numpy.array([1 * (c == code1) for c in x_labels_test])

#######################################################################
# t-SNE
#######################################################################

if args.tsne:
    def category_to_color(c):
        if c == code1:
            return "red"
        elif c == code2:
            return "blue"
        else:
            raise Exception("Unknown category: %s" % (c,))
    colors = [category_to_color(i) for i in x_labels]

    print("t-SNE on 1st-layer features...")
    tsne1 = sklearn.manifold.TSNE(perplexity = 20)
    Y_tsne1 = tsne1.fit_transform(features1)

    s1 = plt.scatter(Y_tsne1[x_labels == code1, 0], Y_tsne1[x_labels == code1, 1], color = "red", s=2)
    s2 = plt.scatter(Y_tsne1[x_labels == code2, 0], Y_tsne1[x_labels == code2, 1], color = "blue", s=2)
    plt.legend((s1, s2), (str(code1), str(code2)))
    epsname = "%s/%s_tsne_layer1.eps" % (args.output_dir, suffix)
    pngname = "%s/%s_tsne_layer1.png" % (args.output_dir, suffix)
    print("Saving %s..." % (pngname,))
    plt.savefig(pngname, bbox_inches='tight')
    print("Saving %s..." % (epsname,))
    plt.savefig(epsname, bbox_inches='tight')
    plt.close()

    print("t-SNE on 2nd-layer features...")

    tsne2 = sklearn.manifold.TSNE(perplexity = 20)
    Y_tsne2 = tsne2.fit_transform(features2)

    s1 = plt.scatter(Y_tsne2[x_labels == code1, 0], Y_tsne2[x_labels == code1, 1], color = "red", s=2)
    s2 = plt.scatter(Y_tsne2[x_labels == code2, 0], Y_tsne2[x_labels == code2, 1], color = "blue", s=2)
    plt.legend((s1, s2), (str(code1), str(code2)))
    epsname = "%s/%s_tsne_layer2.eps" % (args.output_dir, suffix)
    pngname = "%s/%s_tsne_layer2.png" % (args.output_dir, suffix)
    print("Saving %s..." % (pngname,))
    plt.savefig(pngname, bbox_inches='tight')
    print("Saving %s..." % (epsname,))
    plt.savefig(epsname, bbox_inches='tight')
    plt.close()

#######################################################################
# Logistic Regression
#######################################################################

def classification_metrics(Y_pred, Y_true):
    return (sklearn.metrics.accuracy_score (Y_true, Y_pred),
            sklearn.metrics.roc_auc_score  (Y_true, Y_pred),
            sklearn.metrics.precision_score(Y_true, Y_pred),
            sklearn.metrics.recall_score   (Y_true, Y_pred),
            sklearn.metrics.f1_score       (Y_true, Y_pred))

def display_metrics(Y_pred,Y_true):
    print("______________________________________________")
    acc, auc_, precision, recall, f1score = classification_metrics(Y_pred,Y_true)
    print("Accuracy: "+str(acc))
    print("AUC: "+str(auc_))
    print("Precision: "+str(precision))
    print("Recall: "+str(recall))
    print("F1-score: "+str(f1score))
    print("______________________________________________")
    print("")

if args.logistic_regression:
    print("Logistic regression on 1st layer features:")
    model = sklearn.linear_model.LogisticRegression()
    model.fit(features1, x_labels_num)
    pred = model.predict(features1_test)
    display_metrics(pred, x_labels_num_test)

    print("Logistic regression on 2nd layer features:")
    model = sklearn.linear_model.LogisticRegression()
    model.fit(features2, x_labels_num)
    pred = model.predict(features2_test)
    display_metrics(pred, x_labels_num_test)
