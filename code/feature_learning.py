#!/usr/bin/env python

import os
import pandas
import matplotlib.pyplot as plt

input_path = "../data-temp/labs_cohort_predict_518_584.csv"

# Since Spark saves the CSV data by partition, we must still find the
# single file in that path (we coalesced to one partition, but it's
# still not a known filename):
files = [d for d in os.listdir(input_path) if d.endswith(".csv")]
if len(files) != 1:
    raise Exception("Can't find single CSV file in %s" % (input_path,))

fname = os.path.join(input_path, files[0])
print("Loading from %s..." % (fname,))
df = pandas.read_csv()
gr = list(df.groupby((df["HADM_ID"], df["ITEMID"])))
print("Got %d points (%d admissions)." % (len(df), len(gr)))
