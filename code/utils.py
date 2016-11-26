#!/usr/bin/env python3

# (c) 2016 Chris Hodapp, chodapp3@gatech.edu

import os

def get_single_csv(dirname):
    # Since Spark saves the CSV data by partition, we must still find the
    # single file in that path (we coalesced to one partition, but it's
    # still not a known filename):
    files = [d for d in os.listdir(dirname) if d.lower().endswith(".csv")]
    if len(files) != 1:
        raise Exception("Can't find single CSV file in %s" % (dirname,))
    fname = os.path.join(dirname, files[0])
    return fname
