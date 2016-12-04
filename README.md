mimic3_phenotyping
==================

Chris Hodapp, chodapp3@gatech.edu

Requirements
----

- The [MIMIC-III dataset](http://mimic.physionet.org/gettingstarted/access/)
- [SBT](http://www.scala-sbt.org/sbt) (Scala Build Tools) >= 0.13;
  other versions may work, but I have not tried them.
- Apache Spark
- Python 2.7 or 3.x, and the following packages (`pip` versions should be fine):
  - [Keras](https://keras.io/) and ideally a GPU-enabled backend (Theano or TensorFlow)
  - h5py (if you want to save and load trained networks from Keras)
  - [scikit-learn](http://scikit-learn.org/stable/index.html)
  - pydot-ng (optional)

Building
----

`sbt compile` should handle pulling dependencies and building
everything.  `sbt package` should produce a JAR that `spark-submit`
can handle.

`pip install keras h5py scikit-learn pydot-ng` should handle the
Python prerequisites.

Example run
----

To produce what was in the paper, run the below commands from the same
directory as the code.  For the first command, you will need to supply
two paths: the path containing the `.csv.gz` files from MIMIC-III (for
the `-i` option), and the full path to the `data` directory in this
archive (for the `-o` option).

```
spark-submit --master "local[*]" \
    --repositories https://oss.sonatype.org/content/groups/public/ \
    --packages "com.github.scopt:scopt_2.11:3.5.0" \
    target/scala-2.11/mimic3_phenotyping_2.11-1.0.jar \
    -i "file:////mnt/dev/mimic3/" \
    -o "file:///home/hodapp/source/bd4h-project-code/data/" \
    -m -c -r -b --icd9a 428 --icd9b 571 -l "1742-6"

python timeseries_plots.py -d ./data -o ./data \
    --icd9a 428 --icd9b 571 --loinc 1742-6
    
python feature_learning.py -d ./data -o ./data \
    --icd9a 428 --icd9b 571 --loinc 1742-6 \
    --activity_l1 0.0001 --weight_l2 0.001 \
    --load_model 428_571_1742-6.h5 --tsne --logistic_regression
```

The `spark-submit` command still sometimes exhibits an issue in which
it completes the job but fails to return to the prompt.  Check Spark's
web UI (i.e. http://localhost:4040) for all jobs actually being done.

For expediency, this will skip hyperparameter optimization (which can
take 20-30 minutes depending on machine) and use hyperparameters
already estimated, and it will use weights from a pre-trained neural
network instead of training it.  To actually run through the full
process, add `-h` to the first command, and remove the `--load_model`
option from the `feature_learning` invocation.

All output will be in the `data` directory in PNG and EPS format.
This will include CSV and Parquet files from the Spark code, and PNG
and EPS files from the Python code.
