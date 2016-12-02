mimic3_phenotyping
==================

- Background - Lasko's paper, my paper (arxiv?)

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

Running
----

- Amazon EMR or local?
- Filenames contain ICD-9 codes and LOINC code if they are cohort-specific
- Python code is run after

Example run
----

```
spark-submit --master "local[*]" \
    --repositories https://oss.sonatype.org/content/groups/public/ \
    --packages "com.github.scopt:scopt_2.11:3.5.0" \
    target/scala-2.11/mimic3_phenotyping_2.11-1.0.jar \
    -i "file:////mnt/dev/mimic3/" \
    -o "file:///home/hodapp/source/bd4h-project-code/data/" \
    -m -c -h -r --icd9a 428 --icd9b 571 -l "1742-6"

python timeseries_plots.py -d ./data -o ./data --icd9a 428 --icd9b 571 --loinc 1742-6
python feature_learning.py -d ./data -o ./data --icd9a 428 --icd9b 571 --loinc 1742-6 --activity_l1 0.00004 --weight_l2 0.0004 --tsne --logistic_regression
```

Known Problems & Needed Improvements
----

### Spark code

- Hyperparameter optimization is neither fast nor accurate.  It should
  be changed to gradient descent at some point, and the range in which
  it searches should be expanded considerably.
- A problem
  like
  [this](https://stackoverflow.com/questions/34329299/issuing-spark-submit-on-command-line-completes-tasks-but-never-returns-prompt) occurs
  sometimes, especially around hyperparameter optimization.  Check
  Spark's web UI to see when jobs actually finish.  Thus far I've seen
  this only with master `local[*]`, not with YARN on Amazon EMR.
- Parquet emits a lot of annoying messages that I don't know how to
  silence, and they get in the way of messages that matter.
- A more sensible way probably exists to determine the padding and
  sample frequency for the interpolation after Gaussian process
  regression.  At a minimum, the code should supply some statistics
  such as average warped time-series length.
- I don't know of any good way to make Spark overwrite RDD files when
  I use `saveAsObjectFile`.

### Python code
- The code on occasion will segfault, and I have not yet been able to
  reproduce it reliably enough to determine why.
