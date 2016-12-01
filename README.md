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
spark-submit --master "local[*]"
    --repositories https://oss.sonatype.org/content/groups/public/
    --packages "com.github.scopt:scopt_2.11:3.5.0"
    target/scala-2.11/mimic3_phenotyping_2.11-1.0.jar
    -i "file:////mnt/dev/mimic3/"
    -o "file:///home/hodapp/source/bd4h-project-code/data/"
    -m -c -h --icd9a 518 --icd9b 584 -l "11558-4"
```
