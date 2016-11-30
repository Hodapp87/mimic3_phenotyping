mimic3_phenotyping
==================

Prerequisites
----

- MIMIC-III dataset
- sbt & Spark
- Python, Keras, scikit-learn

Building
----

- sbt

Running
----

Example run
----

```
spark-submit --master "local[*]"
    --repositories https://oss.sonatype.org/content/groups/public/
    --packages "com.github.scopt:scopt_2.11:3.5.0"
    target/scala-2.11/cse8803_project_2.11-1.0.jar
    -i "file:////mnt/dev/mimic3/"
    -o "file:///home/hodapp/source/bd4h-project/data/"
    -m -c --icd9a 518 --icd9b 584 -l "11558-4"
```
