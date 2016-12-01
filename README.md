mimic3_phenotyping
==================

- Background - Lasko's paper, my paper (arxiv?)
- Don't mention CSE8803-O01

Prerequisites
----

- MIMIC-III dataset
- sbt & Spark
- Python, Keras, scikit-learn

Building
----

- sbt compile, sbt package

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
    -m -c --icd9a 518 --icd9b 584 -l "11558-4"
```
