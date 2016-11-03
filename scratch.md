# Scratch Work (CSE8803-O01, Big Data Analytics for Healthcare)
- Chris Hodapp <chodapp3@gatech.edu>
- N.B. Add key `zeppelin.spark.printREPLOutput` and value `true` to the properties of the `spark` interpreter in Zeppelin, or the REPL output (including errors) will be hidden.  This should be the default ([source](https://zeppelin.apache.org/docs/latest/interpreter/spark.html)), but for some reason is not - perhaps it is a newer change.  See `ec2/zeppelin_printREPL.sh` for a script to patch this automatically.

## Links
- A good reference for MIMIC-III tables is [here](https://mimic.physionet.org/mimictables/admissions/), to use `ADMISSIONS` as an example.
- [Spark API](https://spark.apache.org/docs/latest/api/scala/index.html)
- [Spark docs](https://spark.apache.org/docs/latest/)
- [LOINC](https://loinc.org/) looks like it will be very useful for the lab results.
- [Spark SQL and DataFrames](https://spark.apache.org/docs/latest/sql-programming-guide.html)
- [Spark interpreter for Zeppelin](https://zeppelin.apache.org/docs/latest/interpreter/spark.html); note that we are `yarn-client` by default on EMR
- [spark-ts](https://sryza.github.io/spark-timeseries) for time-series data
- [A Tale of Three Apache Spark APIs: RDDs, DataFrames, and Datasets](https://databricks.com/blog/2016/07/14/a-tale-of-three-apache-spark-apis-rdds-dataframes-and-datasets.html)
- [Spark 2.0: DataSets and Case Classes](https://blog.codecentric.de/en/2016/07/spark-2-0-datasets-case-classes/)

## Questions
- Do I need to process numbers of these CSVs into a more friendly format?  I could let `com.databricks.spark.csv` infer the schema but that might cause performance woes if it insists on parsing twice; I could also [specify it directly](https://github.com/databricks/spark-csv#scala-api).  A "proper" relational DB might be best here, but I'm going to skip that and use Hadoop/Spark instead for the sake of the class.
- What is Spark SQL getting me here?  I am less familiar with it, and I don't think any of the analysis that I must do will be feasible on its `DataFrame` directly.  I can write the queries in Spark's DSL, but I'm also not clear on what the point of that is (aside from the benefits of static typing).

## Other notes
- I would really rather be typing in Markdown than in org-mode, as familiar as I am with Emacs key bindings.
- I feel that this "notebook" paradigm is an important one.  In some sense, it restores a lost form of network transparency - but this time using the browser rather than the terminal.
- As much as I bitched about Zeppelin earlier, I'm quite liking some of its features.
- To add packages to Zeppelin, just use its existing [Dependency Management](https://zeppelin.apache.org/docs/latest/manual/dependencymanagement.html) for the Spark interpreter to add the Maven ID for a package. Note that it is not enough to modify `interpreter.json` and restart Zeppelin or the interpreter: you must *edit* in the UI and then save.
- Just set up a SOCKS proxy and use FoxyProxy, like the directions say, instead of forwarding specific ports with SSH.  A bunch of things don't work properly with particular ports (though, forwarding just port 8890 is fine if it is only Zeppelin that's needed).
- From what I can tell, Spark handles making individual nodes alive or dead based on the load.
- Use `Ctrl+.` (as in control + period) for tab-completion!  I don't know how well it works compared to, say, ENSIME, and I'm pretty sure it won't do real-time type-checking as ENSIME does, however, it looks useful nonetheless.
