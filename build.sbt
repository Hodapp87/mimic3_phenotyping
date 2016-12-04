name := "mimic3_phenotyping"

version := "1.0"

scalaVersion := "2.11.8"

val sparkVersion = "2.0.1"

resolvers ++= Seq(
  Resolver.sonatypeRepo("releases"),
  Resolver.sonatypeRepo("snapshots")
)

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % sparkVersion,
  "org.apache.spark" %% "spark-sql" % sparkVersion,
  "org.apache.spark" %% "spark-mllib" % sparkVersion,
  "org.apache.spark" %% "spark-streaming" % sparkVersion,
  "org.apache.spark" %% "spark-hive" % sparkVersion,
  "com.github.scopt" %% "scopt" % "3.5.0"
  //"com.databricks"    % "spark-csv_2.11" % "1.5.0"
)

parallelExecution in Test := false
