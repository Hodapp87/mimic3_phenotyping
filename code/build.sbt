name := "cse8803_project"

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
  "com.cloudera.sparkts" % "sparkts" % "0.4.0",
  "com.databricks"    % "spark-csv_2.11" % "1.5.0"
)

parallelExecution in Test := false

assemblyMergeStrategy in assembly := {
 case PathList("META-INF", xs @ _*) => MergeStrategy.discard
 case x => MergeStrategy.first
}

mainClass in assembly := Some("edu.gatech.cse8803.main")

// spark-shell --packages "com.cloudera.sparkts:sparkts:0.4.0"
// Perhaps also consult:
// https://stackoverflow.com/questions/37643831/how-to-run-spark-shell-with-local-packages
