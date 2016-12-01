// TODO: Change this?
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
  "com.github.scopt" %% "scopt" % "3.5.0"
  //"com.databricks"    % "spark-csv_2.11" % "1.5.0"
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

// spark-submit --master "local[*]" --repositories https://oss.sonatype.org/content/groups/public/ --packages "com.github.scopt:scopt_2.11:3.5.0" target/scala-2.11/cse8803_project_2.11-1.0.jar

//val output_dir : String = "s3://bd4h-mimic3/cohort_518_584_50820/"
//val mimic3_dir : String = "s3://bd4h-mimic3/"

    // Two ICD codes; we want one or the other, but not both.
    // val icd_code1 = "518"
    // val icd_code2 = "584"
//    val icd_code1 = "276"
//    val icd_code2 = "427"

    // ITEMID of the test we're interested in:
    // val item_test = 50820
//    val item_test = 51268

// 11558-4
