package edu.gatech.cse8803.util

import org.apache.spark.sql._
import org.apache.spark.sql.types._

object Utils {
  /**
   * Factor out this common functionality into a function:  Look in s3_dir
   * (where I've placed the MIMIC-III data) for a .csv.gz file with the given
   * name.  Return the dataframe for that CSV, and also create a temp table
   * with the same base name.  In the process, this prints out the schema used
   * (for the sake of easier loading later)
   */
  def csv_from_s3(spark : SparkSession, base : String, schema : Option[StructType] = None) : DataFrame = {
    val s3_dir = "s3://bd4h-mimic3/"
    val schema_fn = (f : DataFrameReader) =>
    if (schema.isDefined) f.schema(schema.get) else f
    val df = schema_fn(spark.
      read.
      format("com.databricks.spark.csv").
      option("header", "true").
      option("mode", "DROPMALFORMED")).
      load(f"${s3_dir}${base}.csv.gz")
    
    if (!schema.isDefined) {
      println("Inferred schema:")
      print(df.schema)
    }
    
    df.createOrReplaceTempView(base)
    
    df
  }

  /** Pretty-print a dataframe for Zeppelin.  Either supply
   * a number of rows to take, or 'None' to take all of them.
   */
  def pprint(df : DataFrame, n : Option[Int] = None) = {
    val hdr = df.columns.mkString("\t")
    val array = if (n.isDefined) df.take(n.get) else df.collect
    val table = array.map { _.mkString("\t") }.mkString("\n")
    // To use with Zeppelin, switch out the below:
    //val zeppelin_pfx = "%table "
    val zeppelin_pfx = " "
    println(f"${zeppelin_pfx}${hdr}\n${table}")
  }

  def getSession() = {
  }

  def createContext(appName: String, masterUrl: String): SparkSession = {
    SparkSession.builder.
      master(masterUrl)
      .appName(appName)
      .getOrCreate()
  }

  def createContext(appName: String): SparkSession =
    createContext(appName, "local[*]")

  def createContext: SparkSession =
    createContext("CSE-8803 project", "local[*]")
  
}
