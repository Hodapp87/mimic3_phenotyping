package edu.gatech.cse8803.util

import org.apache.spark.sql._
import org.apache.spark.sql.types._

object Utils {
  /**
   * Load the given CSV file, returning the dataframe for that CSV.
   * In the process, this prints out the schema inferred (for the sake
   * of easier loading later) if one is not supplied.
   */
  def csv_from_s3(spark : SparkSession, fname : String, schema : Option[StructType] = None) : DataFrame = {
    val schema_fn = (f : DataFrameReader) =>
    if (schema.isDefined) f.schema(schema.get) else f

    val df = schema_fn(spark.
      read.
      format("com.databricks.spark.csv").
      option("header", "true").
      option("mode", "DROPMALFORMED")).
      load(fname)
    
    if (!schema.isDefined) {
      println("Inferred schema:")
      print(df.schema)
    }
    
    // df.createOrReplaceTempView(base)
    
    df
  }

  def polynomialTimeWarp(
    ts : Seq[Double], a : Double = 3.0, b : Double = 0.0) :
      Seq[Double] =
  {
    /**
      * Implement the polynomial time warping described in
      * "Computational Phenotype Discovery" by Lasko, Denny, & Levy
      * (2013), equation 5.  This will also remove any offset that is
      * present, that is, the resultant vector will start at 0.
      * 
      * Args:
      *  ts: Time values, which should already be sorted.
      *  a: Optional value of coefficient 'a'; default is 3.0.
      *  b: Optional value of coefficient 'b'; default is 0.0.
      * 
      * Returns:
      *  Warped version of 'ts' (a sequence of the same length).
      */

    val diffs = ts.zip(ts.drop(1)).map { case (x,y) =>
      // Find successive differences, then also warp them:
      Math.pow(y - x, 1/a) + b
    }
    diffs.foldLeft(Seq(0.0))((l,d) => (d + l(0)) +: l).reverse
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
