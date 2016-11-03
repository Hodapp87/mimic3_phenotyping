package edu.gatech.cse8803.main

import org.apache.spark.{SparkConf, SparkContext}                 
import org.apache.spark.rdd.RDD
import org.apache.spark.sql._
import org.apache.spark.sql.types._
import org.apache.spark.storage.StorageLevel
import java.sql.Timestamp
import com.cloudera.sparkts._

object Main {
  def main(args: Array[String]) {
    import org.apache.log4j.Logger
    import org.apache.log4j.Level

    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)

    val sc = createContext
    val sqlContext = new SQLContext(sc)
    import sqlContext.implicits._

    println("Hello, World")
  }

  def createContext(appName: String, masterUrl: String): SparkContext = {
    val conf = new SparkConf().setAppName(appName).setMaster(masterUrl)
    new SparkContext(conf)
  }

  def createContext(appName: String): SparkContext =
    createContext(appName, "local[*]")

  def createContext: SparkContext =
    createContext("CSE-8803 project", "local[*]")
}
