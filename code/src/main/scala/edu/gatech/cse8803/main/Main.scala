package edu.gatech.cse8803.main

import org.apache.spark.{SparkConf, SparkContext}                 
import org.apache.spark.rdd.RDD
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
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
    val sqlctxt = new SQLContext(sc)
    import sqlctxt.implicits._

    val patients = csv_from_s3(sqlctxt, "PATIENTS", Some(patients_schema))
    val d_labitems = csv_from_s3(sqlctxt, "D_LABITEMS", Some(d_labitems_schema))
    d_labitems.cache()
    val labevents = csv_from_s3(sqlctxt, "LABEVENTS", Some(labevents_schema))
    val diagnoses_icd = csv_from_s3(sqlctxt, "DIAGNOSES_ICD", Some(diagnoses_schema))
    val d_icd_diagnoses = csv_from_s3(sqlctxt, "D_ICD_DIAGNOSES", Some(d_icd_diagnoses_schema))

    val loinc = sqlctxt.
      read.
      format("com.databricks.spark.csv").
      option("header", "true").
      load("s3://bd4h-mimic3/LOINC_2.56_Text/loinc.csv")
    loinc.createOrReplaceTempView("loinc")
    loinc.cache()

    val groups_with_type = labs_grouped.
      join(d_labitems, "ITEMID").
      join(loinc, loinc("LOINC_NUM") === d_labitems("LOINC_CODE")).
      select("ITEMID", "SUBJECT_ID", "count", "LABEL", "FLUID", "CATEGORY", "LOINC_CODE", "LONG_COMMON_NAME")
    // pprint(groups_with_type, Some(100))
    val lab_events_per_patient = groups_with_type.
      groupBy("ITEMID").
      mean("count").
      orderBy(desc("avg(count)"))
    // pprint(lab_events_per_patient, Some(100))
    val labs_grouped = labevents.
      groupBy("SUBJECT_ID", "ITEMID").
      count.
      orderBy(desc("count"))
    labs_grouped.persist(StorageLevel.MEMORY_AND_DISK)

    println("Hello, World")
  }

  /** Pretty-print a dataframe for Zeppelin.  Either supply
   * a number of rows to take, or 'None' to take all of them.
   */
  def pprint(df : DataFrame, n : Some[Int]) = {
    val hdr = df.columns.mkString("\t")
    val array = if (n.isDefined) df.take(n.get) else df.collect
    val table = array.map { _.mkString("\t") }.mkString("\n")
    println(f"%%table ${hdr}\n ${table}")
  }

  case class PatientEventSeries(
    subject_id: Int,
    adm_id: Int,
    loinc_id: String,
    unit: String,
    // why must I fully-qualify this despite importing?
    series: List[(java.sql.Timestamp, Double)]
  )

  /****************************************************************************
   * Factor out this common functionality into a function:  Look in s3_dir
   * (where I've placed the MIMIC-III data) for a .csv.gz file with the given
   * name.  Return the dataframe for that CSV, and also create a temp table
   * with the same base name.  In the process, this prints out the schema used
   * (for the sake of easier loading later)
   *****************************************************************************/
  def csv_from_s3(sqlContext : SQLContext, base : String, schema : Option[StructType] = None) : DataFrame = {
    val s3_dir = "s3://bd4h-mimic3/"
    val schema_fn = (f : DataFrameReader) =>
    if (schema.isDefined) f.schema(schema.get) else f
    val df = schema_fn(sqlContext.
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

  val patients_schema = StructType(Array(
    StructField("ROW_ID",IntegerType,true),
    StructField("SUBJECT_ID",IntegerType,true),
    StructField("GENDER",StringType,true),
    StructField("DOB",TimestampType,true),
    StructField("DOD",TimestampType,true),
    StructField("DOD_HOSP",TimestampType,true),
    StructField("DOD_SSN",TimestampType,true),
    StructField("EXPIRE_FLAG",StringType,true)
  ))

  val d_labitems_schema = StructType(Array(
    StructField("ROW_ID",IntegerType,true),
    StructField("ITEMID",IntegerType,true),
    StructField("LABEL",StringType,true),
    StructField("FLUID",StringType,true),
    StructField("CATEGORY",StringType,true),
    StructField("LOINC_CODE",StringType,true)
  ))

  val labevents_schema = StructType(Array(
    StructField("ROW_ID",IntegerType,true),
    StructField("SUBJECT_ID",IntegerType,true),
    StructField("HADM_ID",IntegerType,true),
    StructField("ITEMID",IntegerType,true),
    StructField("CHARTTIME",TimestampType,true),
    StructField("VALUE",StringType,true),
    StructField("VALUENUM",DoubleType,true),
    StructField("VALUEUOM",StringType,true),
    StructField("FLAG",StringType,true)
  ))

  val diagnoses_schema = StructType(Array(
    StructField("ROW_ID",IntegerType,true),
    StructField("SUBJECT_ID",IntegerType,true),
    StructField("HADM_ID",IntegerType,true),
    StructField("SEQ_NUM",IntegerType,true),
    StructField("ICD9_CODE",StringType,true)
  ))

  val d_icd_diagnoses_schema = StructType(Array(
    StructField("ROW_ID",IntegerType,true),
    StructField("ICD9_CODE",StringType,true),
    StructField("SHORT_TITLE",StringType,true),
    StructField("LONG_TITLE",StringType,true)
  ))

  /*
   val admissions = csv_from_s3("ADMISSIONS")
   val callout = csv_from_s3("CALLOUT")
   val caregivers = csv_from_s3("CAREGIVERS")
   val chartevents = csv_from_s3("CHARTEVENTS")
   val cptevents = csv_from_s3("CPTEVENTS")
   val datetimeevents = csv_from_s3("DATETIMEEVENTS")
   val drgcodes = csv_from_s3("DRGCODES")
   val d_cpt = csv_from_s3("D_CPT")
   val d_icd_procedures = csv_from_s3("D_ICD_PROCEDURES")
   val d_items = csv_from_s3("D_ITEMS")
   val icustays = csv_from_s3("ICUSTAYS")
   val inputevents_cv = csv_from_s3("INPUTEVENTS_CV")
   val inputevents_mv = csv_from_s3("INPUTEVENTS_MV")
   val microbiologyevents = csv_from_s3("MICROBIOLOGYEVENTS")
   val noteevents = csv_from_s3("NOTEEVENTS")
   val outputevents = csv_from_s3("OUTPUTEVENTS")
   val prescriptions = csv_from_s3("PRESCRIPTIONS")
   val procedureevents_mv = csv_from_s3("PROCEDUREEVENTS_MV")
   val procedures_icd = csv_from_s3("PROCEDURES_ICD")
   val services = csv_from_s3("SERVICES")
   val transfers = csv_from_s3("TRANSFERS")
   */

  def createContext(appName: String, masterUrl: String): SparkContext = {
    val conf = new SparkConf().setAppName(appName).setMaster(masterUrl)
    new SparkContext(conf)
  }

  def createContext(appName: String): SparkContext =
    createContext(appName, "local[*]")

  def createContext: SparkContext =
    createContext("CSE-8803 project", "local[*]")
}
