package edu.gatech.cse8803.main

import org.apache.spark.rdd.RDD
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.storage.StorageLevel
import java.sql.Timestamp
import com.cloudera.sparkts._

import edu.gatech.cse8803.util.Utils

case class PatientEventSeries(
  subject_id: Int,
  adm_id: Int,
  item_id: Int,
  unit: String,
  // why must I fully-qualify this despite importing?
  series: Iterable[(java.sql.Timestamp, String)]
)

case class LabItem(
  item_id: Int,
  label: String,
  fluid: String,
  category: String,
  loincCode: String
)

object Main {
  def main(args: Array[String]) {
    import org.apache.log4j.Logger
    import org.apache.log4j.Level

    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)

    val sc = Utils.createContext
    val sqlctxt = new SQLContext(sc)
    import sqlctxt.implicits._

    // These are small enough to cache:
    val d_icd_diagnoses = Utils.csv_from_s3(
      sqlctxt, "D_ICD_DIAGNOSES", Some(d_icd_diagnoses_schema))
    d_icd_diagnoses.cache()

    // D_LABITEMS is fairly small, so make a map of it:
    val labitems : Map[Int,LabItem] = Utils.csv_from_s3(
      sqlctxt, "D_LABITEMS", Some(d_labitems_schema)).
      rdd.map { r: Row =>
        val li = LabItem(r.getAs("ITEMID"), r.getAs("LABEL"),
          r.getAs("FLUID"), r.getAs("CATEGORY"), r.getAs("LOINC_CODE"))
        (li.item_id, li)
      }.collect.toMap

    val loinc = sqlctxt.
      read.
      format("com.databricks.spark.csv").
      option("header", "true").
      load("s3://bd4h-mimic3/LOINC_2.56_Text/loinc.csv")
    //loinc.createOrReplaceTempView("loinc")
    //loinc.cache()

    val patients = Utils.csv_from_s3(
      sqlctxt, "PATIENTS", Some(patients_schema))
    val labevents = Utils.csv_from_s3(
      sqlctxt, "LABEVENTS", Some(labevents_schema))
    val diagnoses_icd = Utils.csv_from_s3(
      sqlctxt, "DIAGNOSES_ICD", Some(diagnoses_schema))

    val lab_ts = labevents.
      rdd.map { row =>
        ((row.getAs[Int]("SUBJECT_ID"),
          row.getAs[Int]("HADM_ID"),
          row.getAs[Int]("ITEMID")),
          (row.getAs[String]("VALUEUOM"),
            row.getAs[Timestamp]("CHARTTIME"),
            row.getAs[String]("VALUE"))
        )
        // ((subject ID, adm. ID, item ID),
        //  (unit, value, time))
      }.groupByKey.map { case ((subj, adm, item), l) =>
          // This is clunky (what if units may differ?):
          val unit = l.toList(0)._1
          val series = l.map { case (_,t,v) => (t,v) }
          PatientEventSeries(subj, adm, item, unit, series)
      }
    lab_ts.persist(StorageLevel.MEMORY_AND_DISK)

    /*
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
     */

    println("Hello, World")
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
   val admissions = Utils.csv_from_s3("ADMISSIONS")
   val callout = Utils.csv_from_s3("CALLOUT")
   val caregivers = Utils.csv_from_s3("CAREGIVERS")
   val chartevents = Utils.csv_from_s3("CHARTEVENTS")
   val cptevents = Utils.csv_from_s3("CPTEVENTS")
   val datetimeevents = Utils.csv_from_s3("DATETIMEEVENTS")
   val drgcodes = Utils.csv_from_s3("DRGCODES")
   val d_cpt = Utils.csv_from_s3("D_CPT")
   val d_icd_procedures = Utils.csv_from_s3("D_ICD_PROCEDURES")
   val d_items = Utils.csv_from_s3("D_ITEMS")
   val icustays = Utils.csv_from_s3("ICUSTAYS")
   val inputevents_cv = Utils.csv_from_s3("INPUTEVENTS_CV")
   val inputevents_mv = Utils.csv_from_s3("INPUTEVENTS_MV")
   val microbiologyevents = Utils.csv_from_s3("MICROBIOLOGYEVENTS")
   val noteevents = Utils.csv_from_s3("NOTEEVENTS")
   val outputevents = Utils.csv_from_s3("OUTPUTEVENTS")
   val prescriptions = Utils.csv_from_s3("PRESCRIPTIONS")
   val procedureevents_mv = Utils.csv_from_s3("PROCEDUREEVENTS_MV")
   val procedures_icd = Utils.csv_from_s3("PROCEDURES_ICD")
   val services = Utils.csv_from_s3("SERVICES")
   val transfers = Utils.csv_from_s3("TRANSFERS")
   */
}
