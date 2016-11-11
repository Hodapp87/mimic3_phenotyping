package edu.gatech.cse8803.main

import edu.gatech.cse8803.util.Utils
import edu.gatech.cse8803.util.Schemas

// import org.apache.spark.rdd.RDD
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.storage.StorageLevel
import java.sql.Timestamp
// import com.cloudera.sparkts._

case class PatientEventSeries(
  subject_id: Int,
  adm_id: Int,
  item_id: Int,
  unit: String,
  series: Seq[(java.sql.Timestamp, String)]
)

case class LabItem(
  item_id: Int,
  label: String,
  fluid: String,
  category: String,
  loincCode: String
)

// This is for lab_ts, which inexplicably has a NullPointerException
// if I instead use a 4-tuple of identical fields.
case class LabWrapper(a1 : Int, a2 : Int, a3 : Int, a4 : String)

object Main {
  def main(args: Array[String]) {
    val spark = Utils.createContext
    import spark.implicits._

    import org.apache.log4j.Logger
    import org.apache.log4j.Level

    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)

    // These are small enough to cache:
    val d_icd_diagnoses = Utils.csv_from_s3(
      spark, "D_ICD_DIAGNOSES", Some(Schemas.d_icd_diagnoses))
    d_icd_diagnoses.cache()

    val d_labitems = Utils.csv_from_s3(
      spark, "D_LABITEMS", Some(Schemas.d_labitems))
    // D_LABITEMS is fairly small, so make a map of it:
    val labitemMap : Map[Int,LabItem] = d_labitems.
      rdd.map { r: Row =>
        val li = LabItem(r.getAs("ITEMID"), r.getAs("LABEL"),
          r.getAs("FLUID"), r.getAs("CATEGORY"), r.getAs("LOINC_CODE"))
        (li.item_id, li)
      }.collect.toMap

    val loinc = spark.
      read.
      format("com.databricks.spark.csv").
      option("header", "true").
      load("s3://bd4h-mimic3/LOINC_2.56_Text/loinc.csv")
    //loinc.createOrReplaceTempView("loinc")
    //loinc.cache()

    val patients = Utils.csv_from_s3(
      spark, "PATIENTS", Some(Schemas.patients))
    val labevents = Utils.csv_from_s3(
      spark, "LABEVENTS", Some(Schemas.labevents))
    // For DIAGNOSES_ICD, also get the ICD9 category (which we reuse
    // in various places):
    val diagnoses_icd = Utils.csv_from_s3(
      spark, "DIAGNOSES_ICD", Some(Schemas.diagnoses)).
      withColumn("ICD9_CATEGORY", $"ICD9_CODE".substr(0, 3))

    // What is the minimum length (number of samples/events) in a
    // time-series that we'll consider for a given admission & item?
    val lab_min_series = 50
    val lab_min_patients = 30

    // Two ICD codes; we want one or the other, but not both.
    val icd_code1 = "518"
    val icd_code2 = "584"

    // Get (HADM_ID, ITEM_ID) for those admissions and lab items which
    // meet 'lab_min_series':
    val labs_length_ok : DataFrame = labevents.
      groupBy("HADM_ID", "ITEMID").
      // How long is this time-series for given admission & lab item?
      count.
      filter($"count" >= lab_min_series).
      select("HADM_ID", "ITEMID")

    // Get those admissions which had >= 1 diagnosis of icd_code1, or
    // of icd_code2, but not diagnoses of both.
    val diag_cohort : DataFrame = diagnoses_icd.
      withColumn("is_code1", ($"ICD9_CATEGORY" === icd_code1).cast(IntegerType)).
      withColumn("is_code2", ($"ICD9_CATEGORY" === icd_code2).cast(IntegerType)).
      groupBy("HADM_ID").
      sum("is_code1", "is_code2").
      filter(($"sum(is_code1)" > 0) =!= ($"sum(is_code2)" > 0))

    /***********************************************************************
     * Exploratory stuff
     ***********************************************************************/

    // To bypass:
    /*
    val labs_patients_ok = spark.read.parquet("s3://bd4h-mimic3/temp/labs.parquet")
    val labs_good_df = spark.read.parquet("s3://bd4h-mimic3/temp/labs_and_admissions.parquet")
    val icd9_per_pair = spark.read.parquet("s3://bd4h-mimic3/temp/icd9_per_pair.parquet")
    val pairs_per_icd9 = spark.read.parquet("s3://bd4h-mimic3/temp/pairs_per_icd9.parquet")
    val pairs_per_icd9_category = spark.read.parquet("s3://bd4h-mimic3/temp/pairs_per_icd9_category.parquet")
    val lab_ts = spark.read.parquet("s3://bd4h-mimic3/temp/lab_timeseries.parquet")
     */

    // Get ITEM_ID for those lab items which meet both
    // 'lab_min_series' and 'lab_min_patients'.
    val labs_patients_ok : DataFrame = labs_length_ok.
      groupBy("ITEMID").
      // How many times does this item occur, given that we're
      // concerned only with length >= lab_min_series?
      count.
      filter($"count" >= lab_min_patients)

    // Finally, get all (HADM_ID, ITEM_ID) that satisfy both:
    val labs_good_df : DataFrame = labs_patients_ok.
      join(labs_length_ok, "ITEMID").
      // join is supposed to be an inner join, but whatever...
      filter(not(isnull($"HADM_ID"))).
      select("HADM_ID", "ITEMID")

    labs_patients_ok.cache()
    labs_good_df.cache()

    // For 'nice' output of just which lab items are relevant (and the
    // human-readable form):
    labs_patients_ok.
      dropDuplicates("ITEMID").
      join(d_labitems, "ITEMID").
      sort(desc("count")).
      coalesce(1).
      write.
      parquet("s3://bd4h-mimic3/temp/labs.parquet")
    // and then both labs & admissions:
    labs_good_df.
      coalesce(1).
      write.
      parquet("s3://bd4h-mimic3/temp/labs_and_admissions.parquet")

    // How many unique ICD-9 diagnoses accompany each admission & lab
    // item?
    val icd9_per_pair : DataFrame = labs_good_df.
      join(diagnoses_icd, "HADM_ID").
      groupBy("HADM_ID", "ITEMID").
      count.
      withColumnRenamed("count", "icd9_unique_count")
    icd9_per_pair.
      coalesce(1).
      write.
      parquet("s3://bd4h-mimic3/temp/icd9_per_pair.parquet")

    // How many unique (admission, lab items) accompany each ICD-9
    // code present?  ('lab item' here refers to the entire
    // time-series, not every sample from it.)
    val pairs_per_icd9 : DataFrame = labs_good_df.
      join(diagnoses_icd, "HADM_ID").
      groupBy("ICD9_CODE").
      count.
      withColumnRenamed("count", "adm_and_lab_count").
      // Make a little more human-readable:
      join(d_icd_diagnoses, "ICD9_CODE").
      sort(desc("adm_and_lab_count"))
    pairs_per_icd9.cache()
    pairs_per_icd9.
      coalesce(1).
      write.
      parquet("s3://bd4h-mimic3/temp/pairs_per_icd9.parquet")

    // How many unique (admission, lab items) accompany each ICD-9
    // *category* present?
    val pairs_per_icd9_category : DataFrame = labs_good_df.
      join(diagnoses_icd, "HADM_ID").
      groupBy("ICD9_CATEGORY").
      count.
      withColumnRenamed("count", "adm_and_lab_count").
      sort(desc("adm_and_lab_count"))
    pairs_per_icd9_category.cache()
    pairs_per_icd9_category.
      coalesce(1).
      write.
      parquet("s3://bd4h-mimic3/temp/pairs_per_icd9_category.parquet")
    
    // Get the actual time-series for the selected admissions & lab
    // items:
    val lab_ts = labs_good_df.
      join(labevents, Seq("HADM_ID", "ITEMID"))
    lab_ts.
      coalesce(1).
      write.
      parquet("s3://bd4h-mimic3/temp/lab_timeseries.parquet")

    /*

    // Get lab time-series for each: subject, admission, item
    // (i.e. which lab test), and unit (which should be identical
    // per-item).
    val lab_ts2 = labevents.
      groupByKey { r: Row =>
        LabWrapper(r.getAs[Int]("SUBJECT_ID"),
          r.getAs[Int]("HADM_ID"),
          r.getAs[Int]("ITEMID"),
          r.getAs[String]("VALUEUOM"))
      }.mapGroups { (group, rows) =>
        val series = rows.map { r =>
          (r.getAs[Timestamp]("CHARTTIME"),
            r.getAs[String]("VALUE"))
        }.toSeq
        group match {
          case LabWrapper(subj,hadm,item,uom) =>
            PatientEventSeries(subj, hadm, item, uom, series)
        }
      }
    lab_ts2.persist(StorageLevel.MEMORY_AND_DISK)
    // Above is causing NullPointerException for some reason if I
    // modify the LabWrapper to instead just be a tuple (and likewise
    // change the pattern-match to be over a tuple).  It seems to
    // happen even if I take just the groupByKey output, and if I use
    // 3 elements in the tuple rather than 4.  It goes away if I use
    // only 2 elements.
    //
    // I also observed an issue that looked something like:
    // https://issues.apache.org/jira/browse/SPARK-12063

    // This is (probably) slower, but lacks that NullPointerException:
    val lab_ts_rdd = labevents.
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
          val series = l.map { case (_,t,v) => (t,v) }.toList
          PatientEventSeries(subj, adm, item, unit, series)
      }
    lab_ts_rdd.persist(StorageLevel.MEMORY_AND_DISK)

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

    // ICD9 counts per-patient:
    val icd9counts = diagnoses_icd.
      groupBy("SUBJECT_ID", "ICD9_CODE").
      count.
      sort(desc("count")).
      join(d_icd_diagnoses, "ICD9_CODE")

    // Per-patient, per-encounter:
    val icd9counts_adm = diagnoses_icd.
      groupBy("SUBJECT_ID", "ICD9_CODE", "HADM_ID").
      count.
      join(d_icd_diagnoses, "ICD9_CODE").
      sort(desc("count"))

     */
  }

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
