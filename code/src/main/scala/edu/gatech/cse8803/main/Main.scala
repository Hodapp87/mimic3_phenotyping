package edu.gatech.cse8803.main

import edu.gatech.cse8803.util.Utils
import edu.gatech.cse8803.util.Schemas
import edu.gatech.cse8803.types._

import org.apache.spark.ml.param._
import org.apache.spark.ml.tuning._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.storage.StorageLevel
import java.sql.Timestamp
// import com.cloudera.sparkts._

object Main {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder.
      // Can I just pass this in with spark-submit?
      //master("yarn").
      master("local[*]").
      appName("CSE-8803 project").
      getOrCreate()
    val sc = spark.sparkContext
    import spark.implicits._

    // Input data directories
    val tmp_data_dir : String = "s3://bd4h-mimic3/cohort_518_584_50820/"
    val mimic3_dir : String = "s3://bd4h-mimic3/"
    //val tmp_data_dir : String = "file:///home/hodapp/source/bd4h-project/data-temp/"
    //val mimic3_dir : String = "file:////mnt/dev/mimic3/"

    // TODO: Perhaps pass the above in as commandline options

    import org.apache.log4j.Logger
    import org.apache.log4j.Level

    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)

    // These are small enough to cache:
    val d_icd_diagnoses = Utils.csv_from_s3(
      spark, f"${mimic3_dir}/D_ICD_DIAGNOSES.csv.gz", Some(Schemas.d_icd_diagnoses))
    d_icd_diagnoses.cache()

    val d_labitems = Utils.csv_from_s3(
      spark, f"${mimic3_dir}/D_LABITEMS.csv.gz", Some(Schemas.d_labitems))
    // D_LABITEMS is fairly small, so make a map of it:
    val labitemMap : Map[Int, LabItem] = d_labitems.
      rdd.map { r: Row =>
        val li = LabItem(r.getAs("ITEMID"), r.getAs("LABEL"),
          r.getAs("FLUID"), r.getAs("CATEGORY"), r.getAs("LOINC_CODE"))
        (li.item_id, li)
      }.collect.toMap

    val loinc = spark.
      read.
      format("com.databricks.spark.csv").
      option("header", "true").
      load(f"${mimic3_dir}/LOINC_2.56_Text/loinc.csv")
    //loinc.createOrReplaceTempView("loinc")
    //loinc.cache()

    val patients = Utils.csv_from_s3(
      spark, f"${mimic3_dir}/PATIENTS.csv.gz", Some(Schemas.patients))
    val labevents = Utils.csv_from_s3(
      spark, f"${mimic3_dir}/LABEVENTS.csv.gz", Some(Schemas.labevents))
    // For DIAGNOSES_ICD, also get the ICD9 category (which we reuse
    // in various places):
    val diagnoses_icd = Utils.csv_from_s3(
      spark, f"${mimic3_dir}/DIAGNOSES_ICD.csv.gz", Some(Schemas.diagnoses)).
      withColumn("ICD9_CATEGORY", $"ICD9_CODE".substr(0, 3))

    // What is the minimum length (number of samples/events) in a
    // time-series that we'll consider for a given admission & item?
    val lab_min_series = 3
    val lab_min_patients = 30

    // Two ICD codes; we want one or the other, but not both.
    val icd_code1 = "518"
    val icd_code2 = "584"

    // ITEMID of the test we're interested in:
    val item_test = 50820

    // Get (HADM_ID, ITEM_ID) for those admissions and lab items which
    // meet 'lab_min_series':
    val labs_length_ok : DataFrame = labevents.
      groupBy("HADM_ID", "ITEMID").
      // How long is this time-series for given admission & lab item?
      count.
      filter($"count" >= lab_min_series).
      select("HADM_ID", "ITEMID")

    // To bypass:
    /*
    val diag_cohort = spark.read.parquet(f"${tmp_data_dir}/diag_cohort_${icd_code1}_${icd_code2}.parquet")
    val labs_cohort : RDD[Schemas.PatientEventSeries] = sc.
      objectFile(f"${tmp_data_dir}/labs_cohort_${icd_code1}_${icd_code2}_rdd")
    val labs_cohort_flat = spark.read.parquet(f"${tmp_data_dir}/labs_cohort_${icd_code1}_${icd_code2}.parquet")
     */
  
    // Get those admissions which had >= 1 diagnosis of icd_code1, or
    // of icd_code2, but not diagnoses of both.
    val diag_cohort : DataFrame = diagnoses_icd.
      withColumn("is_code1", ($"ICD9_CATEGORY" === icd_code1).cast(IntegerType)).
      withColumn("is_code2", ($"ICD9_CATEGORY" === icd_code2).cast(IntegerType)).
      groupBy("HADM_ID").
      sum("is_code1", "is_code2").
      withColumnRenamed("sum(is_code1)", "num_code1").
      withColumnRenamed("sum(is_code2)", "num_code2").
      filter(($"num_code1" > 0) =!= ($"num_code2" > 0))
    diag_cohort.cache()
    diag_cohort.write.parquet(f"${tmp_data_dir}/diag_cohort_${icd_code1}_${icd_code2}.parquet")

    // Get the lab events which meet 'lab_min_series', which are from
    // an admission in the cohort, and which are of the desired test.
    val labs_cohort_df : DataFrame = labs_length_ok.
      join(labevents, Seq("HADM_ID", "ITEMID")).
      filter($"ITEMID" === item_test).
      join(diag_cohort, Seq("HADM_ID"))

    // Produce time-series, and warped time-series, for all admissions
    // in the cohort (for just the selected lab items):
    val labs_cohort : RDD[PatientEventSeries] = labs_cohort_df.rdd.
      map { row =>
        val k = (row.getAs[Int]("HADM_ID"),
          row.getAs[Int]("ITEMID"),
          row.getAs[Int]("SUBJECT_ID"),
          row.getAs[String]("VALUEUOM"))
        // Grouping on all of the above might be superfluous.  Unique admission
        val v = (row.getAs[Timestamp]("CHARTTIME").getTime / 86400000.0,
          row.getAs[Double]("VALUENUM"))
        (k, v)
      }.groupByKey.map { case ((adm, item, subj, uom), series_raw) =>
          // Separate out times and values, and pass forward both
          // "original" time series and warped time series:
          val series = series_raw.toSeq.sortBy(_._1)
          val (times, values) = series.unzip
          val start = times.min
          val relTimes = times.map(_ - start)
          val warpedTimes = Utils.polynomialTimeWarp(relTimes.toSeq)
          //val warpedTimes = relTimes
          PatientEventSeries(
            adm,
            item,
            subj,
            uom,
            relTimes.zip(values),
            warpedTimes.zip(values)
          )
      }
    labs_cohort.persist(StorageLevel.MEMORY_AND_DISK)
    labs_cohort.
      coalesce(1).
      saveAsObjectFile(f"${tmp_data_dir}/labs_cohort_${icd_code1}_${icd_code2}_rdd")    // Yes, this coalesce(1) is bad practice.

    // Flatten out to load elsewhere:
    val labs_cohort_flat : DataFrame =
      labs_cohort.flatMap { p: PatientEventSeries =>
        val ts = p.series.zip(p.warpedSeries)
        ts.map { case ((t, _), (tw, value)) =>
          (p.adm_id, p.item_id, p.subject_id, p.unit, t, tw, value) 
        }
      }.toDF("HADM_ID", "ITEMID", "SUBJECT_ID", "VALUEUOM", "CHARTTIME", "CHARTTIME_warped", "VALUENUM")
    labs_cohort_flat.
      coalesce(1).
      write.parquet(f"${tmp_data_dir}/labs_cohort_${icd_code1}_${icd_code2}.parquet")
    labs_cohort_flat.
      coalesce(1).
      write.
      format("com.databricks.spark.csv").
      option("header", "true").
      save(f"${tmp_data_dir}/labs_cohort_${icd_code1}_${icd_code2}.csv")

    // Hyperparameter optimization:
    val sigma2Param = new DoubleParam("", "sigma2", "")
    val alphaParam = new DoubleParam("", "alpha", "")
    val tauParam = new DoubleParam("", "tau", "")

    val paramGrid = new ParamGridBuilder().
      addGrid(sigma2Param, 0.1 to 2.0 by 0.1).
      addGrid(alphaParam, 0.05 to 0.5 by 0.05).
      addGrid(tauParam, 1.5 to 4.0 by 0.05).
      build.
      map { pm =>
        (pm.get(sigma2Param).get,
          pm.get(alphaParam).get,
          pm.get(tauParam).get)
        // The .get is intentional; it *should* throw an exception if
        // it can't find the parameter this early.
      }

    val paramRdd : RDD[(Double, Double, Double)] = sc.parallelize(paramGrid)

    // TODO: Re-run this tonight (commented result is for longer
    // series only)
    if (false) {
      val labs_ll : RDD[((Double, Double, Double), Double)] = paramRdd.
        cartesian(labs_cohort).
        map { case (t@(sigma2, alpha, tau), series) =>
          (t, Utils.gprTrain(series.warpedSeries, sigma2, alpha, tau)._1)
        }.foldByKey(0.0)(_ + _)
      labs_ll.cache()

      // optimal: ((Double, Double, Double), Double) =
      // ((1.5000000000000002,0.125,3.2499999999999964),-109216.21495499206)
      val optimal = labs_ll.
        aggregate((0.0, 0.0, 0.0), Double.NegativeInfinity)(
          { case(_, t) => t },
          { (t1,t2) => if (t1._2 > t2._2) t1 else t2 }
        )
      // Fucking Spark, would it kill you to provide an argmax/argmin
      // function?
    }

    // Regression example:
    val sigma2 = 1.5
    val alpha = 0.125
    val tau = 3.25
    val labs_cohort_split = labs_cohort.randomSplit(Array(0.7, 0.3), 0x12345)
    val labs_cohort_train = labs_cohort_split(0)
    val gprModels = labs_cohort_train.map { p: PatientEventSeries =>
      // Train a model for every time-series in training set:
      val t@(ll, matL, matA) = Utils.gprTrain(p.warpedSeries, sigma2, alpha, tau)

      t
    }
   

    if (false) {

      /***********************************************************************
       * Exploratory stuff
       ***********************************************************************/

      // To bypass:
      /*
       val labs_patients_ok = spark.read.parquet(f"${tmp_data_dir}/labs.parquet")
       val labs_good_df = spark.read.parquet(f"${tmp_data_dir}/labs_and_admissions.parquet")
       val icd9_per_pair = spark.read.parquet(f"${tmp_data_dir}/icd9_per_pair.parquet")
       val pairs_per_icd9 = spark.read.parquet(f"${tmp_data_dir}/pairs_per_icd9.parquet")
       val pairs_per_icd9_category = spark.read.parquet(f"${tmp_data_dir}/pairs_per_icd9_category.parquet")
       val lab_ts = spark.read.parquet(f"${tmp_data_dir}/lab_timeseries.parquet")
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
        parquet(f"${tmp_data_dir}/labs.parquet")
      // and then both labs & admissions:
      labs_good_df.
        coalesce(1).
        write.
        parquet(f"${tmp_data_dir}/labs_and_admissions.parquet")

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
        parquet(f"${tmp_data_dir}/icd9_per_pair.parquet")

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
        parquet(f"${tmp_data_dir}/pairs_per_icd9.parquet")

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
        parquet(f"${tmp_data_dir}/pairs_per_icd9_category.parquet")
    }
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
