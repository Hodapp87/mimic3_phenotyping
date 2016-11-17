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
    //val tmp_data_dir : String = "s3://bd4h-mimic3/cohort_518_584_50820/"
    //val mimic3_dir : String = "s3://bd4h-mimic3/"
    val tmp_data_dir : String = "file:///home/hodapp/source/bd4h-project/data-temp/"
    val mimic3_dir : String = "file:////mnt/dev/mimic3/"

    val computeLabs = false
    val optimizeHyperparams = false
    val regression = true
    // What is the minimum length (number of samples/events) in a
    // time-series that we'll consider for a given admission & item?
    val lab_min_series = 3
    val lab_min_patients = 30

    // Two ICD codes; we want one or the other, but not both.
    val icd_code1 = "518"
    val icd_code2 = "584"

    // ITEMID of the test we're interested in:
    val item_test = 50820

    // For training & test split:
    val randomSeed : Long = 0x12345
    val trainRatio = 0.7

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

    // Get (HADM_ID, ITEM_ID) for those admissions and lab items which
    // meet 'lab_min_series':
    val labs_length_ok : DataFrame = labevents.
      groupBy("HADM_ID", "ITEMID").
      // How long is this time-series for given admission & lab item?
      count.
      filter($"count" >= lab_min_series).
      select("HADM_ID", "ITEMID")

    val (diag_cohort, labs_cohort, labs_cohort_flat) =
      if (computeLabs) {
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
        diag_cohort.
          write.
          mode(SaveMode.Overwrite).
          parquet(f"${tmp_data_dir}/diag_cohort_${icd_code1}_${icd_code2}.parquet")

        // Get the lab events which meet 'lab_min_series', which are from
        // an admission in the cohort, and which are of the desired test.
        val labs_cohort_df : DataFrame = labs_length_ok.
          join(labevents, Seq("HADM_ID", "ITEMID")).
          filter($"ITEMID" === item_test).
          join(diag_cohort, Seq("HADM_ID"))

        // Produce time-series, and warped time-series, for all admissions
        // in the cohort (for just the selected lab items):
        val labs_cohort : RDD[PatientTimeSeries] = labs_cohort_df.rdd.
          map { row =>
            val code1 = row.getAs[Long]("num_code1") > 0
            val code2 = row.getAs[Long]("num_code2") > 0
            val k = (row.getAs[Int]("HADM_ID"),
              row.getAs[Int]("ITEMID"),
              row.getAs[Int]("SUBJECT_ID"),
              row.getAs[String]("VALUEUOM"),
              if (code1) icd_code1 else { if (code2) icd_code2 else "" })
            // Grouping on all of the above might be superfluous.  Unique admission
            // implies unique subject.
            val v = (row.getAs[Timestamp]("CHARTTIME").getTime / 86400000.0,
              row.getAs[Double]("VALUENUM"))
            (k, v)
          }.groupByKey.map { case ((adm, item, subj, uom, code), series_raw) =>
              // Separate out times and values, and pass forward both
              // "original" time series and warped time series:
              val series = series_raw.toSeq.sortBy(_._1)
              val (times, values) = series.unzip
              val start = times.min
              val relTimes = times.map(_ - start)
              val warpedTimes = Utils.polynomialTimeWarp(relTimes.toSeq)
              //val warpedTimes = relTimes
              PatientTimeSeries(
                adm,
                item,
                subj,
                uom,
                code,
                relTimes.zip(values),
                warpedTimes.zip(values)
              )
          }
        // TODO: Get rid of groupByKey above and see if it helps
        // performance.
        // I removed coalesce below, but haven't tested that yet.
        // It seems
        labs_cohort.persist(StorageLevel.MEMORY_AND_DISK)
        labs_cohort.
          saveAsObjectFile(f"${tmp_data_dir}/labs_cohort_${icd_code1}_${icd_code2}_rdd")

        // Flatten out to load elsewhere:
        val labs_cohort_flat : DataFrame = Utils.flattenTimeseries(spark, labs_cohort)
        labs_cohort_flat.
          write.
          mode(SaveMode.Overwrite).
          parquet(f"${tmp_data_dir}/labs_cohort_${icd_code1}_${icd_code2}.parquet")
        Utils.csvOverwrite(labs_cohort_flat).
          save(f"${tmp_data_dir}/labs_cohort_${icd_code1}_${icd_code2}.csv")
        (diag_cohort, labs_cohort, labs_cohort_flat)
      } else {
        println("Loading saved data for diag & labs...")
        val diag_cohort = spark.read.parquet(f"${tmp_data_dir}/diag_cohort_${icd_code1}_${icd_code2}.parquet")
        val labs_cohort : RDD[PatientTimeSeries] = sc.
          objectFile(f"${tmp_data_dir}/labs_cohort_${icd_code1}_${icd_code2}_rdd")
        val labs_cohort_flat = spark.read.parquet(f"${tmp_data_dir}/labs_cohort_${icd_code1}_${icd_code2}.parquet")
        (diag_cohort, labs_cohort, labs_cohort_flat)
      }

    // Separate training & test:
    val labs_cohort_split = labs_cohort.randomSplit(
      Array(trainRatio, 1.0 - trainRatio), randomSeed)
    val labs_cohort_train = labs_cohort_split(0)
    val labs_cohort_test  = labs_cohort_split(1)

    // Save training & test to disk (they'll be needed later):
    if (computeLabs) {
        val train_flat : DataFrame = Utils.flattenTimeseries(spark, labs_cohort_train)
        Utils.csvOverwrite(train_flat).
          save(f"${tmp_data_dir}/labs_cohort_train_${icd_code1}_${icd_code2}.csv")
        val test_flat : DataFrame = Utils.flattenTimeseries(spark, labs_cohort_train)
        Utils.csvOverwrite(test_flat).
          save(f"${tmp_data_dir}/labs_cohort_test_${icd_code1}_${icd_code2}.csv")
    }

    if (optimizeHyperparams) {

      // Hyperparameter optimization:
      val sigma2Param = new DoubleParam("", "sigma2", "")
      val alphaParam = new DoubleParam("", "alpha", "")
      val tauParam = new DoubleParam("", "tau", "")

      val paramGrid : Array[(Double, Double, Double)] = new ParamGridBuilder().
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
      
      val paramGridVar = sc.broadcast(paramGrid)

      // Why is this flatMap seemingly only using a single node?
      // Is it the coalesce() call?  How many partitions exist?
      // Is it the groupByKey above?
      val labs_ll : RDD[((Double, Double, Double), Double)] = 
        labs_cohort.flatMap { series =>
          val s = series.warpedSeries
          paramGridVar.value.map { case t@(sigma2, alpha, tau) =>
            (t, Utils.gprTrain(s, sigma2, alpha, tau)._1)
          }
        }.reduceByKey(_ + _)

      val optimal = labs_ll.
        aggregate((0.0, 0.0, 0.0), Double.NegativeInfinity)(
          { case(_, t) => t },
          { (t1,t2) => if (t1._2 > t2._2) t1 else t2 }
        )
      // Fucking Spark, would it kill you to provide an argmax/argmin
      // function?

      // Quick hack to write the hyperparameters to disk:
      val hyperDf = sc.parallelize(Seq(optimal)).
        map { case ((sig2, a, t), ll) => (sig2, a, t, ll, lab_min_series, item_test) }.
        toDF("sigma2", "alpha", "tau", "log_likelihood", "lab_min_series", "item_test")

      Utils.csvOverwrite(hyperDf).
        save(f"${tmp_data_dir}/hyperparams_${icd_code1}_${icd_code2}.csv")
    }

    if (regression) {
      // Perform regression over training data:
      val sigma2 = 1.4
      val alpha = 0.1
      val tau = 3.55
      val gprModels = labs_cohort_train.map { p: PatientTimeSeries =>
        // Train a model for every time-series in training set:
        val t@(ll, matL, matA) = Utils.gprTrain(p.warpedSeries, sigma2, alpha, tau)

        (p, matL, matA)
      }
      // gprModels then has (PatientTimeSeries, L matrix, A matrix) for
      // every *training* time-series.

      // Now, generate 'predicted' time-series over a sampling of the
      // time range.

      // How many days before & after do we interpolate for?
      val padding = 5.0
      // What interval (in days) do we interpolate with?
      val interval = 0.5
      // Create a new time-series with these predictions:
      val tsInterpolated = gprModels.map { case (p, matL, matA) =>
        val ts = p.warpedSeries.map(_._1)
        val ts2 = (ts.min - padding) to (ts.max + padding) by interval
        val predictions = Utils.gprPredict(ts2, ts, matL, matA, sigma2, alpha, tau)
        PatientTimeSeriesPredicted(p.adm_id, p.item_id, p.subject_id, p.unit,
          p.icd9category,
          (ts2, predictions.map(_._1), predictions.map(_._2)).zipped.toList)
      }

      val tsInterp_flat : DataFrame = Utils.flattenPredictedTimeseries(spark, tsInterpolated)
      tsInterp_flat.
        write.
        mode(SaveMode.Overwrite).
        parquet(f"${tmp_data_dir}/labs_cohort_predict_${icd_code1}_${icd_code2}.parquet")
      Utils.csvOverwrite(tsInterp_flat).
        save(f"${tmp_data_dir}/labs_cohort_predict_${icd_code1}_${icd_code2}.csv")
    }

    /***********************************************************************
     * Exploratory stuff I don't use right now
     ***********************************************************************/
    if (false) {

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
        write.
        parquet(f"${tmp_data_dir}/labs.parquet")
      // and then both labs & admissions:
      labs_good_df.
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
