// (c) 2016 Chris Hodapp, chodapp3@gatech.edu

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
import org.apache.spark.util._
import scopt._

import java.sql.Timestamp

case class Config(
  mimicInput : String = null,
  outputPath : String = null,
  icdMatrix : Boolean = false,
  computeCohort : Boolean = false,
  icd9Code1 : String = null,
  icd9Code2 : String = null,
  loincTest : String = null
)

object Main {
  def main(args: Array[String]): Unit = {

    // TODO: Rename "scopt"
    val parser = new OptionParser[Config]("scopt") {
      head("scopt", "3.x")
      opt[String]('i', "mimic_input").required().action { (x,c) =>
        c.copy(mimicInput = x)
      }.text("Path to the MIMIC-III datasets (.csv.gz); use file:/// for local paths")
      opt[String]('o', "output_path").required().action { (x,c) =>
        c.copy(outputPath = x)
      }.text("Directory to write output (must already exist); use file:/// for local paths")
      opt[Unit]('m', "write_matrix").optional().action { (x,c) =>
        c.copy(icdMatrix = true)
      }.text("Generate matrix of top ICD9 categories & LOINC IDs")
      opt[Unit]('c', "cohort").optional().action { (_,c) =>
        c.copy(computeCohort = true)
      }.text("Generate cohort dataset from given ICD9 codes & LOINC IDs").
        children(
          opt[String]("icd9a").required().action { (x,c) =>
            c.copy(icd9Code1 = x)
          }.text("First ICD9 code to select for in cohort"),
          opt[String]("icd9b").required().action { (x,c) =>
            c.copy(icd9Code2 = x)
          }.text("Second ICD9 code to select for in cohort"),
          opt[String]('l', "loinc").required().action { (x,c) =>
            c.copy(loincTest = x)
          }.text("LOINC ID to select which lab test to use")
        )
    }

    parser.parse(args, Config()) map { config =>
      run(config)
    } getOrElse {
      // Do nothing?
    }

  }

  def run(config : Config) : Unit = {
    /***********************************************************************
     * Boilerplate
     ***********************************************************************/
    val spark = SparkSession.builder.
      //master("yarn").
      //master("local[*]").
      appName("CSE-8803 project").
      getOrCreate()
    val sc = spark.sparkContext
    import spark.implicits._
    import org.apache.log4j.Logger
    import org.apache.log4j.Level

    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)

    /***********************************************************************
     * Run parameters
     ***********************************************************************/
    //val genMatrix = true
    val computeLabs = false
    val optimizeHyperparams = false
    val regression = false
    // What is the minimum length (number of samples/events) in a
    // time-series that we'll consider for a given admission & item?
    val lab_min_series = 3
    val lab_min_patients = 30

    // Two ICD codes; we want one or the other, but not both.
    // val icd_code1 = "518"
    // val icd_code2 = "584"
    val icd_code1 = "276"
    val icd_code2 = "427"

    // ITEMID of the test we're interested in:
    // val item_test = 50820
    val item_test = 51268

    // For training & test split:
    val randomSeed : Long = 0x12345
    val trainRatio = 0.7

    // TODO: Perhaps pass the above in as commandline options

    /***********************************************************************
     * Loading & transforming data
     ***********************************************************************/

    // These are small enough to cache:
    val d_icd_diagnoses = Utils.csv_from_s3(
      spark, f"${config.mimicInput}/D_ICD_DIAGNOSES.csv.gz", Some(Schemas.d_icd_diagnoses))
    d_icd_diagnoses.cache()

    val d_labitems = Utils.csv_from_s3(
      spark, f"${config.mimicInput}/D_LABITEMS.csv.gz", Some(Schemas.d_labitems))
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
      load(f"${config.mimicInput}/LOINC_2.56_Text/loinc.csv")
    //loinc.createOrReplaceTempView("loinc")
    //loinc.cache()

    val patients = Utils.csv_from_s3(
      spark, f"${config.mimicInput}/PATIENTS.csv.gz", Some(Schemas.patients))
    val labevents = Utils.csv_from_s3(
      spark, f"${config.mimicInput}/LABEVENTS.csv.gz", Some(Schemas.labevents))
    // For DIAGNOSES_ICD, also get the ICD9 category (which we reuse
    // in various places):
    val diagnoses_icd = Utils.csv_from_s3(
      spark, f"${config.mimicInput}/DIAGNOSES_ICD.csv.gz", Some(Schemas.diagnoses)).
      withColumn("ICD9_CATEGORY", $"ICD9_CODE".substr(0, 3))

    // Get (HADM_ID, ITEM_ID) for those admissions and lab items which
    // meet 'lab_min_series':
    val labs_length_ok : DataFrame = labevents.
      groupBy("HADM_ID", "ITEMID").
      // How long is this time-series for given admission & lab item?
      count.
      filter($"count" >= lab_min_series).
      select("HADM_ID", "ITEMID")

    if (config.icdMatrix) {

      // To bypass:
      /*
       val labs_patients_ok = spark.read.parquet(f"${config.outputPath}/labs.parquet")
       val labs_good_df = spark.read.parquet(f"${config.outputPath}/labs_and_admissions.parquet")
       val icd9_per_pair = spark.read.parquet(f"${config.outputPath}/icd9_per_pair.parquet")
       val pairs_per_icd9 = spark.read.parquet(f"${config.outputPath}/pairs_per_icd9.parquet")
       val pairs_per_icd9_category = spark.read.parquet(f"${config.outputPath}/pairs_per_icd9_category.parquet")
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

      // Select only the top 'item_count' items for the matrix.  (Why:
      // We're doing a pivot on this - sort of, it's on LOINC code -
      // and we want to limit how many columns this will produce.)
      val item_count = 100

      // How many ICD-9 categories do we want, taken from the most
      // common?  (We don't pivot on this, so it has no such
      // limitation as does item_count.)
      val icd9_count = 125

      val item_parquet = f"${config.outputPath}/items_limit.parquet"
      val item_csv = f"${config.outputPath}/items_limit.csv"
      val mtx_parquet = f"${config.outputPath}/icd9_item_matrix.parquet"
      val mtx_csv = f"${config.outputPath}/icd9_item_matrix.csv"

      // Select the top 'item_count' items and write them to a file.
      // These are used later too.
      val items_limit : DataFrame = labs_patients_ok.
        sort(desc("count")).
        limit(item_count).
        join(d_labitems, "ITEMID")
      items_limit.cache()
      items_limit.
        write.
        mode(SaveMode.Overwrite).
        parquet(item_parquet)
      Utils.csvOverwrite(items_limit).
        save(item_csv)

      // Get ITEMID & LOINC code from the top 100, associate these
      // with admissions and the ICD-9 categories of each, and then
      // pivot to create a new column for each LOINC code (top
      // 'item_count' of them at least), and have each row count up
      // the number of occurrences of each ICD-9 category for each
      // LOINC code.
      val icd9_item_matrix_raw : DataFrame = items_limit.
        select("ITEMID", "LOINC_CODE").
        filter(not(isnull($"LOINC_CODE"))).
        join(labs_good_df, "ITEMID").
        join(diagnoses_icd, "HADM_ID").
        groupBy("ICD9_CATEGORY").
        pivot("LOINC_CODE").
        count.
        na.fill(0)
      // Sum across each row, and order by that in order to take just
      // 'icd9_count' rows:
      val icd9_item_matrix : DataFrame = icd9_item_matrix_raw.
        withColumn("sum",
          icd9_item_matrix_raw.columns.tail.map(col).reduce(_+_)).
        sort(desc("sum")).
        limit(icd9_count)
      icd9_item_matrix.persist(StorageLevel.MEMORY_AND_DISK)
      icd9_item_matrix.
        write.
        mode(SaveMode.Overwrite).
        parquet(mtx_parquet)
      Utils.csvOverwrite(icd9_item_matrix).
        save(mtx_csv)

      println(f"Wrote top item list to: ${item_csv}, ${item_parquet}")
      println(f"Wrote top ICD-9 / LOINC matrix to: ${mtx_csv}, ${mtx_parquet}")
    }

    // Suffix to append to filenames of cohort-derived data:
    val suffix : String = f"${icd_code1}_${icd_code2}_${item_test}"

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
          parquet(f"${config.outputPath}/diag_cohort_${suffix}.parquet")

        // Also write a more externally-usable form:
        val diag_cohort_categories : DataFrame = diag_cohort.
          withColumn("ICD9_CATEGORY",
            when($"num_code1" > 0, icd_code1).
              when($"num_code2" > 0, icd_code2)).
          select("HADM_ID", "ICD9_CATEGORY")
        Utils.csvOverwrite(diag_cohort_categories).
          save(f"${config.outputPath}/diag_cohort_categories_${suffix}.csv")

        // Get the lab events which meet 'lab_min_series', which are from
        // an admission in the cohort, and which are of the desired test.
        val labs_cohort_df : DataFrame = labs_length_ok.
          join(labevents, Seq("HADM_ID", "ITEMID")).
          filter($"ITEMID" === item_test).
          join(diag_cohort, Seq("HADM_ID")).
          na.fill("", Seq("VALUEUOM"))

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
              val (times, values2) = series.unzip
              // Subtract mean from values:
              // TODO: This should really be done elsewhere
              val ymean = values2.sum / values2.size
              val values = values2.map(_ - ymean)
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
          saveAsObjectFile(f"${config.outputPath}/labs_cohort_${suffix}_rdd")

        // Flatten out to load elsewhere:
        val labs_cohort_flat : DataFrame = Utils.flattenTimeseries(spark, labs_cohort)
        labs_cohort_flat.
          write.
          mode(SaveMode.Overwrite).
          parquet(f"${config.outputPath}/labs_cohort_${suffix}.parquet")
        Utils.csvOverwrite(labs_cohort_flat).
          save(f"${config.outputPath}/labs_cohort_${suffix}.csv")
        (diag_cohort, labs_cohort, labs_cohort_flat)
      } else {
        println("Loading saved data for diag & labs...")
        val diag_cohort = spark.read.parquet(f"${config.outputPath}/diag_cohort_${suffix}.parquet")
        val labs_cohort : RDD[PatientTimeSeries] = sc.
          objectFile(f"${config.outputPath}/labs_cohort_${suffix}_rdd")
        val labs_cohort_flat = spark.read.parquet(f"${config.outputPath}/labs_cohort_${suffix}.parquet")
        (diag_cohort, labs_cohort, labs_cohort_flat)
      }

    // Separate training & test:
    val labs_cohort_split = labs_cohort.map { ps: PatientTimeSeries =>
      ((ps.adm_id, ps.item_id, ps.unit), ps)
    }.sortByKey(true).
      map(_._2).
      randomSplit(
      Array(trainRatio, 1.0 - trainRatio), randomSeed)
    val labs_cohort_train = labs_cohort_split(0)
    val labs_cohort_test  = labs_cohort_split(1)

    // For some reason I'm having to write this every time, as if
    // labs_cohort_train/labs_cohort_test are non-deterministic.

    // Save training & test to disk (they'll be needed later):
    // if (computeLabs)
    {
        val train_flat : DataFrame = Utils.flattenTimeseries(spark, labs_cohort_train)
        Utils.csvOverwrite(train_flat).
          save(f"${config.outputPath}/labs_cohort_train_${suffix}.csv")
        val test_flat : DataFrame = Utils.flattenTimeseries(spark, labs_cohort_test)
        Utils.csvOverwrite(test_flat).
          save(f"${config.outputPath}/labs_cohort_test_${suffix}.csv")
    }

    /***********************************************************************
     * Hyperparameter optimization for GPR
     ***********************************************************************/
    if (optimizeHyperparams) {

      // Hyperparameter optimization:
      val sigma2Param = new DoubleParam("", "sigma2", "")
      val alphaParam = new DoubleParam("", "alpha", "")
      val tauParam = new DoubleParam("", "tau", "")

      // sigma2,alpha,tau,log_likelihood,lab_min_series,item_test
      // 0.1,0.15000000000000002,1.5000000000000007,482.8546422203833,3,50820

      // sigma2,alpha,tau,log_likelihood,lab_min_series,item_test
      // 0.01,0.18000000000000002,1.25,116812.48998898288,3,50820

      // sigma2,alpha,tau,log_likelihood,lab_min_series,item_test
      // 0.012000000000000004,0.18900000000000003,1.2459999999999993,117664.8588038926,3,50820

      val paramGrid : Array[(Double, Double, Double)] = new ParamGridBuilder().
        addGrid(sigma2Param, 0.05 to 0.5 by 0.05).
        addGrid(alphaParam, 0.05 to 1.00 by 0.05).
        addGrid(tauParam, 0.2 to 2.0 by 0.05).
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
      // Spark, would providing an argmax/argmin function kill you?

      // Quick hack to write the hyperparameters to disk:
      val hyperDf = sc.parallelize(Seq(optimal)).
        map { case ((sig2, a, t), ll) => (sig2, a, t, ll, lab_min_series, item_test) }.
        toDF("sigma2", "alpha", "tau", "log_likelihood", "lab_min_series", "item_test")

      Utils.csvOverwrite(hyperDf).
        save(f"${config.outputPath}/hyperparams_${suffix}.csv")
    }

    /***********************************************************************
     * Gaussian process regression
     ***********************************************************************/
    if (regression) {
      // Perform regression over training data:
      val sigma2 = 0.012
      val alpha = 0.189
      val tau = 1.246
      // TODO: Pull these out to commandline options?  Or something
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
      val padding = 2.5
      // What interval (in days) do we interpolate with?
      val interval = 0.25
      // TODO: Make these commandline options too?

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
        parquet(f"${config.outputPath}/labs_cohort_predict_${suffix}.parquet")
      Utils.csvOverwrite(tsInterp_flat).
        save(f"${config.outputPath}/labs_cohort_predict_${suffix}.csv")
    }
  }


  // Some scratch stuff I no longer use:
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

  /*

      /*
      // For 'nice' output of just which lab items are relevant (and the
      // human-readable form):
      labs_patients_ok.
        dropDuplicates("ITEMID").
        join(d_labitems, "ITEMID").
        sort(desc("count")).
        write.
        mode(SaveMode.Overwrite).
        parquet(f"${config.outputPath}/labs.parquet")
      // and then both labs & admissions:
      labs_good_df.
        write.
        mode(SaveMode.Overwrite).
        parquet(f"${config.outputPath}/labs_and_admissions.parquet")
       */

      /*
      // How many unique ICD-9 diagnoses accompany each admission & lab
      // item?
      val icd9_per_pair : DataFrame = labs_good_df.
        join(diagnoses_icd, "HADM_ID").
        groupBy("HADM_ID", "ITEMID").
        count.
        withColumnRenamed("count", "icd9_unique_count")
      icd9_per_pair.
        write.
        mode(SaveMode.Overwrite).
        parquet(f"${config.outputPath}/icd9_per_pair.parquet")

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
        parquet(f"${config.outputPath}/pairs_per_icd9.parquet")

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
        mode(SaveMode.Overwrite).
        parquet(f"${config.outputPath}/pairs_per_icd9_category.parquet")
       */
   */
}
