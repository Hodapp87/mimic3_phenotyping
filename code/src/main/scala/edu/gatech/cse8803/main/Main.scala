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
  mimicInput : String = "",
  outputPath : String = "",
  icdMatrix : Boolean = false,
  loincCount : Int = 100,
  icd9Count : Int = 125,
  computeCohort : Boolean = false,
  optimizeGPR : Boolean = false,
  runGPR : Boolean = false,
  icd9Code1 : String = "",
  icd9Code2 : String = "",
  loincTest : String = "",
  suffix : String = ""
)

object Main {
  def main(args: Array[String]): Unit = {

    val parser = new OptionParser[Config]("mimic3_phenotyping") {
      head("mimic3_phenotyping", "1.0")
      opt[String]('i', "mimic_input").required().action { (x,c) =>
        c.copy(mimicInput = x)
      }.text("Path to the MIMIC-III datasets (.csv.gz); use file:/// for local paths")

      opt[String]('o', "output_path").required().action { (x,c) =>
        c.copy(outputPath = x)
      }.text("Directory to write output (must already exist); use file:/// for local paths")

      opt[Unit]('m', "write_matrix").optional().action { (x,c) =>
        c.copy(icdMatrix = true)
      }.text("Generate matrix of top ICD9 categories & LOINC codes").
        children(
          opt[Int]("icd9count").optional().action { (x,c) =>
            c.copy(icd9Count = x)
          }.text("How many ICD9 codes to select for --write_matrix (default 125)"),
          opt[Int]("loincCount").optional().action { (x,c) =>
            c.copy(loincCount = x)
          }.text("How many LOINC IDs to select for --write_matrix (default 100); don't increase this needlessly")
        )

      opt[Unit]('c', "cohort").optional().action { (_,c) =>
        c.copy(computeCohort = true)
      }.text("Generate cohort dataset from given ICD9 codes & LOINC code (requires --icd9a, --icd9b, --loinc)")

      opt[Unit]('h', "optimizeGPR").optional().action { (_,c) =>
        c.copy(optimizeGPR = true)
      }.text("Optimize hyperparameters for GPR (requires --icd9a, --icd9b, --loinc, and to first run --cohort)")

      opt[Unit]('r', "regression").optional().action { (_,c) =>
        c.copy(runGPR = true)
      }.text("Run Gaussian process regression on cohort dataset (requires --icd9a, --icd9b, --loinc, and to first run --cohort)")

      opt[String]("icd9a").optional().action { (x,c) =>
        c.copy(icd9Code1 = x)
      }.text("First ICD9 code to select for in cohort")

      opt[String]("icd9b").optional().action { (x,c) =>
        c.copy(icd9Code2 = x)
      }.text("Second ICD9 code to select for in cohort")

      opt[String]('l', "loinc").optional().action { (x,c) =>
        c.copy(loincTest = x)
      }.text("LOINC code to select which lab test to use")

      checkConfig { c =>
        if (c.runGPR) {
          if (c.icd9Code1.isEmpty() || c.icd9Code1.isEmpty())
            failure("Must specify both ICD9 codes for regression")
          else if (c.loincTest.isEmpty())
            failure("Must specify LOINC code for regression")
          else success
        } else if (c.computeCohort) {
          if (c.icd9Code1.isEmpty() || c.icd9Code1.isEmpty())
            failure("Must specify both ICD9 codes to generate cohort dataset")
          else if (c.loincTest.isEmpty())
            failure("Must specify LOINC code to generate cohort dataset")
          else success
        } else if (!c.icdMatrix && !c.optimizeGPR) {
          failure("Must give at least one command (generate matrix, generate cohort, optimize hyperparams, regression)")
        } else {
          success
        }
      }
    }

    parser.parse(args, Config()) map { config_ =>

      val config = config_.copy(suffix =
        f"cohort_${config_.icd9Code1}_${config_.icd9Code2}_${config_.loincTest}")

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
    // What is the minimum length (number of samples/events) in a
    // time-series that we'll consider for a given admission & item?
    val lab_min_series = 3

    /***********************************************************************
     * Loading & transforming data
     ***********************************************************************/

    val d_icd_diagnoses = Utils.csv_from_s3(
      spark, f"${config.mimicInput}/D_ICD_DIAGNOSES.csv.gz",
      Some(Schemas.d_icd_diagnoses))
    d_icd_diagnoses.cache()

    val d_labitems = Utils.csv_from_s3(
      spark, f"${config.mimicInput}/D_LABITEMS.csv.gz", Some(Schemas.d_labitems))

    val patients = Utils.csv_from_s3(
      spark, f"${config.mimicInput}/PATIENTS.csv.gz", Some(Schemas.patients))
    val labevents = Utils.csv_from_s3(
      spark, f"${config.mimicInput}/LABEVENTS.csv.gz", Some(Schemas.labevents))
    labevents.persist(StorageLevel.MEMORY_AND_DISK)

    // For DIAGNOSES_ICD, also get the ICD9 category (which we reuse
    // in various places):
    val diagnoses_icd = Utils.csv_from_s3(
      spark, f"${config.mimicInput}/DIAGNOSES_ICD.csv.gz",
      Some(Schemas.diagnoses)).
      withColumn("ICD9_CATEGORY", $"ICD9_CODE".substr(0, 3))

    // Get (HADM_ID, ITEM_ID) for those admissions and lab items which
    // meet 'lab_min_series':
    val adm_and_labs : DataFrame = labevents.
      groupBy("HADM_ID", "ITEMID").
      // How long is this time-series for given admission & lab item?
      count.
      filter($"count" >= lab_min_series).
      select("HADM_ID", "ITEMID")
    adm_and_labs.persist(StorageLevel.MEMORY_AND_DISK)

    /***********************************************************************
     * Actually execute commands
     ***********************************************************************/
    if (config.icdMatrix) {
      genMatrix(spark, config, adm_and_labs, d_labitems, diagnoses_icd)
    }

    if (config.computeCohort) {
      computeCohort(spark, config, adm_and_labs, d_labitems,
        diagnoses_icd, labevents)
    }

    if (config.optimizeGPR) {
      val labs_cohort_train : RDD[PatientTimeSeries] = sc.
        objectFile(f"${config.outputPath}/${config.suffix}_train_rdd")

      optimizeHyperparams(spark, config, labs_cohort_train)
    }

    if (config.runGPR) {

      val labs_cohort_train : RDD[PatientTimeSeries] = sc.
        objectFile(f"${config.outputPath}/${config.suffix}_test_rdd")

      runGPR(spark, config, labs_cohort_train)
    }

  }

  def genMatrix(spark : SparkSession, config : Config, adm_and_labs : DataFrame,
    d_labitems : DataFrame, diagnoses_icd : DataFrame) : Unit =
  {
    import spark.implicits._

    val lab_min_patients = 30
    
    // Get ITEM_ID for those lab items which meet both
    // 'lab_min_series' and 'lab_min_patients'.
    val labs_patients_ok : DataFrame = adm_and_labs.
      groupBy("ITEMID").
      // How many times does this item occur, given that we're
      // concerned only with length >= lab_min_series?
      count.
      filter($"count" >= lab_min_patients)

    // Finally, get all (HADM_ID, ITEM_ID) that satisfy both:
    val labs_good_df : DataFrame = labs_patients_ok.
      join(adm_and_labs, "ITEMID").
      // join is supposed to be an inner join, but whatever...
      filter(not(isnull($"HADM_ID"))).
      select("HADM_ID", "ITEMID")

    labs_patients_ok.cache()
    labs_good_df.cache()

    val item_parquet = f"${config.outputPath}/items_limit.parquet"
    val item_csv = f"${config.outputPath}/items_limit.csv"
    val mtx_parquet = f"${config.outputPath}/icd9_item_matrix.parquet"
    val mtx_csv = f"${config.outputPath}/icd9_item_matrix.csv"

    // Select the top 'config.loincCount' items and write them to a file.
    // These are used later too.
    val items_limit : DataFrame = labs_patients_ok.
      sort(desc("count")).
      limit(config.loincCount).
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
    // 'config.loincCount' of them at least), and have each row
    // count up the number of occurrences of each ICD-9 category for
    // each LOINC code.
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
    // 'config.icd9Count' rows:
    val icd9_item_matrix : DataFrame = icd9_item_matrix_raw.
      withColumn("sum",
        icd9_item_matrix_raw.columns.tail.map(col).reduce(_+_)).
      sort(desc("sum")).
      limit(config.icd9Count)
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

  def computeCohort(spark : SparkSession, config : Config,
    adm_and_labs : DataFrame, d_labitems : DataFrame,
    diagnoses_icd : DataFrame, labevents : DataFrame,
    trainRatio : Double = 0.7) : Unit =
  {
    import spark.implicits._

    // Get those admissions which had >= 1 diagnosis of config.icd9Code1, or
    // of config.icd9Code2, but not diagnoses of both.
    val diag_cohort : DataFrame = diagnoses_icd.
      withColumn("is_code1", ($"ICD9_CATEGORY" === config.icd9Code1).cast(IntegerType)).
      withColumn("is_code2", ($"ICD9_CATEGORY" === config.icd9Code2).cast(IntegerType)).
      groupBy("HADM_ID").
      sum("is_code1", "is_code2").
      withColumnRenamed("sum(is_code1)", "num_code1").
      withColumnRenamed("sum(is_code2)", "num_code2").
      filter(($"num_code1" > 0) =!= ($"num_code2" > 0))
    diag_cohort.cache()
    diag_cohort.
      write.
      mode(SaveMode.Overwrite).
      parquet(f"${config.outputPath}/${config.suffix}_diag.parquet")

    // Also write a more externally-usable form:
    val diag_cohort_categories : DataFrame = diag_cohort.
      withColumn("ICD9_CATEGORY",
        when($"num_code1" > 0, config.icd9Code1).
          when($"num_code2" > 0, config.icd9Code2)).
      select("HADM_ID", "ICD9_CATEGORY")
    Utils.csvOverwrite(diag_cohort_categories).
      save(f"${config.outputPath}/${config.suffix}_categories.csv")

    // Get the lab events which meet 'lab_min_series', which are from
    // an admission in the cohort, and which are of the desired test.
    val labs_cohort_df : DataFrame = adm_and_labs.
      join(labevents, Seq("HADM_ID", "ITEMID")).
      withColumnRenamed("ROW_ID", "ROW_ID2").
      join(d_labitems, "ITEMID").
      filter($"LOINC_CODE" === config.loincTest).
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
          if (code1) config.icd9Code1 else { if (code2) config.icd9Code2 else "" })
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
      saveAsObjectFile(f"${config.outputPath}/${config.suffix}_labs_rdd")

    // Flatten out to load elsewhere:
    val labs_cohort_flat : DataFrame = Utils.flattenTimeseries(spark, labs_cohort)
    labs_cohort_flat.
      write.
      mode(SaveMode.Overwrite).
      parquet(f"${config.outputPath}/${config.suffix}_labs.parquet")
    Utils.csvOverwrite(labs_cohort_flat).
      save(f"${config.outputPath}/${config.suffix}_labs.csv")

    val randomSeed : Long = 0x12345

    // Separate training & test:
    val labs_cohort_split = labs_cohort.map { ps: PatientTimeSeries =>
      ((ps.adm_id, ps.item_id, ps.unit), ps)
    }.sortByKey(true).
      map(_._2).
      randomSplit(
        Array(trainRatio, 1.0 - trainRatio), randomSeed)
    val labs_cohort_train = labs_cohort_split(0)
    val labs_cohort_test  = labs_cohort_split(1)
    // Save training & test to disk (they'll be needed later):
    labs_cohort_train.
      saveAsObjectFile(f"${config.outputPath}/${config.suffix}_train_rdd")
    labs_cohort_test.
      saveAsObjectFile(f"${config.outputPath}/${config.suffix}_test_rdd")
    val train_flat : DataFrame = Utils.flattenTimeseries(spark, labs_cohort_train)
    Utils.csvOverwrite(train_flat).
      save(f"${config.outputPath}/${config.suffix}_train.csv")
    val test_flat : DataFrame = Utils.flattenTimeseries(spark, labs_cohort_test)
    Utils.csvOverwrite(test_flat).
      save(f"${config.outputPath}/${config.suffix}_test.csv")
  }

  // Hyperparameter optimization (prior to runGPR)
  def optimizeHyperparams(spark : SparkSession, config : Config,
    labs_cohort : RDD[PatientTimeSeries]) : Unit =
  {
    import spark.implicits._
    val sc = spark.sparkContext
    // Hyperparameter optimization:
    val sigma2Param = new DoubleParam("", "sigma2", "")
    val alphaParam = new DoubleParam("", "alpha", "")
    val tauParam = new DoubleParam("", "tau", "")

    // Compute an array of (sigma2, alpha, tau) parameters, spanning a
    // grid.  This is used for a grid search, which is admittedly a
    // very inefficient way to go about this.  The more efficient way,
    // and probably the more accurate one, would be to derive the
    // gradient of the log-likelihood function (which is probably not
    // especially complicated) and use some form of gradient-descent
    // (like Spark already has built in).  Or, perhaps Spark has some
    // optimization method built in that can numerically estimate
    // gradients.
    val paramGrid : Array[(Double, Double, Double)] = new ParamGridBuilder().
      addGrid(sigma2Param, 0.05 to 2.0 by 0.05).
      addGrid(alphaParam, 0.05 to 2.00 by 0.05).
      addGrid(tauParam, 0.2 to 2.0 by 0.05).
      build.
      map { pm =>
        (pm.get(sigma2Param).get,
          pm.get(alphaParam).get,
          pm.get(tauParam).get)
        // The .get is intentional; it *should* throw an exception if
        // it can't find the parameter this early.
     }

    // Enabling the below (and swapping paramGrid for
    // paramGridVar.value in labs_ll) causes a problem like
    // https://stackoverflow.com/questions/34329299/issuing-spark-submit-on-command-line-completes-tasks-but-never-returns-prompt
    // This problem goes away on much smaller version of 'paramGrid'.
    // I have no idea what is causing this.

    // val paramGridVar = sc.broadcast(paramGrid)

    val labs_ll : RDD[((Double, Double, Double), Double)] =
      labs_cohort.flatMap { series =>
        val s = series.warpedSeries
        val grid = paramGrid
        //val grid = paramGridVar.value
        grid.
          map { case t@(sigma2, alpha, tau) =>
            // We are only concerned with log-likelihood here.  We
            // reuse the model parameters elsewhere, but they're fast
            // enough to recompute that there's not really any point
            // in storing them.  With more ridiculous covariance
            // functions or longer time-series, that may not be true.
            (t, Utils.gprTrain(s, sigma2, alpha, tau).log_likelihood)
        }
      }.reduceByKey(_ + _)

    // Basically argmax to tell what hyperparameters produce the
    // highest sum-log-likelihood:
    val optimal = labs_ll.
      aggregate((0.0, 0.0, 0.0), Double.NegativeInfinity)(
        { case(_, t) => t },
        { (t1,t2) => if (t1._2 > t2._2) t1 else t2 }
      )

    // Write the hyperparameters to disk:
    val hyperDf = sc.parallelize(Seq(optimal)).
      map { case ((sig2, a, t), ll) => (sig2, a, t, ll, config.loincTest) }.
      toDF("sigma2", "alpha", "tau", "log_likelihood", "LOINC_CODE");

    val (sig2, a, t) = optimal._1
    println("Optimized hyperparameters:")
    println(f"sigma^2 = ${sig2}")
    println(f"alpha = ${a}")
    println(f"tau = ${t}")

    val hyperparamsCsv = f"${config.outputPath}/${config.suffix}_hyperparams.csv"
    Utils.csvOverwrite(hyperDf).
      save(hyperparamsCsv)

    print(f"Wrote hyperparameters to: ${hyperparamsCsv}")
  }
  
  // Gaussian process regression
  def runGPR(spark : SparkSession, config : Config, data : RDD[PatientTimeSeries]) : Unit = {

    // First, perform Gaussian process regression over the input data,
    // thus producing a model for each time series:
    val sigma2 = 0.012
    val alpha = 0.189
    val tau = 1.246
    // TODO: Pull these out to commandline options?  Or something
    val gprModels = data.map { p: PatientTimeSeries =>
      // Train a model for every time-series in training set:
      (p, Utils.gprTrain(p.warpedSeries, sigma2, alpha, tau))
    }
    // gprModels then has (PatientTimeSeries, GPR model) for every
    // time-series.

    // Next: For each time-series, generate new time values from a
    // regular sampling of the time range (plus some padding at each
    // end).  From each set of resampled time values, generate
    // predicted values with gprPredict (and the time-series'
    // respective Gaussian process model computed above).

    // How many days before & after do we interpolate for?
    val padding = 2.5
    // What interval (in days) do we interpolate with?
    val interval = 0.25
    // TODO: Make these commandline options too?

    // Create a new time-series with these predictions:
    val tsInterpolated = gprModels.map { case (p, model) =>
      val ts = p.warpedSeries.map(_._1)
      val ts2 = (ts.min - padding) to (ts.max + padding) by interval
      val predictions = Utils.gprPredict(ts2, ts, model)
      PatientTimeSeriesPredicted(p.adm_id, p.item_id, p.subject_id, p.unit,
        p.icd9category,
        (ts2, predictions.map(_._1), predictions.map(_._2)).zipped.toList)
    }

    val tsInterp_flat : DataFrame = Utils.
      flattenPredictedTimeseries(spark, tsInterpolated)
    tsInterp_flat.
      write.
      mode(SaveMode.Overwrite).
      parquet(f"${config.outputPath}/${config.suffix}_predict.parquet")
    Utils.csvOverwrite(tsInterp_flat).
      save(f"${config.outputPath}/${config.suffix}_predict.csv")
  }
}
