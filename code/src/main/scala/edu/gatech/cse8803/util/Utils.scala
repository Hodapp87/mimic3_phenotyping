// (c) 2016 Chris Hodapp, chodapp3@gatech.edu

package edu.gatech.cse8803.util

import edu.gatech.cse8803.types._

import breeze.linalg.{DenseMatrix => BDM, sum}
import breeze.linalg._
import breeze.numerics._

import org.apache.spark.rdd.RDD
import org.apache.spark.sql._
import org.apache.spark.sql.types._

case object Utils {
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

  def covarMtx(x1 : Iterable[Double], x2 : Iterable[Double],
    k : (Double,Double) => Double) : BDM[Double] =
  {
    /** Build a covariance matrix from x1 and x2 given a covariance
      * function.  For inputs of lengths M and N respectively, The
      * resultant matrix will by MxN, and element (i,j) will be
      * k(x1[i], x2[j]). */

    // Breeze constructs matrices column-major, thus, we want our
    // *inner* iteration to be down a column and our outer iteration
    // to be across the rows.  Since each element of x1 corresponds
    // with a row, each traversal of x1 then corresponds to a column,
    // hence, x1 is the inner iteration:
    val data : Array[Double] = x2.flatMap {
      t2 => x1.map { t1 => k(t1, t2) }
    }.toArray
    new BDM(x1.size, x2.size, data)
  }

  
  def rationalQuadraticCovar(x1 : Iterable[Double], x2 : Iterable[Double],
    sigma2 : Double, alpha : Double, tau : Double) : BDM[Double] =
  {
    /**
      * Compute the covariance matrix using the rational quadratic
      * function given in "Computational Phenotype Discovery" by
      * Lasko, Denny, & Levy (2013), equation 3.
      */

    val fn = (t1 : Double, t2 : Double) => {
      val d2 = pow(t1-t2, 2)
      sigma2 * pow((d2 / (2*alpha*tau*tau)) + 1.0, -alpha)
    }
    covarMtx(x1, x2, fn)
  }

  def squaredExpCovar(x1 : Iterable[Double], x2 : Iterable[Double],
    sigma2f : Double, sigma2n : Double, l : Double) : BDM[Double] =
  {
    /**
      * Compute the covariance matrix using the squared-exponential
      * covariance function given in "Gaussian Processes for Machine
      * Learning" (Rasmussen & Williams), equation 2.31.
      */

    // If this ever were being done in much larger inputs, a lot of
    // room for optimization probably exists as this has several
    // completely-parallel multiplications.
    val fn = (t1 : Double, t2 : Double) => {
      val d2 = pow(t1-t2, 2)
      sigma2f * Math.exp(-d2 /(2*l*l)) + (if (t1==t2) sigma2n else 0)
    }
    covarMtx(x1, x2, fn)
  }

  def gprTrain(ts : Iterable[(Double, Double)],
    sigma2 : Double, alpha : Double, tau : Double) :
      (Double, BDM[Double], BDM[Double]) =
  {
    /** 
      * Train a model for Gaussian process regression using algorithm
      * 2.1 of "Gaussian Processes for Machine Learning" (Rasmussen &
      * Williams, 2006).  Input 'ts' is a time-series of form (time,
      * value).
      * 
      * Returns: (log marginal likelihood, L matrix, A matrix) - where
      * 'A' is what the text denotes as alpha (no relation to the
      * hyperparameter alpha)a.  If 'ts' has N elements, then 'L' will
      * have dimensions N x N, and 'A' will have dimensions N x 1.
      */
    // TODO: Comment more fully.
    // ts is (time, value).
    val (x, y) = ts.unzip
    // Refer to Rasmussen & Williams, algorithm 2.1:
    val n : Int = x.size
    //val K = squaredExpCovar(x, x, sigma2f, sigma2n, l)
    val K = rationalQuadraticCovar(x, x, sigma2, alpha, tau)
    val L = cholesky.apply(K + BDM.eye[Double](n) * sigma2)
    val ym : BDM[Double] = new BDM(n, 1, y.toArray)
    val A = L.t \ (L \ ym)
    // ym.t is 1 x n, A is n x 1, thus prod is 1 x 1, so (0,0) in
    // 'll' is the only element.  Also, I must do this separately
    // because indices can't go on the expression for some reason.
    val prod = ym.t * A
    val ll = -(prod(0,0) / 2.0) - sum(log(diag(L))) - (n * log(2*Math.PI))/2
    (ll, L, A)
  }

  def gprPredict(xTest : Iterable[Double], xTrain : Iterable[Double],
    L : BDM[Double], A : BDM[Double], sigma2 : Double, alpha : Double,
    tau : Double) : Iterable[(Double, Double)] = {
    /**
      * Predict the means and variances for test inputs (as time
      * values in 'xTest'), given previously fitted L and A from
      * 'gprTrain'.
      * 
      * Returns: Iterable of (mean, variance) corresponding to each
      * time value in 'xTest'.
      */

    // Compute k*, covariance between training points & test points:
    val ks = rationalQuadraticCovar(xTrain, xTest, sigma2, alpha, tau)    
    //val ks = squaredExpCovar(xTrain, xTest, sigma2f, sigma2n, l)
    // Compute f*:
    val fs = ks.t * A
    // Compute v:
    val v = L \ ks
    val covar = rationalQuadraticCovar(xTest, xTest, sigma2, alpha, tau) - v.t * v    
    // val covar = squaredExpCovar(xTest, xTest, sigma2f, sigma2n, l) - v.t * v

    // I don't know that I actually need to compute the *entire*
    // covariance matrix.  I only need its diagonals for the
    // covariance of every xTest with itself.  Also, this computation
    // may not even be correct outside of the diagonal.

    val means = fs.valuesIterator
    val variances = diag(covar).valuesIterator
    means.zip(variances).toList
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

  def flattenTimeseries(spark : SparkSession, rdd : RDD[PatientTimeSeries]) : DataFrame = {
    import spark.implicits._
    rdd.flatMap { p: PatientTimeSeries =>
      val ts = p.series.zip(p.warpedSeries)
      ts.map { case ((t, _), (tw, value)) =>
        (p.adm_id, p.item_id, p.subject_id, p.unit, p.icd9category, t, tw, value)
      }
    }.toDF("HADM_ID", "ITEMID", "SUBJECT_ID", "VALUEUOM", "ICD9_CATEGORY", "CHARTTIME", "CHARTTIME_warped", "VALUENUM")
  }

  def flattenPredictedTimeseries(spark : SparkSession, rdd : RDD[PatientTimeSeriesPredicted]) : DataFrame = {
    import spark.implicits._
    rdd.flatMap { p: PatientTimeSeriesPredicted =>
      p.predictions.map { case (tval, mean, variance) =>
        (p.adm_id, p.item_id, p.subject_id, p.unit, p.icd9category, tval, mean, variance)
      }
    }.toDF("HADM_ID", "ITEMID", "SUBJECT_ID", "VALUEUOM", "ICD9_CATEGORY", "CHARTTIME_warped", "MEAN", "VARIANCE")
  }

  /** Return a DataFrameWriter to write a coalesced CSV, with header,
    * overwriting old files if present. */
  def csvOverwrite(df : DataFrame) : DataFrameWriter[Row] = {
    df.
      coalesce(1).
      write.
      mode(SaveMode.Overwrite).
      format("com.databricks.spark.csv").
      option("header", "true")
  }

}
