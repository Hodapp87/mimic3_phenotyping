package edu.gatech.cse8803.util

import breeze.linalg.{DenseVector => BDV, DenseMatrix => BDM, sum}
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

  def rationalQuadraticCovar(x1 : Iterable[Double], x2 : Iterable[Double],
    sigma2 : Double, alpha : Double, tau : Double) : BDM[Double] =
  {
    /**
      * Compute the covariance matrix using the rational quadratic
      * function given in "Computational Phenotype Discovery" by
      * Lasko, Denny, & Levy (2013), equation 3.
      */

    // Breeze constructs matrices column-major, thus, we want our
    // *inner* iteration to be down a column and our outer iteration
    // to be across the rows.  Since each element of x1 corresponds
    // with a row, each traversal of x1 then corresponds to a column,
    // hence, x1 is the inner iteration:
    val data : Array[Double] = x2.flatMap {
      t2 => x1.map { t1 => t1 - t2 }
    }.toArray
    val mtx = new BDM(x1.size, x2.size, data)
    // 'mtx' now has every t1-t2

    // Note parenthesis around quotient.  Precedence is a little
    // borked, so the addition must occur outside.
    sigma2 :* pow(((mtx :* mtx) :/ (2*alpha*tau*tau)) + 1.0, -alpha)
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
      * hyperparameter alpha.  If 'ts' has N elements, then 'L' will
      * have dimensions N x N, and 'A' will have dimensions N x 1.
      */
    // TODO: Comment more fully.
    // ts is (time, value).
    val (x, y) = ts.unzip
    // Refer to Rasmussen & Williams, algorithm 2.1:
    val n : Int = x.size
    val K = rationalQuadraticCovar(x, x, sigma2, alpha, tau)
    val L = cholesky.apply(K + BDM.eye[Double](n) :* sigma2)
    val ym : BDM[Double] = new BDM(n, 1, y.toArray)
    val A = L.t \ (L \ ym)
    // ym.t is 1 x n, alph is n x 1, thus prod is 1 x 1, so (0,0) in
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

    // Compute k*, covariance between test points & training points:
    val ks = rationalQuadraticCovar(xTrain, xTest, sigma2, alpha, tau)
    // Compute f*:
    val fs = ks.t * A
    // Compute v:
    val v = L \ ks
    val covar = rationalQuadraticCovar(xTest, xTest, sigma2, alpha, tau) - v.t * v

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
