library(ggplot2)

# SparkR boilerplate
.libPaths(c(.libPaths(), '/usr/lib/spark/R/lib'))  
Sys.setenv(SPARK_HOME = '/usr/lib/spark') 
library(SparkR, lib.loc = c(file.path(Sys.getenv("SPARK_HOME"), "R", "lib")))
sc <- sparkR.session(master = "local[*]", sparkEnvir = list(spark.driver.memory="2g"))
# TODO: Can I make this use YARN instead?

polynomial_time_warp <- function(ts, a = 3.0, b = 0.0) {
  # Implement the polynomial time warping described in "Computational Phenotype 
  # Discovery" by Lasko, Denny, & Levy (2013), equation 5.  This will also
  # remove any offset that is present, that is, the resultant vector will start
  # at 0.
  # 
  # Args:
  #   ts: Time values in a vector.  Should be consecutive.
  #   a: Optional value of coefficient 'a'; default is 3.0.
  #   b: Optional value of coefficient 'b'; default is 0.0.
  #
  # Returns:
  #   Warped version of 'ts' (a vector of the same length).
  dw <- c(0, (diff(ts))**(1/a) + b)
  # This finds the consecutive differences, warps them, and prepend a 0 so that
  # the length of the cumulative sum comes out correct.
  cumsum(dw)
}

covar_mtx <- function(X1, X2, sigma2, alpha, tau) {
  # Compute the covariance matrix from the covariance function of equation 3 of
  # the same paper, given input vectors X1 and X2.
  n1 <- length(X1)
  n2 <- length(X2)
  k <- matrix(rep(0, n1*n2), nrow=n1)
  for (i in 1:n1) {
    for (j in 1:n2) {
      k[i,j] <- X1[i] - X2[j]
    }
  }
  k <- sigma2*(1 + k^2 / (2*alpha*tau*tau))**-alpha
  k
}

# Load data from Parquet:
dataFile <- "s3://bd4h-mimic3/temp/labs_cohort_518_584.parquet"
ts <- SparkR::read.parquet(dataFile)
schema <- structType(structField("HADM_ID", "integer"),
                     structField("ITEMID", "integer"),
                     structField("SUBJECT_ID", "integer"),
                     structField("RelChart", "double"),
                     structField("RelChartWarped", "double"),
                     structField("VALUENUM", "double"))
ts_warped <- gapply(
  ts,
  c("HADM_ID", "ITEMID"),
  function(key, x) {
    x_sort <- x[order(x$CHARTTIME),]
    y <- data.frame(HADM_ID = as.integer(key[[1]]),
                    ITEMID = as.integer(key[[2]]),
                    SUBJECT_ID = x[1,]$SUBJECT_ID,
                    RelChart = x_sort$CHARTTIME - x_sort[1,]$CHARTTIME,
                    RelChartWarped = polynomial_time_warp(x_sort$CHARTTIME),
                    VALUENUM = x_sort$VALUENUM,
                    stringsAsFactors = FALSE)
  },
  schema
)
# Is there a cleaner way of doing the above?  I'm having to duplicate the
# schema, basically.  It's sort of a pain to have to make it explicit.
cache(ts_warped)

sum_log_likelihood <- function(ts, sigma2, alpha, tau) {
  series_log_lh <- gapplyCollect(
    ts_warped,
    c("HADM_ID", "ITEMID"),
    function(key, s) {
      X <- s$RelChartWarped
      Y <- s$VALUENUM
      K <- covar_mtx(X, X, sigma2, alpha, tau)
      L <- chol(K + sigma2)
      den <- solve(L, Y)
      alph <- solve(t(L), den)
      # Log likelihood:
      ll <- (-1/2) %*% t(Y) %*% alph - sum(log(diag(L))) - (length(Y)/2) * log(2 * pi)
      y <- data.frame(HADM_ID = as.integer(key[[1]]),
                      ITEMID = as.integer(key[[2]]),
                      SUBJECT_ID = s[1,]$SUBJECT_ID,
                      LogLikelihood = ll)
    }
  )
  sum(series_log_lh$LogLikelihood, na.rm = TRUE)
}
# printSchema(ts)

# Plot as a test:
ts_multi <- collect(filter(ts_warped, (ts_warped$SUBJECT_ID == 18944) | (ts_warped$SUBJECT_ID == 68135) | (ts_warped$SUBJECT_ID == 6466)))
ggplot(ts_multi, aes(x=RelChartWarped, y=VALUENUM, group = SUBJECT_ID)) +
  xlab("Chart time") +
  ylab("Value") +
  geom_line(aes(colour = factor(SUBJECT_ID)))
