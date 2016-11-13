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
cache(ts_warped)

# printSchema(ts)



ts_multi <- collect(filter(ts_warped, (ts_warped$SUBJECT_ID == 18944) | (ts_warped$SUBJECT_ID == 68135) | (ts_warped$SUBJECT_ID == 6466)))
ggplot(ts_multi, aes(x=RelChartWarped, y=VALUENUM, group = SUBJECT_ID)) +
  xlab("Chart time") +
  ylab("Value") +
  geom_line(aes(colour = factor(SUBJECT_ID)))
