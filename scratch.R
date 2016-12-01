library(ggplot2)

# SparkR boilerplate
# spark_dir <- "/usr/lib/spark"
spark_dir <- "/opt/apache-spark"
.libPaths(c(.libPaths(), paste(spark_dir, "/R/lib", sep="")))
Sys.setenv(SPARK_HOME = spark_dir) 
library(SparkR, lib.loc = c(file.path(Sys.getenv("SPARK_HOME"), "R", "lib")))
sc <- sparkR.session(master = "local[*]", sparkEnvir = list(spark.driver.memory="2g"))

# Load data from Parquet:
# dataFile <- "s3://bd4h-mimic3/temp/labs_cohort_518_584.parquet"
dataFile <- "file:///home/hodapp/source/bd4h-project/data-temp/labs_cohort_518_584.parquet"
ts_a <- collect(SparkR::read.parquet(dataFile))
ts_a$Type <- "Actual"

predictionsFile <- "file:///home/hodapp/source/bd4h-project/data-temp/labs_cohort_predict_518_584.parquet"
ts_p <- collect(SparkR::read.parquet(predictionsFile))
ts_p$Type <- "Predicted"

ts <- rbind(ts_a, ts_p)

# Plot as a test:
ts_ <- ts[ts$SUBJECT_ID == 6466,]
# ts$SUBJECT_ID == 18944 | ts$SUBJECT_ID == 68135 | 
ggplot(ts_, aes(x=CHARTTIME_warped, y=VALUENUM, group = interaction(SUBJECT_ID, Type))) +
  #geom_line(aes(x = CHARTTIME_warped, y=VALUE, group = SUBJECT_ID), linetype = "dashed") +
  xlab("Chart time") +
  ylab("Value") +
  geom_line(aes(colour = interaction(factor(SUBJECT_ID), Type)))
