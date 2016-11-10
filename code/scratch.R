library(ggplot2)

#Set the path for the R libraries you would like to use. 
#You may need to modify this if you have custom R libraries. 
.libPaths(c(.libPaths(), '/usr/lib/spark/R/lib'))  

#Set the SPARK_HOME environment variable to the location on EMR
Sys.setenv(SPARK_HOME = '/usr/lib/spark') 

#Load the SparkR library into R
library(SparkR, lib.loc = c(file.path(Sys.getenv("SPARK_HOME"), "R", "lib")))

#Initiate a Spark context and identify where the master node is located.
#local is used here because the RStudio server 
#was installed on the master node

sc <- sparkR.init(master = "local[*]", sparkEnvir = list(spark.driver.memory="2g"))
# TODO: Can I make this use YARN instead?
# sqlContext <- sparkRSQL.init(sc) 

ts <- SparkR::read.parquet("s3://bd4h-mimic3/temp/lab_timeseries.parquet")
printSchema(ts)

ts_f <- collect(filter(ts, ts$HADM_ID == 124271 & ts$ITEMID == 51221))

ggplot(ts_f, aes(x=CHARTTIME, y=VALUENUM)) + xlab("Chart time") + ylab("Value") + geom_line()

ts_multi <- collect(filter(ts, ts$HADM_ID == 124271))
ggplot(ts_multi, aes(x=CHARTTIME, y=VALUENUM, group = ITEMID)) +
  xlab("Chart time") +
  ylab("Value") +
  geom_line(aes(colour = factor(ITEMID)))
