package edu.gatech.cse8803.util

import org.apache.spark.sql.types._

object Schemas {

  val patients = StructType(Array(
    StructField("ROW_ID",IntegerType,true),
    StructField("SUBJECT_ID",IntegerType,true),
    StructField("GENDER",StringType,true),
    StructField("DOB",TimestampType,true),
    StructField("DOD",TimestampType,true),
    StructField("DOD_HOSP",TimestampType,true),
    StructField("DOD_SSN",TimestampType,true),
    StructField("EXPIRE_FLAG",StringType,true)
  ))

  val d_labitems = StructType(Array(
    StructField("ROW_ID",IntegerType,true),
    StructField("ITEMID",IntegerType,true),
    StructField("LABEL",StringType,true),
    StructField("FLUID",StringType,true),
    StructField("CATEGORY",StringType,true),
    StructField("LOINC_CODE",StringType,true)
  ))

  val labevents = StructType(Array(
    StructField("ROW_ID",IntegerType,true),
    StructField("SUBJECT_ID",IntegerType,true),
    StructField("HADM_ID",IntegerType,true),
    StructField("ITEMID",IntegerType,true),
    StructField("CHARTTIME",TimestampType,true),
    StructField("VALUE",StringType,true),
    StructField("VALUENUM",DoubleType,true),
    StructField("VALUEUOM",StringType,true),
    StructField("FLAG",StringType,true)
  ))

  val diagnoses = StructType(Array(
    StructField("ROW_ID",IntegerType,true),
    StructField("SUBJECT_ID",IntegerType,true),
    StructField("HADM_ID",IntegerType,true),
    StructField("SEQ_NUM",IntegerType,true),
    StructField("ICD9_CODE",StringType,true)
  ))

  val d_icd_diagnoses = StructType(Array(
    StructField("ROW_ID",IntegerType,true),
    StructField("ICD9_CODE",StringType,true),
    StructField("SHORT_TITLE",StringType,true),
    StructField("LONG_TITLE",StringType,true)
  ))
  
}
