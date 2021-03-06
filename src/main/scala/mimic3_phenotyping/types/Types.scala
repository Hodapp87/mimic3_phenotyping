// (c) 2016 Chris Hodapp, chodapp3@gatech.edu

package mimic3_phenotyping.types

case class PatientTimeSeries(
  adm_id: Int,
  item_id: Int,
  subject_id: Int,
  unit: String,
  icd9category: String,
  // series & warpedSeries: (time in days, value)
  series: Iterable[(Double, Double)],
  warpedSeries: Iterable[(Double, Double)]
)

case class PatientTimeSeriesPredicted(
  adm_id: Int,
  item_id: Int,
  subject_id: Int,
  unit: String,
  icd9category: String,
  // (time in days, mean, variance)
  predictions: Iterable[(Double, Double, Double)]
)

case class LabItem(
  item_id: Int,
  label: String,
  fluid: String,
  category: String,
  loincCode: String
)

/*
case class GaussionProcessModel(
  // sigma^2, i.e. noise level
  sigma2: Double,
  // alpha for rational quadratic function:
  alpha: Double,
  // tau (time scale) for rational quadratic function:
  tau: Double,
  // 

)
 */

// This is for lab_ts, which inexplicably has a NullPointerException
// if I instead use a 4-tuple of identical fields.
// case class LabWrapper(a1 : Int, a2 : Int, a3 : Int, a4 : String)
