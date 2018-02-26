package edu.tulane.cs.hetml.nlp.sprl.Triplets

import java.io.PrintStream

import edu.illinois.cs.cogcomp.lbjava.classify.{FeatureVector, ScoreSet}
import edu.illinois.cs.cogcomp.lbjava.learn.Learner
import edu.tulane.cs.hetml.nlp.BaseTypes.Relation
import edu.tulane.cs.hetml.nlp.sprl.Triplets.MultiModalSpRLDataModel._

class VisualGenomeRCC8Support extends Learner("sprl.VisualGenomeRCC8") {

  override def allowableValues: Array[String] = {
    Array[String]("false", "true")
  }

  override def equals(o: Any): Boolean = {
    getClass == o.getClass
  }

  override def scores(example: AnyRef): ScoreSet = {
    val result: ScoreSet = new ScoreSet
    val r = example.asInstanceOf[Relation]
    val args = tripletHeadWordForm(r).split("::")
    val tr = args(0)
    val lm = args(2)
    val sp = args(1)
    val vgStat = relationsStats().filter(v => {
      v.getPredicate==sp && v.getSubject==tr && v.getObject==lm
    })
    if (vgStat.size==0) {
      result.put("none", 1.0)
      result.put("true", 0.0)
      result.put("false", 0.0)
    }
    else {
      val scores = vgStat.head.getScoreRCC8
      val score =  vgStat.head.getPoScore //scores.max
      if (score <= 0) {
        result.put("none", 0.0)
        result.put("true", 0.0)
        result.put("false", 1.0)
      }
      else {
        result.put("none", 0.0)
        result.put("true", score)
        result.put("false", 0.0)
      }
    }
    result
  }

  override def write(printStream: PrintStream): Unit = ???

  override def scores(ints: Array[Int], doubles: Array[Double]): ScoreSet = ???

  override def classify(ints: Array[Int], doubles: Array[Double]): FeatureVector = ???

  override def learn(ints: Array[Int], doubles: Array[Double], ints1: Array[Int], doubles1: Array[Double]): Unit = ???
}