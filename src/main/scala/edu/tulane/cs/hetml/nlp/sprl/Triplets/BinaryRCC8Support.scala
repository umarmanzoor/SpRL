package edu.tulane.cs.hetml.nlp.sprl.Triplets

import java.io.PrintStream

import edu.illinois.cs.cogcomp.lbjava.classify.{FeatureVector, ScoreSet}
import edu.illinois.cs.cogcomp.lbjava.learn.Learner
import edu.tulane.cs.hetml.nlp.BaseTypes.Relation
import edu.tulane.cs.hetml.nlp.sprl.Triplets.MultiModalSpRLDataModel._

class BinaryRCC8Support(rcc8: String) extends Learner("sprl.VisualGenomeRcc8") {

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

      val scoreVal =
        if(rcc8=="EC")
          vgStat.head.getEcScore
        else if (rcc8=="DC")
          vgStat.head.getDcScore
        else if(rcc8=="PO")
          vgStat.head.getPoScore
        else if(rcc8=="TPP")
          vgStat.head.getTppScore
        else if(rcc8=="TPPi")
          vgStat.head.getTppiScore
        else if(rcc8=="NTTP")
          vgStat.head.getNtppScore
        else if(rcc8=="NTTPi")
          vgStat.head.getNtppiScore
        else if(rcc8=="EQ")
          vgStat.head.getEqScore
        else
          -1.0

      val score = scoreVal.asInstanceOf[Double]
      if(score == -1.0)
        println("Warning: Score not found...")

      if (score <= 50) {
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