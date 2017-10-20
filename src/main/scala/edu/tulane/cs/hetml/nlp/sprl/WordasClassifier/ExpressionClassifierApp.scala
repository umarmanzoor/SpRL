package edu.tulane.cs.hetml.nlp.sprl.WordasClassifier

import java.io.PrintWriter

import edu.illinois.cs.cogcomp.saul.classifier.Results
import edu.tulane.cs.hetml.nlp.BaseTypes._
import edu.tulane.cs.hetml.nlp.LanguageBaseTypeSensors
import edu.tulane.cs.hetml.nlp.LanguageBaseTypeSensors._
import edu.tulane.cs.hetml.nlp.sprl.Eval.SpRLEvaluation
import edu.tulane.cs.hetml.nlp.sprl.Helpers.ReportHelper.convertToEval
import edu.tulane.cs.hetml.nlp.sprl.MultiModalSpRLDataModel._
import edu.tulane.cs.hetml.nlp.sprl.MultiModalSpRLSensors._
import edu.tulane.cs.hetml.nlp.sprl.WordasClassifier.WordasClassifierClassifiers._
import edu.tulane.cs.hetml.nlp.sprl.WordasClassifier.WordasClassifierDataModel.loadWordClassifiers
import edu.tulane.cs.hetml.nlp.sprl.mSpRLConfigurator._
import edu.tulane.cs.hetml.vision._
import me.tongfei.progressbar.ProgressBar

import scala.collection.JavaConversions._
import scala.collection.mutable
import scala.collection.mutable.{ArrayBuffer, HashMap, ListBuffer}

/** Created by Umar on 2017-10-04.
  */

object ExpressionClassifierApp extends App {

  // Preprocess RefExp
  val stopWords = Array("the", "an", "a")
  var combinedResults = Seq[SpRLEvaluation]()

  val relWords = Array("below", "above", "between", "not", "behind", "under", "underneath", "front of", "right of",
    "left of", "ontop of", "next to", "middle of")

  val CLEFGoogleNETReaderHelper = new CLEFGoogleNETReader(imageDataPath)
  val classifierDirectory = s"models/mSpRL/expressionClassifer/"

  val allImages =
    if(isTrain)
      CLEFGoogleNETReaderHelper.trainImages.take(100).toList
    else
      CLEFGoogleNETReaderHelper.testImages.toList

  val allsegments =
    if(!useAnntotatedClef) {
        CLEFGoogleNETReaderHelper.allSegments.filter(s => {allImages.exists(i=> i.getId==s.getAssociatedImageID)})
    } else {
      CLEFGoogleNETReaderHelper.allSegments.toList
    }

  val pb = new ProgressBar("Processing Data", allsegments.size)
  pb.start()

  val trainInstances = new ListBuffer[Segment]()
  val wordInstances = new ListBuffer[WordSegment]()

  wordInstances += new WordSegment("hello", allsegments(0), false, false, "")
  wordInstances += new WordSegment("dummy", allsegments(1), false, false, "")

  allsegments.foreach(s => {
    if (s.referItExpression != null) {

      val refExp = s.referItExpression.toLowerCase.replaceAll("[^a-z]", " ").replaceAll("( )+", " ").trim

      // Saving filtered tokens for later use
      s.filteredTokens = refExp

      if (refExp != "" && refExp.length > 1) {

        getPostags(s).foreach(p => {
          val tokenPair = p._1.getText + "," + p._2
          s.tagged.add(tokenPair)
        })

        // Create Positive Example
        trainInstances += s

        // Create Negative Example(s)
        val ImageSegs = allsegments.filter(t => t.getAssociatedImageID.equals(s.getAssociatedImageID) &&
          t.getSegmentId != s.getSegmentId)

        if (ImageSegs.nonEmpty) {
          val len = if (ImageSegs.size < 5) ImageSegs.size else 5
          for (iter <- 0 until len) {
            val negSeg = ImageSegs(iter)
            if (negSeg.referItExpression != "" && negSeg.referItExpression.length > 1) {
              if(negSeg.filteredTokens!=null)
                trainInstances += new Segment(negSeg.getAssociatedImageID, negSeg.getSegmentId, negSeg.getSegmentFeatures,
                negSeg.getExpression, false)
              else {
                val exp = negSeg.referItExpression.toLowerCase.replaceAll("[^a-z]", " ").replaceAll("( )+", " ").trim
                val newSeg = new Segment(negSeg.getAssociatedImageID, negSeg.getSegmentId, negSeg.getSegmentFeatures,
                  negSeg.getExpression, false)
                newSeg.filteredTokens = exp
                getPostags(newSeg).foreach(p => {
                  val tokenPair = p._1.getText + "," + p._2
                  newSeg.tagged.add(tokenPair)
                })
                trainInstances += newSeg
              }
            }
          }
        }
      }
    }
    pb.step()
  })
  pb.stop()

  wordsegments.populate(wordInstances)
  segments.populate(trainInstances)

  if(isTrain) {
    println("Training...")
    loadWordClassifiers()
    ExpressionasClassifer.learn(iterations)
    ExpressionasClassifer.save()
  }

  if(!isTrain) {
    println("Testing...")
    ExpressionasClassifer.load()
    ExpressionasClassifer.test()
  }

  def getPostags(s: Segment): List[(Token, String)] ={
    val d = new Document(s.getAssociatedImageID)
    val senID = s.getAssociatedImageID + "_" + s.getSegmentId.toString
    val sen = new Sentence(d, senID, 0, s.filteredTokens.length, s.filteredTokens)
    val toks = LanguageBaseTypeSensors.sentenceToTokenGenerating(sen)
    //Applying postag
    val pos = LanguageBaseTypeSensors.getPos(sen)
    //Generating token-postag Pair
    toks.zip(pos).toList
  }
}