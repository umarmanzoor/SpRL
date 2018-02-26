package edu.tulane.cs.hetml.nlp.sprl.VisualTriplets

import edu.illinois.cs.cogcomp.saul.datamodel.DataModel
import edu.tulane.cs.hetml.nlp.BaseTypes.{Document, Sentence, Token}
import edu.tulane.cs.hetml.nlp.LanguageBaseTypeSensors.getPos
import edu.tulane.cs.hetml.vision._
import edu.tulane.cs.hetml.nlp.sprl.MultiModalSpRLSensors._
import edu.tulane.cs.hetml.nlp.LanguageBaseTypeSensors._
import edu.tulane.cs.hetml.relations.VisualGenomeStats

/** Created by Umar on 2017-11-09.
  */
object VisualTripletsDataModel extends DataModel {

  val visualTriplets = node[ImageTriplet]
  val sentences = node[Sentence]
  val tokens = node[Token]
  val visualGenomeStats = node[VisualGenomeStats]

  val tripletToSentence = edge(visualTriplets, sentences)
  tripletToSentence.addReverseSensor((x: ImageTriplet) => {
    val doc = new Document(x.getImageId)
    val text = x.getTrajector + " " + x.getSp + " " + x.getLandmark
    val s = new Sentence(doc, x.getImageId + "_" + x.getFirstSegId + "_" + x.getFirstSegId + "_" + x.getSp, 0,
      text.length, text)
    List(s)
  })

  val sentenceToToken = edge(sentences, tokens)
  sentenceToToken.addSensor(sentenceToTokenGenerating _)

  val visualTripletLabel = property(visualTriplets) {
    t: ImageTriplet =>
      t.getSp.toLowerCase
  }

  val visualTripletGnomeStats = property(visualTriplets) {
    t: ImageTriplet =>
      val rels = visualGenomeStats().filter(r => {
        r.getPredicate.toLowerCase==t.getSp.toLowerCase &&
          (if((t.getTrajector.toLowerCase.split(" ").map(_.contains(r.getSubject)).filter(_==true)).size > 0) true else false) &&
          (if((t.getLandmark.toLowerCase.split(" ").map(_.contains(r.getObject)).filter(_==true)).size > 0) true else false)
      })
      val a =
        if(rels.size>0)
          rels.head.getScoreArray.toList.map(_.toDouble)
        else
          List.fill(12)(0.0)
      a
  }

  val visualTripletTrajector = property(visualTriplets) {
    t: ImageTriplet =>
      t.getTrajector.toLowerCase
  }

  val visualTripletlandmark = property(visualTriplets) {
    t: ImageTriplet =>
      t.getLandmark.toLowerCase
  }

  val visualTripletTrajectorW2V = property(visualTriplets, ordered = true) {
    t: ImageTriplet =>
      getGoogleWordVector(t.getTrajector)
  }

  val visualTripletlandmarkW2V = property(visualTriplets, ordered = true) {
    t: ImageTriplet =>
      getGoogleWordVector(t.getLandmark)
  }

  val visualTripletTrVector = property(visualTriplets, ordered = true) {
    t: ImageTriplet =>
      List(t.getTrVectorX, t.getTrVectorY)
  }

  val visualTripletTrajectorAreaWRTLanmark = property(visualTriplets) {
    t: ImageTriplet =>
      if(t.getTrAreawrtLM.isNaN)
        logger.warn(s"Nan TrAreawrtLM in ${t.getTrajector}, ${t.getSp}, ${t.getLandmark} ")
      t.getTrAreawrtLM
  }

  val visualTripletTrajectorAspectRatio = property(visualTriplets) {
    t: ImageTriplet =>
      t.getTrAspectRatio
  }

  val visualTripletLandmarkAspectRatio = property(visualTriplets) {
    t: ImageTriplet =>
      t.getLmAspectRatio
  }

  val visualTripletTrajectorAreaWRTBbox = property(visualTriplets) {
    t: ImageTriplet =>
      t.getTrAreaBbox
  }

  val visualTripletLandmarkAreaWRTBbox = property(visualTriplets) {
    t: ImageTriplet =>
      t.getLmAreaBbox
  }

  val visualTripletIOU = property(visualTriplets) {
    t: ImageTriplet =>
      t.getIou
  }

  val visualTripletEuclideanDistance = property(visualTriplets) {
    t: ImageTriplet =>
      t.getEuclideanDistance
  }

  val visualTripletTrajectorAreaWRTImage = property(visualTriplets) {
    t: ImageTriplet =>
      t.getTrAreaImage
  }

  val visualTripletLandmarkAreaWRTImage = property(visualTriplets) {
    t: ImageTriplet =>
      t.getLmAreaImage
  }

  val visualTripletAbove = property(visualTriplets) {
    t: ImageTriplet =>
      t.getAbove
  }

  val visualTripletBelow = property(visualTriplets) {
    t: ImageTriplet =>
      t.getBelow
  }

  val visualTripletLeft = property(visualTriplets) {
    t: ImageTriplet =>
      t.getLeft
  }

  val visualTripletRight = property(visualTriplets) {
    t: ImageTriplet =>
      t.getRight
  }

  val visualTripletTrPos = property(visualTriplets, cache = true) {
    t: ImageTriplet =>
      val w = (visualTriplets(t) ~> tripletToSentence ~> sentenceToToken).toList.minBy(_.getStart)
      getPos(w).mkString
  }

  val visualTripletLmPos = property(visualTriplets, cache = true) {
    t: ImageTriplet =>
      val w = (visualTriplets(t) ~> tripletToSentence ~> sentenceToToken).toList.maxBy(_.getStart)
      getPos(w).mkString
  }

  val visualTripletTrLemma = property(visualTriplets, cache = true) {
    t: ImageTriplet =>
      val tokens = (visualTriplets(t) ~> tripletToSentence ~> sentenceToToken).toList
      if(tokens.isEmpty)
        logger.warn("empty tokens")
      val w = tokens.minBy(_.getStart)
      getLemma(w).mkString
  }

  val visualTripletLmLemma = property(visualTriplets, cache = true) {
    t: ImageTriplet =>
      val tokens = (visualTriplets(t) ~> tripletToSentence ~> sentenceToToken).toList
      if(tokens.isEmpty)
        logger.warn("empty tokens")
      val w = tokens.maxBy(_.getStart)
      getLemma(w).mkString
  }

  val visualTripletIntersection = property(visualTriplets) {
    t: ImageTriplet =>
      t.getIntersectionArea
  }

  val visualTripletUnion = property(visualTriplets) {
    t: ImageTriplet =>
      t.getUnionArea
  }
}
