package edu.tulane.cs.hetml.nlp.sprl.Triplets

import java.awt.geom.Rectangle2D

import edu.illinois.cs.cogcomp.saul.util.Logging
import edu.tulane.cs.hetml.nlp.BaseTypes._
import edu.tulane.cs.hetml.nlp.LanguageBaseTypeSensors.documentToSentenceGenerating
import edu.tulane.cs.hetml.nlp.sprl.Helpers._
import MultiModalSpRLDataModel.{segments, _}
import edu.tulane.cs.hetml.nlp.sprl.Triplets.TripletSensors.alignmentHelper
import edu.tulane.cs.hetml.nlp.sprl.Triplets.tripletConfigurator.{isTrain, _}
import edu.tulane.cs.hetml.vision.{ImageTripletReader, JSONReader, Segment, WordSegment}
import edu.tulane.cs.hetml.relations.{ClefRelationsHeadwords, RelationInformationReader}
import java.io._

import scala.collection.JavaConversions._
import scala.collection.mutable.{ArrayBuffer, ListBuffer}
import edu.tulane.cs.hetml.nlp.sprl.MultiModalSpRLSensors._
import me.tongfei.progressbar.ProgressBar

import scala.util.control.Breaks.{break, breakable}
/** Created by Taher on 2017-02-12.
  */

object MultiModalPopulateData extends Logging {

  LexiconHelper.path = spatialIndicatorLex
  lazy val xmlTestReader = new SpRLXmlReader(testFile, globalSpans)
  lazy val xmlTrainReader = new SpRLXmlReader(trainFile, globalSpans)

  def xmlReader = if (isTrain) xmlTrainReader else xmlTestReader

  lazy val imageTrainReader = new ImageReaderHelper(imageDataPath, trainFile, testFile, true)
  lazy val imageTestReader = new ImageReaderHelper(imageDataPath, trainFile, testFile, false)

  def imageReader = if (isTrain) imageTrainReader else imageTestReader

  lazy val alignmentTrainReader = new AlignmentReader(alignmentAnnotationPath, true)
  lazy val alignmentTestReader = new AlignmentReader(alignmentAnnotationPath, false)

  var distinctRels = scala.collection.mutable.Map[String, String]()
  // load visualgenome relations
  val relationReader = new RelationInformationReader();
  relationReader.loadRelations(imageDataPath);
  val visualgenomeRelationsList = relationReader.visualgenomeRelations.toList

  def alignmentReader = if (isTrain) alignmentTrainReader else alignmentTestReader

  def populateRoleDataFromAnnotatedCorpus(populateNullPairs: Boolean = true): Unit = {
    logger.info("Role population started ...")
    if (isTrain && onTheFlyLexicon) {
      LexiconHelper.createSpatialIndicatorLexicon(xmlReader)
    }
    documents.populate(xmlReader.getDocuments, isTrain)
    sentences.populate(xmlReader.getSentences, isTrain)

    if (populateNullPairs) {
      phrases.populate(List(dummyPhrase), isTrain)
    }

    val phraseInstances = (if (isTrain) phrases.getTrainingInstances.toList else phrases.getTestingInstances.toList)
      .filter(_.getId != dummyPhrase.getId)

    if (globalSpans) {
      phraseInstances.foreach {
        p =>
          p.setStart(p.getSentence.getStart + p.getStart)
          p.setEnd(p.getSentence.getStart + p.getEnd)
          p.setGlobalSpan(globalSpans)
      }
    }

    xmlReader.setRoles(phraseInstances)

    if (populateImages) {
      alignmentReader.setAlignments(phraseInstances)
      images.populate(imageReader.getImageList, isTrain)
      val segs = getAdjustedSegments(imageReader.getSegmentList)
      segments.populate(segs, isTrain)
      imageSegmentsDic = getImageSegmentsDic()
      if (alignmentMethod != "topN") {
        setBestAlignment()
      }
      else {
        val ws = segmentPhrasePairs().map {
          pair =>
            val s = (segmentPhrasePairs(pair) ~> -segmentToSegmentPhrasePair).head
            val p = (segmentPhrasePairs(pair) ~> segmentPhrasePairToPhrase).head
            val segs = (segments(s) ~> -imageToSegment ~> imageToSegment).toList
            val lemma = headWordLemma(p)
            val wordSegs = segs.map(x => new WordSegment(lemma, x, false))
            val topIds = alignmentHelper.predictTopSegmentIds(wordSegs, tripletConfigurator.topAlignmentCount)
            if (topIds.contains(s.getSegmentId)) {
              val wordSegment = new WordSegment(lemma, s, false)
              wordSegment.setPhrase(p)
              wordSegment
            }
            else
              null
        }.filter(x => x != null)
        wordSegments.populate(ws)
      }

    }

    logger.info("Role population finished.")
  }

  def populateTripletDataFromAnnotatedCorpus(
                                              trFilter: (Phrase) => Boolean,
                                              spFilter: (Phrase) => Boolean,
                                              lmFilter: (Phrase) => Boolean
                                            ): Unit = {

    logger.info("Triplet population started ...")
    val candidateRelations = TripletCandidateGenerator.generateAllTripletCandidates(
      trFilter,
      spFilter,
      lmFilter,
      isTrain
    )
    xmlReader.setTripletRelationTypes(candidateRelations)

    if(useCoReference) {
      candidateRelations.foreach(r => {
        if(r.getArgument(2).toString=="it")
          r.setProperty("ImplicitLandmark","true")
        else
          r.setProperty("ImplicitLandmark","false")
      })
      println("Processing for Co-Reference...")
      //**
      // Landmark Candidates
      val instances = if (isTrain) phrases.getTrainingInstances else phrases.getTestingInstances
      val landmarks = instances.filter(t => t.getId != dummyPhrase.getId && lmFilter(t)).toList
        .sortBy(x => x.getSentence.getStart + x.getStart)

      //**
      // Headwords of triplets
      relationReader.loadClefRelationHeadwords(imageDataPath, isTrain)
      val clefHWRelation = relationReader.clefHeadwordRelations.toList

      val ILcandidateRelations = candidateRelations.filter(r => r.getProperty("ImplicitLandmark")=="true")

      val p = new ProgressBar("Processing Relations", ILcandidateRelations.size)
      val writeRels = new PrintWriter(resultsDir + "/implicitLandmarks.txt")
      p.start()
      ILcandidateRelations.foreach(r => {
        p.step()
        val headWordsTriplet = clefHWRelation.filter(c => {
          c.getId==r.getId
        })

        //**
        // get possible Landmarks from Visual Genome
        val possibleLMs = visualgenomeRelationsList.filter(v => v.getPredicate==headWordsTriplet.head.getSp
          && v.getSubject==headWordsTriplet.head.getTr)
        if(possibleLMs.size>0) {
          //Count Unique Instances
          var uniqueRelsForLM = scala.collection.mutable.Map[String, Int]()
          possibleLMs.foreach(t => {
            val key = t.getSubject + "-" + t.getPredicate + "-" + t.getObject
            if(!(uniqueRelsForLM.keySet.exists(_ == key)))
              uniqueRelsForLM += (key -> 1)
            else {
              var count = uniqueRelsForLM.get(key).get
              count = count + 1
              uniqueRelsForLM.update(key, count)
            }
          })

          //**
          // get all landmark candidates for the sentence
          val rSId = r.getArgumentId(0).split("\\(")(0)
          val sentenceLMs = landmarks.filter(l => {
            l.getSentence.getId == rSId && l.getText!=r.getArgument(0).toString && l.getText!=r.getArgument(2).toString
          })

          //**
          // Similarity scroes for each
          val w2vVectorScores = uniqueRelsForLM.toList.map(t => {
            val rSplited = t._1.split("-")
            if(rSplited.length==3)
              getMaxW2VScore(rSplited(2), sentenceLMs)
            else {
              val dummyLM = new Phrase()
              dummyLM.setText("None")
              dummyLM.setId("dummy")
              (dummyLM, -1, 0.0)
            }
          })
          val ProbableLandmark = w2vVectorScores.sortBy(_._3).last
          if(ProbableLandmark._3>0.90 && ProbableLandmark._1.getText!="None") {
            val pLM = headWordFrom(ProbableLandmark._1)
            writeRels.println(r.getArgument(0).toString + "-" + r.getArgument(1).toString + "-" + r.getArgument(2).toString + "->" + pLM)
            r.setProperty("ProbableLandmark", pLM)
          }
          else
            r.setProperty("ProbableLandmark", "None")
        }
        else {
          r.setProperty("ProbableLandmark", "None")
        }
      })
      p.stop()
      writeRels.close()

      relationReader.loadStats(imageDataPath, isTrain)
      val vgStats = relationReader.visualGenomeStats.toList
      relationsStats.populate(vgStats)
    }

    triplets.populate(candidateRelations, isTrain)

    //    if(!usePreprocessedVisualGenome)
//      saveRelationsScores()

    logger.info("Triplet population finished.")
  }

  def tripletHeadwords() = {

    val pw =
      if(isTrain)
        new PrintWriter(new File(resultsDir + "tripletsHeadWordsTrain.txt"))
      else
        new PrintWriter(new File(resultsDir + "tripletsHeadWordsTest.txt"))
    triplets().foreach(t => {
      val (first, second, third) = getTripletArguments(t)
      val tr = headWordFrom(first)
      val sp = headWordFrom(second)
      val lm = headWordFrom(third)
      pw.println(t.getId + "~" + tr + "~" + sp + "~" + lm)
    })
    pw.close()
  }

  def matchLabels(s1: String, s2: String): Boolean = {
    var matched = false
    breakable {
      s1.split(" ").foreach(f => {
        s2.split(" ").foreach(l => {
          if(f==l) {
            matched = true
            break
          }
        })
        if(matched)
          break
      })
    }
    matched
  }
  def getMaxW2VScore(replacementLM: String, sentenceLMs: List[Phrase]): (Phrase, Int, Double) = {
    val scores = sentenceLMs.map(w => {
      val phraseText = w.getText.replaceAll("[^A-Za-z0-9]", " ")
      phraseText.replaceAll("\\s+", " ");
      w.setText(phraseText)
      getGoogleSimilarity(replacementLM, headWordFrom(w))
    })
    val maxScore = scores.max
    val maxIndex = scores.indexOf(scores.max)
    val possibleLM = sentenceLMs(maxIndex)
    (possibleLM, maxIndex, maxScore)
  }
  def populateDataFromPlainTextDocuments(documentList: List[Document],
                                         indicatorClassifier: Phrase => Boolean,
                                         populateNullPairs: Boolean = true
                                        ): Unit = {

    logger.info("Data population started ...")
    val isTrain = false

    documents.populate(documentList, isTrain)
    sentences.populate(documentList.flatMap(d => documentToSentenceGenerating(d)), isTrain)
    if (populateNullPairs) {
      phrases.populate(List(dummyPhrase), isTrain)
    }
    val spCandidatesTrain = TripletCandidateGenerator.getIndicatorCandidates(phrases().toList)
    val trCandidatesTrain = TripletCandidateGenerator.getTrajectorCandidates(phrases().toList)
      .filterNot(x => spCandidatesTrain.contains(x))
    val lmCandidatesTrain = TripletCandidateGenerator.getLandmarkCandidates(phrases().toList)
      .filterNot(x => spCandidatesTrain.contains(x))


    logger.info("Triplet population started ...")
    val candidateRelations = TripletCandidateGenerator.generateAllTripletCandidates(
      x => trCandidatesTrain.exists(_.getId == x.getId),
      x => indicatorClassifier(x),
      x => lmCandidatesTrain.exists(_.getId == x.getId),
      isTrain
    )

    triplets.populate(candidateRelations, isTrain)

    logger.info("Data population finished.")
  }

  def getAdjustedSegments(segments: List[Segment]): List[Segment] = {

    val alignedPhrases = phrases().filter(_.containsProperty("goldAlignment"))
    val update = alignedPhrases
      .filter(p => segments.exists(s => s.getAssociatedImageID == p.getPropertyFirstValue("imageId") &&
        p.getPropertyValues("segId").exists(_.toInt == s.getSegmentId)))

    update.foreach {
      p =>
        segments.filter(x =>
          x.getAssociatedImageID == p.getPropertyFirstValue("imageId") &&
            p.getPropertyValues("segId").exists(_.toInt == x.getSegmentId)
        ).foreach {
          seg =>
            val im = images().find(_.getId == seg.getAssociatedImageID).get
            val x = Math.min(im.getWidth, Math.max(0, p.getPropertyFirstValue("segX").toDouble))
            val y = Math.min(im.getHeight, Math.max(0, p.getPropertyFirstValue("segY").toDouble))
            val w = Math.min(im.getWidth - x, p.getPropertyFirstValue("segWidth").toDouble)
            val h = Math.min(im.getHeight - y, p.getPropertyFirstValue("segHeight").toDouble)
            if (seg.getBoxDimensions == null)
              seg.setBoxDimensions(new Rectangle2D.Double(x, y, w, h))
            else {
              seg.getBoxDimensions.setRect(x, y, w, h)
            }
        }
    }

    segments
  }

  private def setBestAlignment() = {
    sentences().foreach(s => {
      val phraseSegments = (sentences(s) ~> sentenceToPhrase)
        .toList.flatMap(p => (phrases(p) ~> -segmentPhrasePairToPhrase).toList)
        .sortBy(x => x.getProperty("similarity").toDouble).reverse
      val usedSegments = ListBuffer[String]()
      val usedPhrases = ListBuffer[String]()
      phraseSegments.foreach(pair => {
        if (!usedPhrases.contains(pair.getArgumentId(0)) && !usedSegments.contains(pair.getArgumentId(1))) {
          usedPhrases.add(pair.getArgumentId(0))
          usedSegments.add(pair.getArgumentId(1))
          val p = (segmentPhrasePairs(pair) ~> segmentPhrasePairToPhrase).head
          if (pair.getProperty("similarity").toDouble > 0.30 || alignmentMethod == "classifier") {
            p.addPropertyValue("bestAlignment", pair.getArgumentId(1))
            p.addPropertyValue("bestAlignmentScore", pair.getProperty("similarity"))
          }
        }
      }
      )
    })
  }

  def saveRelationsScores() = {
    val p = new ProgressBar("Processing Relations", triplets().size)
    val gtRel = triplets().toList
    print(gtRel.size)
    val visualRelFilename =
      if(!useW2VViusalGenome)
        if(isTrain)
          resultsDir + "extactMatchRelsTrain.txt"
        else
          resultsDir + "extactMatchRelsTest.txt"
      else
      if(isTrain)
        resultsDir + "W2VMatchRelsTrain.txt"
      else
        resultsDir + "W2VMatchRelsTest.txt"

    val pw = new PrintWriter(new File(visualRelFilename))
    var count = 0
    p.start()
    gtRel.foreach(t => {
      p.step()
      val (first,second,third) = getTripletArguments(t)
      val tr = headWordLemma(first)
      val lm = headWordLemma(third)
      val sp = headWordFrom(second)

      val relKey =  sp + "," + tr + "," + lm
      if(!(distinctRels.keySet.exists(_ == relKey))) {
        val vgRels = visualgenomeRelationsList.filter(r => {
          if(useW2VViusalGenome)
            r.getPredicate==sp && (getGoogleSimilarity(r.getSubject,tr) >= 0.50) && (getGoogleSimilarity(r.getObject,lm)>=0.50)
          else
            r.getPredicate==sp && r.getSubject==tr && r.getObject==lm
        })

        if(vgRels.size>0) {
          // EC
          val ecRels = vgRels.filter(r => r.getRcc8Label=="EC")
          val ecScore = "%1.2f".format(ecRels.size.toDouble / vgRels.size)
          // DC
          val dcRels = vgRels.filter(r => r.getRcc8Label=="DC")
          val dcScore = "%1.2f".format(dcRels.size.toDouble / vgRels.size)
          // TPP
          val tppRels = vgRels.filter(r => r.getRcc8Label=="TPP")
          val tppScore = "%1.2f".format(tppRels.size.toDouble / vgRels.size)
          // TPPi
          val tppiRels = vgRels.filter(r => r.getRcc8Label=="TPPi")
          val tppiScore = "%1.2f".format(tppiRels.size.toDouble / vgRels.size)
          // NTTP
          val nttpRels = vgRels.filter(r => r.getRcc8Label=="NTPP")
          val nttpScore = "%1.2f".format(nttpRels.size.toDouble / vgRels.size)
          // NTPPi
          val nttpiRels = vgRels.filter(r => r.getRcc8Label=="NTPPi")
          val nttpiScore = "%1.2f".format(nttpiRels.size.toDouble / vgRels.size)
          // EQ
          val eqRels = vgRels.filter(r => r.getRcc8Label=="EQ")
          val eqScore = "%1.2f".format(eqRels.size.toDouble / vgRels.size)
          // PO
          val poRels = vgRels.filter(r => r.getRcc8Label=="PO")
          val poScore = "%1.2f".format(poRels.size.toDouble / vgRels.size)

          // Above
          val aboveRels = vgRels.filter(r => r.getDirectionLabel=="ABOVE")
          val aboveScore = "%1.2f".format(aboveRels.size.toDouble / vgRels.size)
          // Below
          val belowRels = vgRels.filter(r => r.getDirectionLabel=="BELOW")
          val belowScore = "%1.2f".format(belowRels.size.toDouble / vgRels.size)
          // Left
          val leftRels = vgRels.filter(r => r.getDirectionLabel=="LEFT")
          val leftScore = "%1.2f".format(leftRels.size.toDouble / vgRels.size)
          // Right
          val rightRels = vgRels.filter(r => r.getDirectionLabel=="RIGHT")
          val rightScore = "%1.2f".format(rightRels.size.toDouble / vgRels.size)


          val relStats = vgRels.size + "," + ecScore + "," + dcScore + "," + tppScore + "," + tppiScore + "," + nttpScore + "," + nttpiScore + "," + eqScore + "," + poScore + "," + aboveScore + "," + belowScore + "," + leftScore + "," + rightScore

          distinctRels += (relKey -> relStats)
          count = count + 1
        }
        else {
          val relKey =  sp + "," + tr + "," + lm
          val relStats = "0,0,0,0,0,0,0,0,0,0,0,0,0"
          if(!(distinctRels.keySet.exists(_ == relKey)))
            distinctRels += (relKey -> relStats)

        }
        //println(tr + "-" + lm + "-" + sp + "-> " + vgRels.size)
      }
    })
    p.stop()
    pw.write("Total Matched ->" + count + "\n")
    for ((k,v) <- distinctRels)
      pw.write(k + "," + v + "\n")

    pw.close

    println(count)
  }
}

