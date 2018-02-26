package edu.tulane.cs.hetml.nlp.sprl.VisualTriplets

import java.io.{File, FileOutputStream, PrintWriter}

import scala.io.Source
import edu.tulane.cs.hetml.nlp.sprl.Helpers.ReportHelper
import edu.tulane.cs.hetml.vision._
import edu.tulane.cs.hetml.nlp.sprl.VisualTriplets.VisualTripletsDataModel._
import edu.tulane.cs.hetml.nlp.sprl.VisualTriplets.VisualTripletClassifiers._
import edu.tulane.cs.hetml.nlp.sprl.WordasClassifier.WordasClassifierConfigurator.resultsDir
import edu.tulane.cs.hetml.relations.RelationInformationReader
import edu.tulane.cs.hetml.nlp.sprl.MultiModalSpRLSensors._
import scala.collection.JavaConversions._
import scala.collection.mutable.ListBuffer

/** Created by Umar on 2017-11-09.
  */
object VisualTripletsApp extends App {

  val frequentPrepositions = List("in", "on", "between", "in front of", "behind", "above", "in between", "around",
    "over", "at", "next to")
  val imageDataPath = "data/mSprl/saiapr_tc-12"
  val isTrain = true
  val usePreprocessScores = false
  val classifierDirectory = s"models/mSpRL/VisualTriplets/"
  val classifierSuffix = "preposition"
  val cleftestInstances = new ListBuffer[ImageTriplet]()
  val clefTrainInstances = new ListBuffer[ImageTriplet]()
  val useW2VViusalGenome = false

  populateVisualTripletsFromExternalData()

//  if (isTrain) {
//
//    VisualTripletClassifier.modelDir = classifierDirectory
//    VisualTripletClassifier.modelSuffix = classifierSuffix
//    VisualTripletClassifier.learn(50)
//    VisualTripletClassifier.save()
//    VisualTripletClassifier.test(visualTriplets())
//  }
//  else {
//    VisualTripletClassifier.modelDir = classifierDirectory
//    VisualTripletClassifier.modelSuffix = classifierSuffix
//    VisualTripletClassifier.load()
//    val results = VisualTripletClassifier.test()
//    val outStream = new FileOutputStream(s"$resultsDir/VisualClassifier-VG-test.txt", false)
//    ReportHelper.saveEvalResults(outStream, "Visual triplet(within data model)", results)
//  }

  def populateVisualTripletsFromExternalData(): Unit = {
//    println("Reading Visual Genome Stat Scores")
//    val vgReader = new VisualGenomeReader();
//    vgReader.loadStats(resultsDir)
//    val vgStats = vgReader.visualGenomeStats.toList
//    visualGenomeStats.populate(vgStats)

    println("Reading Visual Genome Relations")
    val VGJsonReader = new JSONReader()
    VGJsonReader.readJsonFile(imageDataPath + "/VGData/")
    val vgFrequentSPTriplets = VGJsonReader.allImageTriplets.filter(x => frequentPrepositions.contains(x.getSp))
    val trainingSize = vgFrequentSPTriplets.size() * 0.8

    val (trainingInstances, testInstances) = vgFrequentSPTriplets.splitAt(trainingSize.toInt)

    println("Populating Visual Triplets from External Dataset...")
    if(isTrain)
      visualTriplets.populate(trainingInstances, isTrain)
    else
      visualTriplets.populate(testInstances, isTrain)

    if(!usePreprocessScores)
      saveRelationsScores()
  }

  def saveRelationsScores() = {
    var distinctRels = scala.collection.mutable.Map[String, String]()
    val gtRel = visualTriplets().toList
    val vgReader = new RelationInformationReader();
    vgReader.loadRelations(imageDataPath);
    val visualgenomeRelationsList = vgReader.visualgenomeRelations.toList

    val visualRelFilename =
      if(!useW2VViusalGenome)
        if(isTrain)
          resultsDir + "extactMatchVisualGenomeRelsTrain.txt"
        else
          resultsDir + "extactMatchVisualGenomeRelsRelsTest.txt"
      else
      if(isTrain)
        resultsDir + "W2VMatchVisualGenomeRelsTrain.txt"
      else
        resultsDir + "W2VMatchVisualGenomeRelsTest.txt"

    val pw = new PrintWriter(new File(visualRelFilename))
    var count = 0
    gtRel.foreach(t => {
      val tr = t.getTrajector.toLowerCase
      val lm = t.getLandmark.toLowerCase
      val sp = t.getSp.toLowerCase
      val vgRels = visualgenomeRelationsList.filter(r => {
        if(useW2VViusalGenome)
          r.getPredicate.toLowerCase==sp && (getGoogleSimilarity(r.getSubject.toLowerCase(),tr) >= 0.50) && (getGoogleSimilarity(r.getObject.toLowerCase(),lm)>=0.50)
        else
          r.getPredicate.toLowerCase==sp && r.getSubject.toLowerCase()==tr && r.getObject.toLowerCase()==lm
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

        val relKey =  sp + "," + tr + "," + lm
        val relStats = vgRels.size + "," + ecScore + "," + dcScore + "," + tppScore + "," + tppiScore + "," + nttpScore + "," + nttpiScore + "," + eqScore + "," + poScore + "," + aboveScore + "," + belowScore + "," + leftScore + "," + rightScore
        if(!(distinctRels.keySet.exists(_ == relKey)))
          distinctRels += (relKey -> relStats)

        count = count + 1
      }
      else {
        val relKey =  sp + "," + tr + "," + lm
        val relStats = "0,0,0,0,0,0,0,0,0,0,0,0,0"
        if(!(distinctRels.keySet.exists(_ == relKey)))
          distinctRels += (relKey -> relStats)
      }

      println(tr + "-" + lm + "-" + sp + "-> " + vgRels.size)
    })
    pw.write("Total Matched ->" + count + "\n")
    for ((k,v) <- distinctRels)
      pw.write(k + "," + v + "\n")

    pw.close

    println(count)
  }

  def loadClefTestData() = {

    val filename = s"$resultsDir/clef_prep_test.txt"
    for (line <- Source.fromFile(filename).getLines) {
      val parts = line.split(",")
      if(parts(0)!="-") {
        val trBox = RectangleHelper.parseRectangle(parts(3), "-")
        val lmBox = RectangleHelper.parseRectangle(parts(4), "-")
        val imageWidth = parts(5).toDouble
        val imageHeight = parts(6).toDouble
        parts(0) = parts(0).replaceAll("_", " ")
        cleftestInstances += new ImageTriplet(parts(0), parts(1), parts(2), trBox, lmBox, imageWidth, imageHeight)
      }
    }
  }

  def clefTrainData() = {

        val filename = s"$resultsDir/clefprepdata.txt"
        for (line <- Source.fromFile(filename).getLines) {
          val parts = line.split(",")
          val trBox = RectangleHelper.parseRectangle(parts(3), "-")
          val lmBox = RectangleHelper.parseRectangle(parts(4), "-")
          val imageWidth = parts(5).toDouble
          val imageHeight = parts(6).toDouble
          clefTrainInstances += new ImageTriplet(parts(0), parts(1), parts(2), trBox, lmBox, imageWidth, imageHeight)
        }
  }

  def getInstances() : (List[ImageTriplet], List[ImageTriplet]) = {
      val flickerTripletReader = new ImageTripletReader("data/mSprl/saiapr_tc-12/imageTriplets", "Flickr30k.majorityhead")
      val msCocoTripletReader = new ImageTripletReader("data/mSprl/saiapr_tc-12/imageTriplets", "MSCOCO.originalterm")
      val trainTriplets = flickerTripletReader.trainImageTriplets ++ msCocoTripletReader.trainImageTriplets
      val testTriplets = flickerTripletReader.testImageTriplets ++ flickerTripletReader.testImageTriplets
    (trainTriplets.toList, testTriplets.toList)
  }
}

