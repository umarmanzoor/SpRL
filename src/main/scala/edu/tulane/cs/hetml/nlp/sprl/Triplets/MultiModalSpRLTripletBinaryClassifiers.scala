package edu.tulane.cs.hetml.nlp.sprl.Triplets

import edu.illinois.cs.cogcomp.lbjava.learn.{SparseAveragedPerceptron, SparseNetworkLearner, SparsePerceptron}
import edu.illinois.cs.cogcomp.saul.classifier.Learnable
import edu.tulane.cs.hetml.nlp.sprl.Triplets.MultiModalSpRLDataModel._
import  edu.tulane.cs.hetml.nlp.sprl.Triplets.MultiModalSpRLTripletClassifiers._

object MultiModalSpRLTripletBinaryClassifiers {

  object TripletGeneralDirectionClassifier extends Learnable(triplets) {
    def label = tripletGeneralType is "direction"

    override lazy val classifier = new SparsePerceptron()

    override def feature = (tripletFeatures)
      .diff(List(tripletLmVector, tripletMatchingSegmentRelationFeatures))
  }

  object TripletGeneralRegionClassifier extends Learnable(triplets) {
    def label = tripletGeneralType is "region"

    override lazy val classifier = new SparsePerceptron()

    override def feature = (tripletFeatures)
      .diff(List(tripletLmVector, tripletMatchingSegmentRelationFeatures))
  }

  object TripletDirectionBehindClassifier extends Learnable(triplets) {
    def label = tripletDirection is "behind"

    override lazy val classifier = new SparsePerceptron()

    override def feature = (tripletFeatures)
      .diff(List(tripletMatchingSegmentRelationFeatures))
  }

  object TripletDirectionBelowClassifier extends Learnable(triplets) {
    def label = tripletDirection is "below"

    override lazy val classifier = new SparsePerceptron()

    override def feature = (tripletFeatures)
      .diff(List(tripletMatchingSegmentRelationFeatures))
  }
  object TripletDirectionLeftClassifier extends Learnable(triplets) {
    def label = tripletDirection is "left"

    override lazy val classifier = new SparsePerceptron()

    override def feature = (tripletFeatures)
      .diff(List(tripletMatchingSegmentRelationFeatures))
  }
  object TripletDirectionAboveClassifier extends Learnable(triplets) {
    def label = tripletDirection is "above"

    override lazy val classifier = new SparsePerceptron()

    override def feature = (tripletFeatures)
      .diff(List(tripletMatchingSegmentRelationFeatures))
  }
  object TripletDirectionRightClassifier extends Learnable(triplets) {
    def label = tripletDirection is "right"

    override lazy val classifier = new SparsePerceptron()

    override def feature = (tripletFeatures)
      .diff(List())
  }
  object TripletDirectionFrontClassifier extends Learnable(triplets) {
    def label = tripletDirection is "front"

    override lazy val classifier = new SparsePerceptron()

    override def feature = (tripletFeatures)
      .diff(List(tripletMatchingSegmentRelationFeatures))
  }

  object TripletRegionTPPClassifier extends Learnable(triplets) {
    def label = tripletRegion is "TPP"

    override lazy val classifier = new SparsePerceptron()

    override def feature =  (tripletFeatures)
      .diff(List(tripletLmVector))
  }

  object TripletRegionEQClassifier extends Learnable(triplets) {
    def label = tripletRegion is "EQ"

    override lazy val classifier = new SparsePerceptron()

    override def feature =  (tripletFeatures)
      .diff(List(tripletLmVector))
  }

  object TripletRegionECClassifier extends Learnable(triplets) {
    def label = tripletRegion is "EC"

    override lazy val classifier = new SparsePerceptron()

    override def feature =  (tripletFeatures)
      .diff(List(tripletLmVector))
  }

  object TripletRegionDCClassifier extends Learnable(triplets) {
    def label = tripletRegion is "DC"

    override lazy val classifier = new SparsePerceptron()

    override def feature =  (tripletFeatures)
      .diff(List(tripletLmVector, tripletMatchingSegmentRelationFeatures))
  }

  object TripletRegionPOClassifier extends Learnable(triplets) {
    def label = tripletRegion is "PO"

    override lazy val classifier = new SparsePerceptron()

    override def feature =  (tripletFeatures)
      .diff(List(tripletLmVector, tripletMatchingSegmentRelationFeatures))
  }
}
