package com.hop.NaiveBayes;

import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;

import java.io.File;
import java.util.ArrayList;
import java.util.Random;

public class Main {
    final static String trainCSV = "data/naive_bayes/stage1_train.csv";
    final static String testCSV  = "data/naive_bayes/stage1_test.csv";
    final static String trainARFF = "data/naive_bayes/stage1_train.arff";
    final static String testARFF  = "data/naive_bayes/stage1_test.arff";
    static int seed = 42;

    public static void main(String[] args) throws Exception {
        NaiveBayesStage1(trainARFF, testARFF);
    }

    private static void NaiveBayesStage1(String trainPath, String testPath) throws Exception {
        // Load CSVs
        ArffLoader loader = new ArffLoader();
        loader.setSource(new File(trainPath));
        Instances train = loader.getDataSet();

        loader = new ArffLoader();
        loader.setSource(new File(testPath));
        Instances test = loader.getDataSet();

        if (train.numAttributes() == 0 || test.numAttributes() == 0) {
            throw new IllegalArgumentException("Empty dataset(s). Check CSV paths: `"+trainPath+"`, `"+testPath+"`.");
        }

        // Set class index to last attribute (expects DIED to be last in the CSV)
        train.setClassIndex(train.numAttributes() - 1);
        test.setClassIndex(test.numAttributes() - 1);

        // Convert numeric attributes to nominal (categorical) using the train header
        NumericToNominal n2n = new NumericToNominal();
        n2n.setAttributeIndices("first-last");
        n2n.setInputFormat(train);
        train = Filter.useFilter(train, n2n);
        // Apply the same conversion to test (using train's input format ensures consistent nominal mapping)
        test = Filter.useFilter(test, n2n);

        // Ensure class index is still correctly set after filtering
        train.setClassIndex(train.numAttributes() - 1);
        test.setClassIndex(test.numAttributes() - 1);

        System.out.println("Training rows: " + train.numInstances());
        System.out.println("Testing rows:  " + test.numInstances());

        // Build Naive Bayes classifier
        NaiveBayes nb = new NaiveBayes();
        nb.buildClassifier(train);

        // Evaluate
        Evaluation eval = new Evaluation(train);
        eval.evaluateModel(nb, test);

        double accuracy = eval.pctCorrect(); // percentage
        System.out.printf("\n--- STAGE 1 (NAIVE BAYES) RESULTS ---%n");
        System.out.printf("Accuracy: %.2f%%%n", accuracy);

        // Print per-class metrics
        System.out.println("\nClassification Report:");
        for (int i = 0; i < test.classAttribute().numValues(); i++) {
            String label = test.classAttribute().value(i);
            double prec = eval.precision(i);
            double rec  = eval.recall(i);
            double f1   = eval.fMeasure(i);
            System.out.printf("Class %s: Precision=%.3f  Recall=%.3f  F1=%.3f%n", label, prec, rec, f1);
        }

        // Confusion matrix and TP/FP/TN/FN for mapping 0=Cured, 1=Died (assumes those nominal values exist)
        double[][] cm = eval.confusionMatrix();

        int idx0 = test.classAttribute().indexOfValue("0");
        int idx1 = test.classAttribute().indexOfValue("1");
        // fallback: if "0"/"1" not present, assume index 0 = cured, 1 = died (order from CSV)
        if (idx0 == -1 || idx1 == -1) {
            idx0 = 0;
            idx1 = Math.min(1, test.classAttribute().numValues() - 1);
        }

        int tn = (int) cm[idx0][idx0];
        int fp = (int) cm[idx0][idx1];
        int fn = (int) cm[idx1][idx0];
        int tp = (int) cm[idx1][idx1];

        System.out.println("\nConfusion Matrix:");
        System.out.printf("True Positive (TP): %d%n", tp);
        System.out.printf("False Positive (FP): %d%n", fp);
        System.out.printf("True Negative (TN): %d%n", tn);
        System.out.printf("False Negative (FN): %d%n", fn);

        // Find test instance indices predicted as '1' (Died)
        ArrayList<Integer> suspects = new ArrayList<>();
        for (int i = 0; i < test.numInstances(); i++) {
            double pred = nb.classifyInstance(test.instance(i));
            int predIndex = (int) pred;
            if (predIndex == idx1) suspects.add(i); // i is test row index (0-based)
        }
        System.out.printf("%nTunnel Handoff: %d patients flagged as 'At Risk' sent to Stage 2.%n", suspects.size());

        // Optional: reproducibility seed used for any random operations (not needed here but kept)
        Random r = new Random(seed);
    }


}

