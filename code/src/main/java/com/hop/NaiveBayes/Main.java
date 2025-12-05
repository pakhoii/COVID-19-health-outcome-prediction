package com.hop.NaiveBayes;

import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.core.converters.ConverterUtils.DataSource;

import java.util.ArrayList;
import java.util.Random;

import static com.hop.utils.Utils.*;

public class Main {

    public static void main(String[] args) throws Exception {
        System.out.println("--- Testing Naive Bayes Classifier for Stage 1 ---");
        System.out.println("Method 1: Train/Test Split Evaluation");
        NaiveBayesStage1(trainARFF, testARFF);
        System.out.println("\nMethod 2: 10-Fold Cross-Validation with Strategic Undersampling");
        CrossValidateNaiveBayesWithUndersampling(dataFullARFF);
        System.out.println("\nMethod 3: 10-Fold Cross-Validation without Undersampling");
        CrossValidateNaiveBayes(dataFullARFF);
        System.out.println("\n--- End of Naive Bayes Testing ---");
    }

    private static void NaiveBayesStage1(String trainPath, String testPath) throws Exception {
        // Load training and testing data
        Instances train = loadData(trainPath);
        Instances test = loadData(testPath);

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

        System.out.printf("\n--- STAGE 1 (NAIVE BAYES) RESULTS ---%n");
        printEvaluationMetrics(eval, test);
        printConfusionMatrix(eval, test);

        int idx1 = test.classAttribute().indexOfValue("1");
        // fallback: if "0"/"1" not present, assume index 0 = cured, 1 = died (order from CSV)
        if (idx1 == -1) {
            idx1 = Math.min(1, test.classAttribute().numValues() - 1);
        }

        // Find test instance indices predicted as '1' (Died)
        ArrayList<Integer> suspects = new ArrayList<>();
        for (int i = 0; i < test.numInstances(); i++) {
            double pred = nb.classifyInstance(test.instance(i));
            int predIndex = (int) pred;
            if (predIndex == idx1) suspects.add(i);
        }
        System.out.printf("%nTunnel Handoff: %d patients flagged as 'At Risk' sent to Stage 2.%n", suspects.size());
    }

    private static void CrossValidateNaiveBayesWithUndersampling(String filePath) throws Exception {
        // --- LOAD DATA ---
        Instances data = loadData(filePath);

        // Set class attribute (target)
        data.setClass(data.attribute(TARGET));

        // Keep only the features we need + target
        Instances filteredData = filterAttributes(data);

        // --- 10-FOLD CROSS-VALIDATION ---
        int numFolds = 10;
        Random random = new Random(42);
        filteredData.randomize(random);
        filteredData.stratify(numFolds);

        // Lists to store metrics per fold
        ArrayList<Double> accScores = new ArrayList<>();
        ArrayList<Double> recallScores = new ArrayList<>();
        ArrayList<Double> precisionScores = new ArrayList<>();
        ArrayList<Double> f1Scores = new ArrayList<>();

        System.out.println("\nStarting 10-Fold Cross-Validation for Stage 1 (Naive Bayes)...");
        System.out.println("------------------------------------------------------------");
        System.out.printf("%-5s | %-10s | %-10s | %-10s | %-10s\n",
                "Fold", "Accuracy", "Recall", "Precision", "F1-Score");
        System.out.println("------------------------------------------------------------");

        // Perform 10-fold cross-validation
        for (int fold = 0; fold < numFolds; fold++) {
            // 1. Split Data
            Instances train = filteredData.trainCV(numFolds, fold, random);
            Instances test = filteredData.testCV(numFolds, fold);

            // 2. Undersample ONLY the Training Data (Train = 50/50, Test = Reality)
            Instances trainBalanced = strategicUndersample(train, random);

            // 3. Train Naive Bayes
            NaiveBayes nb = new NaiveBayes();
            nb.buildClassifier(trainBalanced);

            // 4. Evaluate on Test Data (Reality)
            Evaluation eval = new Evaluation(test);
            eval.evaluateModel(nb, test);

            // 5. Record Metrics (assuming class index 1 is "DIED")
            double acc = eval.pctCorrect() / 100.0;
            double recall = eval.recall(1);
            double precision = eval.precision(1);
            double f1 = eval.fMeasure(1);

            accScores.add(acc);
            recallScores.add(recall);
            precisionScores.add(precision);
            f1Scores.add(f1);

            System.out.printf("%-5d | %.4f     | %.4f     | %.4f     | %.4f\n",
                    fold + 1, acc, recall, precision, f1);
        }

        System.out.println("------------------------------------------------------------");
        System.out.println("\n--- FINAL 10-FOLD SUMMARY (STAGE 1) ---");
        System.out.printf("Mean Accuracy:  %.2f%% (+/- %.2f%%)\n",
                mean(accScores) * 100, stdev(accScores) * 100);
        System.out.printf("Mean Recall:    %.2f%% (Target: >90%%)\n",
                mean(recallScores) * 100);
        System.out.printf("Mean Precision: %.2f%% (Expected: Low)\n",
                mean(precisionScores) * 100);
        System.out.printf("Mean F1-Score:  %.4f\n", mean(f1Scores));
    }

    private static void CrossValidateNaiveBayes(String filePath) throws Exception {
        // --- LOAD DATA ---
        Instances data = loadData(filePath);

        // Set class attribute (target)
        data.setClass(data.attribute(TARGET));

        // Keep only the features we need + target
        Instances filteredData = filterAttributes(data);

        // --- 10-FOLD CROSS-VALIDATION ---
        int numFolds = 10;
        Random random = new Random(42);
        filteredData.randomize(random);
        filteredData.stratify(numFolds);

        // Lists to store metrics per fold
        ArrayList<Double> accScores = new ArrayList<>();
        ArrayList<Double> recallScores = new ArrayList<>();
        ArrayList<Double> precScores = new ArrayList<>();
        ArrayList<Double> f1Scores = new ArrayList<>();

        System.out.println("\nStarting 10-Fold Cross-Validation for Stage 1 (Naive Bayes)...");
        System.out.println("------------------------------------------------------------");
        System.out.printf("%-5s | %-10s | %-10s | %-10s | %-10s\n",
                "Fold", "Accuracy", "Recall", "Precision", "F1-Score");
        System.out.println("------------------------------------------------------------");

        // Perform 10-fold cross-validation
        for (int fold = 0; fold < numFolds; fold++) {
            // 1. Split Data
            Instances train = filteredData.trainCV(numFolds, fold, random);
            Instances test = filteredData.testCV(numFolds, fold);

            // 2. Train Naive Bayes
            NaiveBayes nb = new NaiveBayes();
            nb.buildClassifier(train);

            // 3. Evaluate on Test Data (Reality)
            Evaluation eval = new Evaluation(test);
            eval.evaluateModel(nb, test);

            // 4. Record Metrics (assuming class index 1 is "DIED")
            double acc = eval.pctCorrect() / 100.0;
            double recall = eval.recall(1);
            double precision = eval.precision(1);
            double f1 = eval.fMeasure(1);

            accScores.add(acc);
            recallScores.add(recall);
            precScores.add(precision);
            f1Scores.add(f1);

            System.out.printf("%-5d | %.4f     | %.4f     | %.4f     | %.4f\n",
                    fold + 1, acc, recall, precision, f1);
        }

        System.out.println("------------------------------------------------------------");
        System.out.println("\n--- FINAL 10-FOLD SUMMARY (STAGE 1) ---");
        System.out.printf("Mean Accuracy:  %.2f%% (+/- %.2f%%)\n",
                mean(accScores) * 100, stdev(accScores) * 100);
        System.out.printf("Mean Recall:    %.2f%% (Target: >90%%)\n",
                mean(recallScores) * 100);
        System.out.printf("Mean Precision: %.2f%% (Expected: Low)\n",
                mean(precScores) * 100);
        System.out.printf("Mean F1-Score:  %.4f\n", mean(f1Scores));
    }


}

