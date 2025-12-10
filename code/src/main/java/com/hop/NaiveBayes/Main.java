package com.hop.NaiveBayes;

import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.SerializationHelper;

import java.util.ArrayList;
import java.util.Random;

import static com.hop.utils.Utils.*;

public class Main {

    public static void main(String[] args) throws Exception {
        System.out.println("--- Testing Naive Bayes Classifier for Stage 1 ---");
        System.out.println("\nMethod 1: 10-Fold Cross-Validation with Strategic Undersampling");
        CrossValidateNaiveBayesWithUndersampling(dataFullBalanceARFF);
        System.out.println("\nMethod 2: 10-Fold Cross-Validation without Undersampling");
        CrossValidateNaiveBayes(dataFullBalanceARFF);
        System.out.println("\n--- End of Naive Bayes Testing ---");
    }

    public static void CrossValidateNaiveBayesWithUndersampling(String filePath) throws Exception {
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
        ArrayList<Double> buildTime = new ArrayList<>();
        // Perform 10-fold cross-validation
        for (int fold = 0; fold < numFolds; fold++) {
            // 1. Split Data
            Instances train = filteredData.trainCV(numFolds, fold, random);
            Instances test = filteredData.testCV(numFolds, fold);

            // 2. Undersample ONLY the Training Data (Train = 50/50, Test = Reality)
            Instances trainBalanced = strategicUndersample(train, random);

            // 3. Train Naive Bayes
            NaiveBayes nb = new NaiveBayes();
            long startBuild = System.nanoTime();
            nb.buildClassifier(trainBalanced);
            long endBuild = System.nanoTime();
            buildTime.add((endBuild - startBuild) / 1e9); // in seconds
            if (fold == 4) SerializationHelper.write(saveModelStage1UnderSampling, nb);

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

        printWekaStyleSummary(accScores, recallScores, precisionScores, f1Scores, filteredData);
        System.out.println("\nAverage Model Build Time per Fold: " + mean(buildTime) + " seconds");
    }

    public static void CrossValidateNaiveBayes(String filePath) throws Exception {
        // --- LOAD DATA ---
        Instances data = loadData(filePath);

        // Set class attribute
        data.setClass(data.attribute(TARGET));

        // Keep only selected features
        Instances filteredData = filterAttributes(data);

        // --- 10-FOLD CROSS VALIDATION ---
        int numFolds = 10;
        Random random = new Random(42);
        filteredData.randomize(random);
        filteredData.stratify(numFolds);

        ArrayList<Double> accScores = new ArrayList<>();
        ArrayList<Double> recallScores = new ArrayList<>();
        ArrayList<Double> precisionScores = new ArrayList<>();
        ArrayList<Double> f1Scores = new ArrayList<>();
        ArrayList<Double> buildTime = new ArrayList<>();
        System.out.println("\nStarting 10-Fold Cross-Validation (Naive Bayes, No Undersampling)...");
        System.out.println("------------------------------------------------------------");
        System.out.printf("%-5s | %-10s | %-10s | %-10s | %-10s\n",
                "Fold", "Accuracy", "Recall", "Precision", "F1-Score");
        System.out.println("------------------------------------------------------------");

        // Perform cross-validation
        for (int fold = 0; fold < numFolds; fold++) {
            // 1. Split Data
            Instances train = filteredData.trainCV(numFolds, fold, random);
            Instances test = filteredData.testCV(numFolds, fold);

            // 2. Train Naive Bayes DIRECTLY (NO BALANCING)
            NaiveBayes nb = new NaiveBayes();
            long startBuild = System.nanoTime();
            nb.buildClassifier(train);
            long endBuild = System.nanoTime();
            buildTime.add((endBuild - startBuild) / 1e9); // in seconds
            if (fold == 4) SerializationHelper.write(saveModelStage1, nb);

            // 3. Evaluate on test data
            Evaluation eval = new Evaluation(test);
            eval.evaluateModel(nb, test);

            // 4. Extract metrics (Assume class index 1 is "DIED")
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

        printWekaStyleSummary(accScores, recallScores, precisionScores, f1Scores, filteredData);
        System.out.println("\nAverage Model Build Time per Fold: " + mean(buildTime) + " seconds");
    }



}

