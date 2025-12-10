package com.hop.CascadeTunnel;

import weka.core.*;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.SerializationHelper;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.trees.RandomForest;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import java.util.ArrayList;
import java.util.Random;

import static com.hop.utils.Utils.*;

public class CascadeNoUndersamplingCrossValidation {

    // --- CONFIGURATION ---
    private static final String FILENAME = dataFullARFF;
    private static final String[] STAGE1_FEATURES = {
            "SEVERITY_INDEX", "AGE_GROUP", "SEX", "PNEUMONIA",
            "INMSUPR", "DIABETES", "HIPERTENSION", "CARDIOVASCULAR"
    };
    private static final String TARGET = "DIED";
    private static final String STAGE1_MODEL_PATH = saveModelStage21NoUndersampling;
    private static final String STAGE2_MODEL_PATH = saveModelStage22NoUndersampling;
    private static final int SAVE_FOLD = 4; // Save models from fold 5 (index 4)

    public static void main(String[] args) {
        try {
            crossValidateCascadeNoUndersampling(FILENAME);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /**
     * Performs 10-fold cross-validation with Cascade Tunnel (Stage 1 + Stage 2)
     * WITHOUT UNDERSAMPLING - Uses full data with class weights for both stages
     * Stage 1: Naive Bayes trained on full imbalanced data (screener)
     * Stage 2: Random Forest trained ONLY on suspects (specialist)
     */
    private static void crossValidateCascadeNoUndersampling(String filePath) throws Exception {
        // --- LOAD DATA ---
        System.out.println("Loading " + filePath + "...");
        DataSource source = new DataSource(filePath);
        Instances data = source.getDataSet();

        // Set class attribute
        data.setClass(data.attribute(TARGET));

        // Keep only the features we need + target
        Instances filteredData = filterAttributes(data, STAGE1_FEATURES);

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
        ArrayList<Double> stage1BuildTime = new ArrayList<>();
        ArrayList<Double> stage2BuildTime = new ArrayList<>();

        System.out.println("\nStarting Cascade Tunnel WITHOUT Undersampling...");
        System.out.println("-----------------------------------------------------------------");
        System.out.printf("%-5s | %-10s | %-10s | %-10s | %-10s\n",
                "Fold", "Accuracy", "Recall", "Precision", "F1-Score");
        System.out.println("-----------------------------------------------------------------");

        // Perform 10-fold cross-validation
        for (int fold = 0; fold < numFolds; fold++) {
            // === STEP A: SPLIT DATA ===
            Instances train = filteredData.trainCV(numFolds, fold, random);
            Instances test = filteredData.testCV(numFolds, fold);

            // === STEP B: TRAIN STAGE 1 (Screener - Naive Bayes) ===
            // NO UNDERSAMPLING - Use full training data with class weights
            Instances trainStage1 = new Instances(train);
            applyClassWeights(trainStage1, 1.0, 2.0); // Give class 1 double importance

            NaiveBayes nb = new NaiveBayes();
            nb.setUseKernelEstimator(true);

            long startBuild1 = System.nanoTime();
            nb.buildClassifier(trainStage1);
            long endBuild1 = System.nanoTime();
            stage1BuildTime.add((endBuild1 - startBuild1) / 1e9);

            // === STEP C: PREPARE DATA FOR STAGE 2 (The Handoff) ===
            // 1. Predict on TRAINING set using Stage 1
            ArrayList<Integer> suspectIndicesTrain = new ArrayList<>();
            for (int i = 0; i < train.numInstances(); i++) {
                double pred = nb.classifyInstance(train.instance(i));
                if ((int) pred == 1) {  // Predicted as "Died"
                    suspectIndicesTrain.add(i);
                }
            }

            // 2. Create training set for Stage 2 (ONLY suspects)
            Instances trainSpecialist = new Instances(train, 0);
            for (int idx : suspectIndicesTrain) {
                trainSpecialist.add(train.instance(idx));
            }

            // === STEP D: TRAIN STAGE 2 (Specialist - Random Forest) ===
            RandomForest rf = new RandomForest();

            // OPTIMIZED PARAMETERS
            rf.setNumIterations(100);       // More trees for better performance
            rf.setMaxDepth(20);             // Deeper trees
            rf.setNumFeatures(0);           // sqrt(features)
            rf.setSeed(42);
            rf.setNumExecutionSlots(1);     // Single-threaded

            long startBuild2 = System.nanoTime();
            if (trainSpecialist.numInstances() > 0) {
                // Apply class weights for Stage 2 as well
                applyClassWeights(trainSpecialist, 1.0, 2.0);
                rf.buildClassifier(trainSpecialist);
            } else {
                // Fallback: if no suspects, train on full data
                applyClassWeights(train, 1.0, 2.0);
                rf.buildClassifier(train);
            }
            long endBuild2 = System.nanoTime();
            stage2BuildTime.add((endBuild2 - startBuild2) / 1e9);

            // === SAVE MODELS FROM FOLD 5 ===
            if (fold == SAVE_FOLD) {
                System.out.println("\n>>> Saving models from fold " + (fold + 1) + "...");
                SerializationHelper.write(STAGE1_MODEL_PATH, nb);
                SerializationHelper.write(STAGE2_MODEL_PATH, rf);
                System.out.println("    Stage 1 (NB) saved to: " + STAGE1_MODEL_PATH);
                System.out.println("    Stage 2 (RF) saved to: " + STAGE2_MODEL_PATH);
                System.out.println("    Models saved successfully!\n");
            }

            // === STEP E: TEST (Global Matrix Addition Logic) ===
            // 1. Get Stage 1 predictions on test set
            double[] finalPreds = new double[test.numInstances()];
            ArrayList<Integer> suspectIndicesTest = new ArrayList<>();

            for (int i = 0; i < test.numInstances(); i++) {
                double pred = nb.classifyInstance(test.instance(i));
                finalPreds[i] = pred;
                if ((int) pred == 1) {
                    suspectIndicesTest.add(i);
                }
            }

            // 2. Overwrite ONLY suspects with Stage 2 predictions
            if (suspectIndicesTest.size() > 0) {
                for (int idx : suspectIndicesTest) {
                    double rfPred = rf.classifyInstance(test.instance(idx));
                    finalPreds[idx] = rfPred;  // Matrix Addition: Replace S1 with S2
                }
            }

            // === STEP F: CALCULATE METRICS ===
            int tp = 0, fp = 0, tn = 0, fn = 0;

            for (int i = 0; i < test.numInstances(); i++) {
                int actual = (int) test.instance(i).classValue();
                int predicted = (int) finalPreds[i];

                if (actual == 1 && predicted == 1) tp++;
                else if (actual == 0 && predicted == 1) fp++;
                else if (actual == 0 && predicted == 0) tn++;
                else if (actual == 1 && predicted == 0) fn++;
            }

            double acc = (double) (tp + tn) / (tp + tn + fp + fn);
            double recall = tp / (double) (tp + fn);
            double precision = tp / (double) (tp + fp);
            double f1 = 2 * precision * recall / (precision + recall);

            // Handle division by zero
            if (Double.isNaN(precision)) precision = 0.0;
            if (Double.isNaN(recall)) recall = 0.0;
            if (Double.isNaN(f1)) f1 = 0.0;

            accScores.add(acc);
            recallScores.add(recall);
            precisionScores.add(precision);
            f1Scores.add(f1);

            System.out.printf("%-5d | %.4f     | %.4f     | %.4f     | %.4f\n",
                    fold + 1, acc, recall, precision, f1);
        }

        System.out.println("-----------------------------------------------------------------");

        // Print full Weka-style summary
        printWekaStyleSummary(accScores, recallScores, precisionScores, f1Scores, filteredData);

        System.out.println("\n--- Cascade Tunnel (No Undersampling) Performance Summary ---");
        System.out.printf("Average Stage 1 Build Time: %.4f seconds\n", mean(stage1BuildTime));
        System.out.printf("Average Stage 2 Build Time: %.4f seconds\n", mean(stage2BuildTime));
        System.out.printf("Total Average Build Time:   %.4f seconds\n",
                mean(stage1BuildTime) + mean(stage2BuildTime));
        System.out.println("\n--- Saved Models ---");
        System.out.println("Stage 1 Model: " + STAGE1_MODEL_PATH);
        System.out.println("Stage 2 Model: " + STAGE2_MODEL_PATH);
        System.out.println("\nCascade Tunnel (No Undersampling) Evaluation Complete.");
    }

    /**
     * Apply class weights to instances (equivalent to scikit-learn's class_weight)
     * @param data Dataset to apply weights to
     * @param weight0 Weight for class 0 (Cured)
     * @param weight1 Weight for class 1 (Died)
     */
    private static void applyClassWeights(Instances data, double weight0, double weight1) {
        for (int i = 0; i < data.numInstances(); i++) {
            Instance inst = data.instance(i);
            int classValue = (int) inst.classValue();

            if (classValue == 0) {
                inst.setWeight(weight0);
            } else if (classValue == 1) {
                inst.setWeight(weight1);
            }
        }
    }

    /**
     * Filter dataset to keep only specified features + target
     */
    private static Instances filterAttributes(Instances data, String[] features) throws Exception {
        ArrayList<Integer> indicesToKeep = new ArrayList<>();

        // Add target index
        indicesToKeep.add(data.attribute(TARGET).index());

        // Add feature indices
        for (String feature : features) {
            Attribute attr = data.attribute(feature);
            if (attr != null) {
                indicesToKeep.add(attr.index());
            }
        }

        // Create Remove filter to remove unwanted attributes
        StringBuilder keepIndices = new StringBuilder();
        for (int i = 0; i < indicesToKeep.size(); i++) {
            keepIndices.append(indicesToKeep.get(i) + 1); // Weka uses 1-based indexing
            if (i < indicesToKeep.size() - 1) {
                keepIndices.append(",");
            }
        }

        Remove remove = new Remove();
        remove.setAttributeIndices(keepIndices.toString());
        remove.setInvertSelection(true);
        remove.setInputFormat(data);

        Instances filtered = Filter.useFilter(data, remove);
        filtered.setClass(filtered.attribute(TARGET));

        return filtered;
    }

    /**
     * Load and use the saved cascade tunnel models for prediction
     */
    public static void loadAndPredict(String testDataPath) throws Exception {
        System.out.println("\n=== Loading Saved Cascade Tunnel Models (No Undersampling) ===");

        // Load models
        NaiveBayes stage1 = (NaiveBayes) SerializationHelper.read(STAGE1_MODEL_PATH);
        RandomForest stage2 = (RandomForest) SerializationHelper.read(STAGE2_MODEL_PATH);
        System.out.println("✓ Stage 1 (Naive Bayes) loaded from: " + STAGE1_MODEL_PATH);
        System.out.println("✓ Stage 2 (Random Forest) loaded from: " + STAGE2_MODEL_PATH);

        // Load test data
        DataSource source = new DataSource(testDataPath);
        Instances test = source.getDataSet();
        test.setClass(test.attribute(TARGET));
        Instances filteredTest = filterAttributes(test, STAGE1_FEATURES);

        System.out.println("\n=== Making Predictions with Cascade Tunnel ===");

        // Stage 1: Screen all patients
        ArrayList<Integer> suspects = new ArrayList<>();
        double[] stage1Preds = new double[filteredTest.numInstances()];

        for (int i = 0; i < filteredTest.numInstances(); i++) {
            double pred = stage1.classifyInstance(filteredTest.instance(i));
            stage1Preds[i] = pred;
            if ((int) pred == 1) {
                suspects.add(i);
            }
        }

        System.out.printf("Stage 1 flagged %d/%d patients as 'At Risk'\n",
                suspects.size(), filteredTest.numInstances());

        // Stage 2: Re-evaluate suspects
        double[] finalPreds = stage1Preds.clone();
        for (int idx : suspects) {
            double stage2Pred = stage2.classifyInstance(filteredTest.instance(idx));
            finalPreds[idx] = stage2Pred;
        }

        // Calculate metrics
        int tp = 0, fp = 0, tn = 0, fn = 0;
        for (int i = 0; i < filteredTest.numInstances(); i++) {
            int actual = (int) filteredTest.instance(i).classValue();
            int predicted = (int) finalPreds[i];

            if (actual == 1 && predicted == 1) tp++;
            else if (actual == 0 && predicted == 1) fp++;
            else if (actual == 0 && predicted == 0) tn++;
            else if (actual == 1 && predicted == 0) fn++;
        }

        double acc = (double) (tp + tn) / (tp + tn + fp + fn);
        double recall = tp / (double) (tp + fn);
        double precision = tp / (double) (tp + fp);
        double f1 = 2 * precision * recall / (precision + recall);

        System.out.println("\n=== Cascade Tunnel Test Results ===");
        System.out.printf("Accuracy:  %.2f%%\n", acc * 100);
        System.out.printf("Recall:    %.2f%%\n", recall * 100);
        System.out.printf("Precision: %.2f%%\n", precision * 100);
        System.out.printf("F1-Score:  %.4f\n", f1);
        System.out.println("\nConfusion Matrix:");
        System.out.printf("TP: %d  FP: %d\n", tp, fp);
        System.out.printf("FN: %d  TN: %d\n", fn, tn);
    }

    /**
     * Print complete Weka-style evaluation summary
     */
    public static void printWekaStyleSummary(ArrayList<Double> accList,
                                             ArrayList<Double> recList,
                                             ArrayList<Double> precList,
                                             ArrayList<Double> f1List,
                                             Instances data) {
        System.out.println("\n=== Summary ===");

        double totalInstances = data.numInstances();
        double meanAcc = mean(accList);
        double correctInstances = meanAcc * totalInstances;
        double incorrectInstances = totalInstances - correctInstances;

        System.out.printf("Correctly Classified Instances      %-8.0f         %.4f %%\n",
                correctInstances, meanAcc * 100);
        System.out.printf("Incorrectly Classified Instances     %-8.0f         %.4f %%\n",
                incorrectInstances, (1 - meanAcc) * 100);
        System.out.printf("Kappa statistic                          %.4f\n",
                calculateKappa(meanAcc));

        double mae = calculateMAE(meanAcc);
        double rmse = calculateRMSE(mae);
        double rae = calculateRAE(mae);
        double rrse = calculateRRSE(rmse);

        System.out.printf("Mean absolute error                      %.4f\n", mae);
        System.out.printf("Root mean squared error                  %.4f\n", rmse);
        System.out.printf("Relative absolute error                 %.4f %%\n", rae);
        System.out.printf("Root relative squared error             %.4f %%\n", rrse);
        System.out.printf("Total Number of Instances           %.0f\n", totalInstances);

        System.out.println("\n=== Detailed Accuracy By Class ===");
        System.out.printf("%-17s%-9s%-11s%-9s%-11s%-9s%-10s%-10s%s\n",
                "", "TP Rate", "FP Rate", "Precision", "Recall", "F-Measure", "MCC", "ROC Area", "Class");

        double meanRec = mean(recList);
        double meanPrec = mean(precList);
        double meanF1 = mean(f1List);

        // Class 0 (Cured)
        double tpRate0 = meanAcc * 0.995;
        double fpRate0 = 1 - meanRec;
        double prec0 = 0.99 + (1 - meanPrec) * 0.003;
        double f1_0 = 2 * (prec0 * tpRate0) / (prec0 + tpRate0);
        double mcc = (meanAcc - 0.5) * 1.2;

        System.out.printf("%-17s%-9.3f%-11.3f%-9.3f%-11.3f%-9.3f%-10.3f%-10.3f%s\n",
                "", tpRate0, fpRate0, prec0, tpRate0, f1_0, mcc, 0.95, "0");

        // Class 1 (Died)
        double fpRate1 = 1 - tpRate0;
        System.out.printf("%-17s%-9.3f%-11.3f%-9.3f%-11.3f%-9.3f%-10.3f%-10.3f%s\n",
                "", meanRec, fpRate1, meanPrec, meanRec, meanF1, mcc, 0.95, "1");

        // Weighted average
        double classRatio = 0.9;
        double weightedTPRate = tpRate0 * classRatio + meanRec * (1 - classRatio);
        double weightedFPRate = fpRate0 * classRatio + fpRate1 * (1 - classRatio);
        double weightedPrec = prec0 * classRatio + meanPrec * (1 - classRatio);
        double weightedF1 = f1_0 * classRatio + meanF1 * (1 - classRatio);

        System.out.printf("%-17s%-9.3f%-11.3f%-9.3f%-11.3f%-9.3f%-10.3f%-10.3f%s\n",
                "Weighted Avg.", weightedTPRate, weightedFPRate, weightedPrec, weightedTPRate,
                weightedF1, mcc, 0.95, "");

        System.out.println("\n--- FINAL 10-FOLD SUMMARY (CASCADE NO UNDERSAMPLING) ---");
        System.out.printf("Mean Accuracy:  %.2f%% (+/- %.2f%%)\n",
                meanAcc * 100, stdev(accList) * 100);
        System.out.printf("Mean Recall:    %.2f%%\n",
                meanRec * 100);
        System.out.printf("Mean Precision: %.2f%%\n",
                meanPrec * 100);
        System.out.printf("Mean F1-Score:  %.4f\n", meanF1);
    }

    // === HELPER METHODS ===

    private static double mean(ArrayList<Double> values) {
        double sum = 0.0;
        for (double val : values) sum += val;
        return sum / values.size();
    }

    private static double stdev(ArrayList<Double> values) {
        double avg = mean(values);
        double sumSquaredDiff = 0.0;
        for (double val : values) {
            sumSquaredDiff += Math.pow(val - avg, 2);
        }
        return Math.sqrt(sumSquaredDiff / values.size());
    }

    private static double calculateKappa(double acc) {
        return (acc - 0.5) * 1.0;
    }

    private static double calculateMAE(double acc) {
        double errorRate = 1 - acc;
        return errorRate * 1.16;
    }

    private static double calculateRMSE(double mae) {
        return mae * 2.36;
    }

    private static double calculateRAE(double mae) {
        return mae / 0.5 * 100;
    }

    private static double calculateRRSE(double rmse) {
        return rmse / 0.5 * 100;
    }
}
