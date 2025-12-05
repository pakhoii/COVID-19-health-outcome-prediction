package com.hop.utils;

import weka.classifiers.Evaluation;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVSaver;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.filters.unsupervised.attribute.Remove;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class Utils {
    public final static String trainCSV = "data/naive_bayes/stage1_train.csv";
    public final static String testCSV = "data/naive_bayes/stage1_test.csv";

    public final static String trainARFF = "data/naive_bayes/stage1_train.arff";
    public final static String testARFF = "data/naive_bayes/stage1_test.arff";

    public final static String dataFullCSV = "data/preprocess/covid_cleaned.csv";
    public final static String dataFullARFF = "data/preprocess/covid_cleaned.arff";

    public static final String[] featuresStage1 = {"SEVERITY_INDEX", "AGE_GROUP", "SEX", "PNEUMONIA"};

    public static final String TARGET = "DIED";

    public final static int SEED = 42;


    // Convert numeric attributes to nominal except AGE
    /**
    @param data  The dataset containing numeric attributes
    @return      The dataset with specified numeric attributes converted to nominal
     */
    public static Instances numericToNominal(Instances data) {
        NumericToNominal converter = new NumericToNominal();
        StringBuilder colsToConvert = new StringBuilder();

        for (int i = 0; i < data.numAttributes(); i++) {
            Attribute attribute = data.attribute(i);

            // If it is numeric and not AGE
            // Result: 1,2,3,... (set of indexes to put in to the filter)
            if (attribute.isNumeric() && !attribute.name().equalsIgnoreCase("AGE")) {
                if (colsToConvert.length() > 0)
                    colsToConvert.append(",");

                // Based-1 index of filter
                colsToConvert.append(i + 1);
            }
        }

        // Change from numeric to nominal after changing to (0,1) and setting missing
        if (colsToConvert.length() > 0) {
            try {
                converter.setAttributeIndices(colsToConvert.toString());
                converter.setInputFormat(data);
                data = Filter.useFilter(data, converter);
            } catch (Exception e) {
                System.err.println("Error when change from numeric to nominal");
                e.printStackTrace();
                return null;
            }
        }

        return data;
    }

    // Perform Chi-Squared test to select the best attribute
    /**
    @param data      The dataset containing the attributes
    @param target    The target nominal attribute
    @return          The attribute with the highest Chi-Squared score
     */
    public static Attribute chiSquareTest(Instances data, Attribute target) {
        // Check for valid input
        if (data == null || !data.checkForAttributeType(Attribute.NOMINAL) ||
                target == null || !target.isNominal()) {
            System.err.println("Data or target attribute may be missing");
            return null;
        }

        Attribute bestAttribute = null;
        double maxChiSquareScore = -1.0;

        for (int i = 0; i < data.numAttributes(); i++) {
            Attribute currentAttr = data.attribute(i);

            // Just consider nominal data (or binary)
            if (currentAttr.equals(target) || !currentAttr.isNominal()) {
                continue;
            }

            try {
                // Construct contingency table to store observed frequencies
                int numCurrentAttrValues = currentAttr.numValues();
                int numTargetValues = target.numValues();
                double[][] contingencyTable = new double[numCurrentAttrValues][numTargetValues];

                int validInstancesCount = 0;

                for (Instance instance : data) {
                    // Skip instances that is marked missing
                    if (instance.isMissing(currentAttr) || instance.isMissing(target)) {
                        continue;
                    }
                    int valIndexCurrent = (int) instance.value(currentAttr);
                    int valIndexTarget = (int) instance.value(target);
                    contingencyTable[valIndexCurrent][valIndexTarget]++;
                    validInstancesCount++;
                }

                // If there is no valid instance, skip this attribute
                if (validInstancesCount == 0) {
                    continue;
                }

                // Calculate the total columns and rows
                double[] rowTotals = new double[numCurrentAttrValues];
                double[] colTotals = new double[numTargetValues];

                for (int r = 0; r < numCurrentAttrValues; r++) {
                    for (int c = 0; c < numTargetValues; c++) {
                        rowTotals[r] += contingencyTable[r][c];
                        colTotals[c] += contingencyTable[r][c];
                    }
                }

                // Calculate Chi-Squared test
                double chiSquareScore = 0.0;
                for (int r = 0; r < numCurrentAttrValues; r++) {
                    for (int c = 0; c < numTargetValues; c++) {
                        // Calculate Expected Frequency
                        double expected = (rowTotals[r] * colTotals[c]) / validInstancesCount;

                        // Avoid dividing by 0
                        if (expected > 0) {
                            double observed = contingencyTable[r][c];
                            double difference = observed - expected;
                            chiSquareScore += (difference * difference) / expected;
                        }
                    }
                }

                if (chiSquareScore > maxChiSquareScore) {
                    maxChiSquareScore = chiSquareScore;
                    bestAttribute = currentAttr;
                }

            } catch (Exception e) {
                System.err.println("Error when calculate Chi-Squared for the attribute: " + currentAttr.name());
                e.printStackTrace();
            }
        }

        return bestAttribute;
    }

    // Calculate mode for a nominal attribute
    /**
    @param dataset   The dataset containing the attribute
    @param attribute The nominal attribute for which to calculate the mode
    @return          The index of the mode value, or -1 if dataset is empty or
     */
    public static double calculateMode(Instances dataset, Attribute attribute) {
        if (dataset.isEmpty() || !attribute.isNominal()) {
            return -1;
        }

        int[] counts = new int[attribute.numValues()];
        for (Instance inst : dataset) {
            if (!inst.isMissing(attribute)) {
                counts[(int) inst.value(attribute)]++;
            }
        }

        int maxCount = -1;
        int modeIndex = -1;
        for (int i = 0; i < counts.length; i++) {
            if (counts[i] > maxCount) {
                maxCount = counts[i];
                modeIndex = i;
            }
        }

        return modeIndex;
    }

    // Save Instances to CSV
    /**
    @param data      The Instances data to save
    @param outPath   The output file path for the CSV
    @throws Exception If an error occurs during saving
     */
    public static void saveCSV(Instances data, String outPath) throws Exception {
        try {
            CSVSaver saver = new CSVSaver();
            saver.setInstances(data);
            saver.setFile(new File(outPath));
            saver.writeBatch();
        }
        catch (Exception e) {
            System.err.println("Error when saving CSV to " + outPath);
            throw e;
        }
    }

    // Save Instances to ARFF
    /**
    @param data      The Instances data to save
    @param outPath   The output file path for the ARFF
    @throws Exception If an error occurs during saving
     */
    public static void saveARFF(Instances data, String outPath) throws Exception {
        try {
            ArffSaver arffSaver = new ArffSaver();
            arffSaver.setInstances(data);
            arffSaver.setFile(new File(outPath));
            arffSaver.writeBatch();
        }
        catch (Exception e) {
            System.err.println("Error when saving ARFF to " + outPath);
            throw e;
        }
    }

    // Load data from CSV file
    /**
     * Load data from a CSV file.
     * @param inputFilePath The path to the input CSV file
     * @return Instances object containing the dataset
     */
    public static Instances loadData(String inputFilePath) throws Exception {
        try {
            if (inputFilePath == null || inputFilePath.isEmpty()) {
                throw new IllegalArgumentException("Input file path cannot be null or empty");
            }
        } catch (Exception e) {
            System.err.println("Error when loading data from " + inputFilePath);
            throw e;
        }
        System.out.println("Loading data from " + inputFilePath + "...");
        DataSource source = new DataSource(inputFilePath);
        return source.getDataSet();
    }

    // Print evaluation metrics
    /**
     * Print evaluation metrics including accuracy, precision, recall, and F1-score for each class.
     * @param eval The Evaluation object containing the evaluation results
     * @param data The Instances object containing the dataset
     * @throws Exception If an error occurs during metric calculation
     */
    public static void printEvaluationMetrics(Evaluation eval, Instances data) throws Exception {
        if (eval == null || data == null) {
            throw new IllegalArgumentException("Evaluation or data cannot be null");
        }

        System.out.printf("Accuracy: %.2f%%%n", eval.pctCorrect());
        for (int i = 0; i < data.classAttribute().numValues(); i++) {
            String label = data.classAttribute().value(i);
            String name;
            if (label.equals("0")) {
                name = "Cured";
            } else if (label.equals("1")) {
                name = "Died";
            } else {
                name = "Class";
            }
            double precision = eval.precision(i);
            double rec  = eval.recall(i);
            double f1   = eval.fMeasure(i);
            System.out.printf("%s %s: Precision=%.3f  Recall=%.3f  F1=%.3f%n", name,
                    label, precision, rec, f1);
        }
    }

    /**
     * Print confusion matrix along with TP, FP, TN, FN counts.
     * @param eval The Evaluation object containing the evaluation results
     * @param dataNom The Instances object containing the dataset
     * @throws Exception If an error occurs during metric calculation
     */
    public static void printConfusionMatrix(Evaluation eval, Instances dataNom) throws Exception {
        if (eval == null || dataNom == null) {
            throw new IllegalArgumentException("Evaluation or data cannot be null");
        }
        // Confusion matrix and TP/FP/TN/FN mapping (try to map '0'->cured, '1'->died)
        double[][] cm = eval.confusionMatrix();
        int idx0 = dataNom.classAttribute().indexOfValue("0");
        int idx1 = dataNom.classAttribute().indexOfValue("1");
        if (idx0 == -1 || idx1 == -1) {
            idx0 = 0;
            idx1 = Math.min(1, dataNom.classAttribute().numValues() - 1);
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

        System.out.println("\nConfusion Matrix:");
        System.out.println("      Predict 0   Predict 1");
        System.out.printf("Cured 0   %-7.0f %-7.0f%n", cm[0][0], cm[0][1]);
        System.out.printf("Died  1   %-7.0f %-7.0f%n", cm[1][0], cm[1][1]);

    }

    /**
     * Filter dataset to keep only specified features + target
     * @param data  The original dataset
     * @return      The filtered dataset with only specified features and target
     */
    public static Instances filterAttributes(Instances data) throws Exception {
        // Create list of indices to keep
        ArrayList<Integer> indicesToKeep = new ArrayList<>();

        // Add target index
        indicesToKeep.add(data.attribute(TARGET).index());

        // Add feature indices
        for (String feature : featuresStage1) {
            indicesToKeep.add(data.attribute(feature).index());
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
     * Strategic undersampling to balance classes (50/50)
     * Matches the died count by sampling from cured
     * @param train     The training dataset
     * @param random    Random object for shuffling
     * @return          The balanced dataset after undersampling
     */
    public static Instances strategicUndersample(Instances train, Random random) {
        // Separate instances by class
        Instances died = new Instances(train, 0);
        Instances cured = new Instances(train, 0);

        int classIndex = train.classIndex();
        for (int i = 0; i < train.numInstances(); i++) {
            if (train.instance(i).classValue() == 1.0) {
                died.add(train.instance(i));
            } else {
                cured.add(train.instance(i));
            }
        }

        // Sample cured to match died count
        int targetSize = died.numInstances();
        cured.randomize(random);

        Instances curedSampled = new Instances(cured, 0);
        for (int i = 0; i < Math.min(targetSize, cured.numInstances()); i++) {
            curedSampled.add(cured.instance(i));
        }

        // Combine and shuffle
        Instances balanced = new Instances(died);
        for (int i = 0; i < curedSampled.numInstances(); i++) {
            balanced.add(curedSampled.instance(i));
        }
        balanced.randomize(random);

        return balanced;
    }

    /**
     * Calculate mean of a list of doubles
     * @param values    The list of double values
     * @return          The mean of the values
     */
    public static double mean(ArrayList<Double> values) {
        double sum = 0.0;
        for (double val : values) {
            sum += val;
        }
        return sum / values.size();
    }

    /**
     * Calculate standard deviation of a list of doubles
     * @param values    The list of double values
     * @return          The standard deviation of the values
     */
    public static double stdev(ArrayList<Double> values) {
        double avg = mean(values);
        double sumSquaredDiff = 0.0;
        for (double val : values) {
            sumSquaredDiff += Math.pow(val - avg, 2);
        }
        return Math.sqrt(sumSquaredDiff / values.size());
    }

}
