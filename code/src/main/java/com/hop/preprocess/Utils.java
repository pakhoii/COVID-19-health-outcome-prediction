package com.hop.preprocess;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;

public class Utils {

    protected static Instances numericToNominal(Instances data) {
        NumericToNominal converter = new NumericToNominal();
        StringBuilder colsToConvert = new StringBuilder();

        for (int i = 0; i < data.numAttributes(); i++) {
            Attribute attribute = data.attribute(i);

            // If it is numeric and not AGE
            // Result: 1,2,3,... (set of indexes to put in to the filter)
            if (attribute.isNumeric() && attribute.name().equals("AGE")) {
                if (!colsToConvert.isEmpty())
                    colsToConvert.append(",");

                // Based-1 index of filter
                colsToConvert.append(i + 1);
            }
        }

        // Change from numeric to nominal after changing to (0,1) and setting missing
        if (!colsToConvert.isEmpty()) {
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

    protected static Attribute chiSquareTest(Instances data, Attribute target) {
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

    protected static double calculateMode(Instances dataset, Attribute attribute) {
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
}
