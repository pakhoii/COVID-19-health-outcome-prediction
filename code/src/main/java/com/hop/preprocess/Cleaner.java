package com.hop.preprocess;

import weka.core.Instances;
import weka.core.Instance;
import weka.core.Attribute;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.instance.RemoveWithValues;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;

import java.util.ArrayList;

public class Cleaner {
    public Cleaner() {}

    public Instances preprocess(Instances data) {
        // Remove patients who negative with COVID
        RemoveWithValues filterRemoveNonCovid = new RemoveWithValues();
        filterRemoveNonCovid.setAttributeIndex(
            String.valueOf(
                // Need to plus 1 since weka filter index is 1-based
                data.attribute("CLASIFFICATION_FINAL").index() + 1
            )
        );

        // Filter and remove any row that has classification > 4 (non covid patient)
        filterRemoveNonCovid.setNominalIndices("4-last");

        try {
            filterRemoveNonCovid.setInputFormat(data);
            data = Filter.useFilter(data, filterRemoveNonCovid);
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }

        // Feature Extraction: DIED - extract from DATE_DIED
        ArrayList<String> diedValue = new ArrayList<>();
        diedValue.add("0"); // alive
        diedValue.add("1"); // died

        Attribute diedAttribute = new Attribute("DIED", diedValue); 
        data.insertAttributeAt(diedAttribute, data.numAttributes());

        int dateDiedIndex = data.attribute("DATE_DIED").index();
        int diedIndex = data.attribute("DIED").index();

        for (Instance instance : data) {
            String date = instance.stringValue(dateDiedIndex);
            if(date.equals("9999-99-99")) 
                instance.setValue(diedIndex, "0");
            else 
                instance.setValue(diedIndex, "1");
        }

        // Remove unnessary columns (CLASIFFICATION_FINAL and DATE_DIED)
        Remove filterRemoveColumns = new Remove();
        
        int classificationIndex = data.attribute("CLASIFFICATION_FINAL").index() + 1; // Weka index
        dateDiedIndex++; // Weka index

        filterRemoveColumns.setAttributeIndices(classificationIndex + "," + dateDiedIndex);
        
        try {
            filterRemoveColumns.setInputFormat(data);
            data = Filter.useFilter(data, filterRemoveColumns);
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }

        // Standardize binary value and marking missing
        int sexIndex = data.attribute("SEX").index();
        int pregnantIndex = data.attribute("PREGNANT").index();

        for (Instance instance : data) {
            for (int i = 0; i < instance.numAttributes(); i++) {
                // Change binary values from (1,2) to (0,1)
                if (data.attribute(i).isNumeric())
                    if (instance.value(i) == 2.0)
                        instance.setValue(i, 0);
                    
                // Set missing value
                if (data.attribute(i).isNominal() && (instance.value(i) == 97 || instance.value(i) == 99)) 
                    instance.setMissing(i);
            }

            // Set PREGNANT to 0 when SEX is 0 (male)
            if (instance.value(sexIndex) == 0.0 && instance.isMissing(pregnantIndex))
                instance.setValue(pregnantIndex, 0);
        }

        // Find the need to be filled columns
        for (int i = 0; i < data.numAttributes(); i++) {
            Attribute targetAttr = data.attribute(i);

            // If the attribute is not nominal/binary, we skip it
            if (!targetAttr.isNominal() || data.attributeStats(i).missingCount == 0) {
                continue;
            }

           
            // Find best attribute based on chiSquare test
            Attribute bestAttr = chiSquareTest(data, targetAttr);

            if (bestAttr == null) {
                continue;
            }

            // Group by based on bestAttr and calculate mode for each group
            // Using HashMap to store the mode for each value of bestAttr
            // Key: value of bestAttr (as Double)
            // Value: mode of targetAttr in that group (as Double)
            java.util.Map<Double, Double> groupModes = new java.util.HashMap<>();

            // Loop through all possible values of bestAttr to create groups
            for (int j = 0; j < bestAttr.numValues(); j++) {
                // Create a subset of data containing only instances belonging to this group
                Instances subset = new Instances(data, 0);
                for (Instance inst : data) {
                    if (!inst.isMissing(bestAttr) && inst.value(bestAttr) == j) {
                        subset.add(inst);
                    }
                }

                // If the group is not empty, calculate the mode of targetAttr in that group
                if (!subset.isEmpty()) {
                    double mode = calculateMode(subset, targetAttr);
                    groupModes.put((double) j, mode);
                }
            }
            
            // Fill missing values using the calculated mode for each group
            int imputedCount = 0;
            for (Instance inst : data) {
                // If targetAttr is missing and bestAttr is not missing
                if (inst.isMissing(targetAttr) && !inst.isMissing(bestAttr)) {
                    double groupValue = inst.value(bestAttr);
                    Double modeForGroup = groupModes.get(groupValue);

                    // Nếu có mode cho nhóm này, điền vào
                    if (modeForGroup != null) {
                        inst.setValue(targetAttr, modeForGroup);
                        imputedCount++;
                    }
                }
            }
        }
                
        // Handle any remaining missing values (if any) using global replacement
        // This happens when bestAttr is also missing or a group has no mode
        ReplaceMissingValues filterImpute = new ReplaceMissingValues();
        try {
            filterImpute.setInputFormat(data);
            data = Filter.useFilter(data, filterImpute);
        } catch (Exception e) {
            e.printStackTrace();
        }
        return data;
    }

    private Attribute chiSquareTest(Instances data, Attribute target) {
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
                        // Tính tần số kỳ vọng (Expected Frequency)
                        double expected = (rowTotals[r] * colTotals[c]) / validInstancesCount;

                        // Tránh chia cho 0. Nếu kỳ vọng là 0, đóng góp vào tổng là 0.
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

    private double calculateMode(Instances dataset, Attribute attribute) {
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