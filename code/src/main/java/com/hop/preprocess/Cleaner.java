package com.hop.preprocess;

import weka.core.*;

import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.instance.RemoveWithValues;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;

import static com.hop.preprocess.Utils.chiSquareTest;
import static com.hop.preprocess.Utils.calculateMode;
import static com.hop.preprocess.Utils.numericToNominal;

import java.util.*;

public class Cleaner {
    public Cleaner() {}

    protected Instances preprocess(Instances data, String name) {
        // Dealing with BOM csv file, remove the prefix \uFEFF
        for (int i = 0; i < data.numAttributes(); i++) {
            Attribute attribute = data.attribute(i);
            String originalName = attribute.name();

            if (originalName.startsWith("\uFEFF")) {
                String newName = originalName.substring(1);
                data.renameAttribute(attribute, newName);
            }
        }

        return switch (name.toLowerCase()) {
            case "covid" -> preprocessCovid(data);
            case "comorbidity" -> preprocessComorbidity(data);
            case "symptoms" -> preprocessSymptoms(data);
            default -> {
                System.err.println("Unknown dataset name: " + name);
                yield null;
            }
        };
    }

    private Instances preprocessCovid(Instances data) {
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
            if (date.equals("9999-99-99"))
                instance.setValue(diedIndex, "0");
            else
                instance.setValue(diedIndex, "1");
        }

        // Remove unnecessary columns (CLASSIFICATION_FINAL and DATE_DIED)
        Remove filterRemoveColumns = new Remove();

        // Instances in weka has based-0 index
        // But filter in weka use based-1 index -> need to plus 1
        int classificationIndex = data.attribute("CLASIFFICATION_FINAL").index() + 1;
        dateDiedIndex = data.attribute("DATE_DIED").index() + 1;

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
                Attribute attribute = data.attribute(i);

                if (attribute.isNumeric()) {
                    double currentValue = instance.value(i);

                    // Since the value is (1, 2, 97, 99) -> Weka will treat this as numeric
                    if (currentValue == 97.0 || currentValue == 99.0 || currentValue == 98.0)
                        instance.setMissing(i);
                        // Change (1, 2) binary set to (0, 1)
                    else if (currentValue == 2.0)
                        instance.setValue(i, 0);
                }
            }

            // Set PREGNANT to 0 when SEX is 0 (male)
            if (instance.value(sexIndex) == 0.0 && instance.isMissing(pregnantIndex))
                instance.setValue(pregnantIndex, 0);
        }

        // Set AGE_GROUP AGE_GROUP nominal attribute based on AGE
        /* Revised AGE_GROUP:
        - 0-17 (Pediatric)
        - 18-39 (Young Adult - Low Risk)
        - 40-49 (Adult - Moderate Risk)
        - 50-59 (Late Adult - High Risk)
        - 60-69 (Senior - Very High Risk)
        - 70+ (Elderly - Critical Risk)
         */
        List<String> ageGroups = new ArrayList<>();
        ageGroups.add("0-17");
        ageGroups.add("18-39");
        ageGroups.add("40-49");
        ageGroups.add("50-59");
        ageGroups.add("60-69");
        ageGroups.add("70+");

        Attribute ageGroupAttr = new Attribute("AGE_GROUP", ageGroups);

        data.insertAttributeAt(ageGroupAttr, data.numAttributes());

        int ageGroupIndex = data.attribute("AGE_GROUP").index();
        int ageIndex = data.attribute("AGE").index();

        for (Instance inst : data) {
            if (inst.isMissing(ageIndex)) {
                inst.setMissing(ageGroupIndex);
                continue;
            }
            double age = inst.value(ageIndex);

            if (age <= 17) inst.setValue(ageGroupIndex, ageGroups.get(0));
            else if (age <= 39) inst.setValue(ageGroupIndex, ageGroups.get(1));
            else if (age <= 49) inst.setValue(ageGroupIndex, ageGroups.get(2));
            else if (age <= 59) inst.setValue(ageGroupIndex, ageGroups.get(3));
            else if (age <= 69) inst.setValue(ageGroupIndex, ageGroups.get(4));
            else inst.setValue(ageGroupIndex, ageGroups.get(5));
        }

        data = numericToNominal(data);

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
            Map<Double, Double> groupModes = new HashMap<>();

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

            for (Instance inst : data) {
                // If targetAttr is missing and bestAttr is not missing
                if (inst.isMissing(targetAttr) && !inst.isMissing(bestAttr)) {
                    double groupValue = inst.value(bestAttr);
                    Double modeForGroup = groupModes.get(groupValue);

                    if (modeForGroup != null) {
                        inst.setValue(targetAttr, modeForGroup);
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

    private Instances preprocessComorbidity(Instances data) {
        int sexIndex = data.attribute("sex").index();

        // Before: 1 for female and 2 for male
        // After: 0 for male and 1 for female
        for (Instance instance : data) {
            if (instance.value(sexIndex) == 2.0)
                instance.setValue(sexIndex, 0);
        }

        // Change data type to nominal
        data = numericToNominal(data);

        return data;
    }

    private Instances preprocessSymptoms(Instances data) {
        int genderIndex = data.attribute("gender").index();

        // Before: 1 for female and 2 for male
        // After: 0 for male and 1 for female
        for (Instance instance : data) {
            if (instance.value(genderIndex) == 2.0)
                instance.setValue(genderIndex, 0);
        }

        data.renameAttribute(genderIndex, "sex");
        data = numericToNominal(data);

        return data;
    }
}