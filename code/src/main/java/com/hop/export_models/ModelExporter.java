package com.hop.export_models;

import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.supervised.instance.Resample;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.util.ArrayList;

import com.hop.utils.Utils;

public class ModelExporter {
    private static final String[] STAGE1_FEATURES = {
            "SEVERITY_INDEX", "AGE_GROUP", "SEX", "PNEUMONIA",
            "INMSUPR", "DIABETES", "HIPERTENSION", "CARDIOVASCULAR"
    };
    private static final String TARGET = "DIED";

    /**
     * Export the results of a single model (Stage 1 only)
     */
    public static void exportSingleModel(String modelPath, String datasetPath, String exportPath) {
        try {
            System.out.println("--- Exporting Single Model: " + modelPath + " ---");
            Classifier classifier = (Classifier) SerializationHelper.read(modelPath);

            // Load dataset w/ fixed seed for consistency
            Instances data = loadDataset(datasetPath);
            data = Utils.filterAttributes(data);

            BufferedWriter writer = new BufferedWriter(new FileWriter(exportPath));
            writer.write("InstanceID,Actual,Predicted,Prob_Cured,Prob_Died,Is_Correct\n");

            for (int i = 0; i < data.numInstances(); i++) {
                Instance instance = data.instance(i);
                double actual = instance.classValue();
                double predicted = classifier.classifyInstance(instance);
                double[] dist = classifier.distributionForInstance(instance);

                // Write results to the file
                writeLine(writer, i, actual, predicted, dist);
            }
            writer.close();
            System.out.println("Done! Saved to: " + exportPath);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /**
     * Export the results of the Cascade Tunnel system
     */
    public static void exportCascadeModel(String stage1Path, String stage2Path, String datasetPath, String exportPath) {
        try {
            System.out.println("--- Exporting Cascade Tunnel System ---");
            // Load both models for the cascade
            Classifier stage1 = (Classifier) SerializationHelper.read(stage1Path);
            Classifier stage2 = (Classifier) SerializationHelper.read(stage2Path);

            // Load data w/ same sampling as Single Model
            Instances data = loadDataset(datasetPath);
            data = filterAttributes(data, STAGE1_FEATURES);

            BufferedWriter writer = new BufferedWriter(new FileWriter(exportPath));
            writer.write("InstanceID,Actual,Predicted,Prob_Cured,Prob_Died,Is_Correct,Stage_Used\n");

            int s1Count = 0;
            int s2Count = 0;

            for (int i = 0; i < data.numInstances(); i++) {
                Instance instance = data.instance(i);
                double actual = instance.classValue();

                // --- LOGIC CASCADE ---
                double predicted;
                double[] dist;
                String stageUsed;

                // Step 1: Ask Stage 1 (General Practitioner)
                double s1Pred = stage1.classifyInstance(instance);

                if (s1Pred == 0.0) {
                    // If Stage 1 says "Cured" (0) -> Accept immediately
                    predicted = 0.0;
                    dist = stage1.distributionForInstance(instance);
                    stageUsed = "Stage1_NB";
                    s1Count++;
                } else {
                    // If Stage 1 says "Died" (1) -> Ask Stage 2 (Specialist)
                    predicted = stage2.classifyInstance(instance);
                    dist = stage2.distributionForInstance(instance);
                    stageUsed = "Stage2_RF";
                    s2Count++;
                }

                // Write results to the file
                String line = String.format("%d,%.0f,%.0f,%.4f,%.4f,%d,%s\n",
                        i + 1,
                        actual,
                        predicted,
                        dist[0], // Prob Cured
                        dist[1], // Prob Died
                        (actual == predicted ? 1 : 0),
                        stageUsed
                );
                writer.write(line);
            }
            writer.close();
            System.out.println("Done! Saved to: " + exportPath);
            System.out.println("Stats: Stage 1 decided: " + s1Count + " cases. Stage 2 decided: " + s2Count + " cases.");

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    // --- HELPER METHODS ---

    private static void writeLine(BufferedWriter writer, int id, double actual, double pred, double[] dist) throws Exception {
        writer.write(String.format("%d,%.0f,%.0f,%.4f,%.4f,%d\n",
                id + 1, actual, pred, dist[0], dist[1], (actual == pred ? 1 : 0)));
    }

    private static Instances loadDataset(String path) throws Exception {
        System.out.println("Loading dataset from: " + path);
        DataSource source = new DataSource(path);
        Instances data = source.getDataSet();

        // Set class index
        if (data.attribute(TARGET) != null) {
            data.setClass(data.attribute(TARGET));
        } else {
            data.setClassIndex(data.numAttributes() - 1);
        }

        // Resampling logic for consistency across models
        int maxSamplesForVis = 100000;
        if (data.numInstances() > maxSamplesForVis) {
            double percent = ((double) maxSamplesForVis / data.numInstances()) * 100;
            System.out.println("Data too large (" + data.numInstances() + "). Resampling to " + String.format("%.2f", percent) + "% (~" + maxSamplesForVis + " rows)...");

            Resample resample = new Resample();
            resample.setBiasToUniformClass(0.0);
            resample.setNoReplacement(true);
            resample.setSampleSizePercent(percent);
            resample.setRandomSeed(42); // FIXED SEED guarantees same sample for same dataset
            resample.setInputFormat(data);
            data = Filter.useFilter(data, resample);
        }

        return data;
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
}