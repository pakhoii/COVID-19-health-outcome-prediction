package com.hop.preprocess;

import weka.core.Instances;

import static com.hop.utils.Utils.*;


public class Preprocess {

    public static void main(String[] args) throws Exception {
        preprocessAndExport("covid");
    }

    /**
     * Preprocess the dataset and export to ARFF and CSV files.
     * @param name The base name of the dataset (without extension)
     * @throws Exception if an error occurs during preprocessing or saving
     */
    private static void preprocessAndExport(String name) throws Exception {
        String inputFilePath = "data/raw/" + name + ".csv";
        String arffOutputFilePath = "data/preprocess/" + name + "_cleaned.arff";
        String csvOutputFilePath = "data/preprocess/" + name + "_cleaned.csv";

        Cleaner cleaner = new Cleaner();
        Instances data = loadData(inputFilePath);

        if (data != null) {
            Instances cleaned_data = cleaner.preprocess(data, name);
            System.out.println("Preprocess " + name + ".csv successfully");
            if (cleaned_data != null) {
                saveARFF(cleaned_data, arffOutputFilePath);
                saveCSV(cleaned_data, csvOutputFilePath);
                System.out.println("Saved " + name + " files successfully");
            }

        }

    }

}