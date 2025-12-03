package com.hop.preprocess;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

import static com.hop.utils.Utils.saveCSV;
import static com.hop.utils.Utils.saveARFF;


public class Preprocess {
    
    public static void main(String[] args) throws Exception {
        preprocessAndExport("covid");
    }

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

    private static Instances loadData(String inputFilePath) {
        try {
            DataSource source = new DataSource(inputFilePath);
            return source.getDataSet();
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }

}