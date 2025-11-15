package com.hop.preprocess;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVSaver;
import java.io.File;

public class Preprocess {
    public static void main(String[] args) {
        String inputFilePath = "data/covid.csv";
        String arffOutputFilePath = "data/covid_cleaned.arff";
        String csvOutputFilePath = "data/covid_cleaned.csv";

        Cleaner cleaner = new Cleaner();
        Instances data = loadData(inputFilePath);
        
        if (data != null) {
            Instances cleaned_data = cleaner.preprocess(data);
            System.out.println("Preprocess successfully");
            if (cleaned_data != null) {
                saveData(cleaned_data, arffOutputFilePath, csvOutputFilePath);
                System.out.println("Saved successfully");
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

    private static void saveData(Instances data, String arffPath, String csvPath) {
        try {
            ArffSaver arffSaver = new ArffSaver();
            arffSaver.setInstances(data);
            arffSaver.setFile(new File(arffPath));
            arffSaver.writeBatch();

            CSVSaver csvSaver = new CSVSaver();
            csvSaver.setInstances(data);
            csvSaver.setFile(new File(csvPath));
            csvSaver.writeBatch();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}