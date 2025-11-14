package com.hop.preprocess;

import weka.core.Instances;
import weka.core.Instance;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVSaver;
import weka.core.Attribute;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.instance.RemoveWithValues;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;

import java.io.File;
import java.util.ArrayList;

public class Cleaner {
    private String inputFilePath;
    private String arffOutputFilePath;
    private String csvOutputFilePath;
    private DataSource source;

    public Cleaner() {
        inputFilePath = "data/covid.csv";
        arffOutputFilePath = "data/covid_cleaned.arff";
        csvOutputFilePath = "data/covid_cleaned.arff";
    }

    public Instances loadData() {
        Instances data = null;
        try {
            source = new DataSource(inputFilePath);
            data = source.getDataSet();
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }

        return data;
    }

    public void preprocess(Instances data) {
        // Remove patients who negative with COVID
        RemoveWithValues filterRemoveNonCovid = new RemoveWithValues();
        filterRemoveNonCovid.setAttributeIndex(
            String.valueOf(
                // Need to plus 1 since weka filter index is 1-based
                data.attribute("CLASSIFICATION").index() + 1
            )
        );

        // Filter and remove any row that has classification > 4 (non covid patient)
        filterRemoveNonCovid.setNominalIndices("4-last");

        try {
            filterRemoveNonCovid.setInputFormat(data);
            data = Filter.useFilter(data, filterRemoveNonCovid);
        } catch (Exception e) {
            e.printStackTrace();
            return;
        }

        // Feature Extraction: DIED - extract from DATE_DIED
        ArrayList<String> diedValue = new ArrayList<>();
        diedValue.add("0"); // alive
        diedValue.add("1"); // died

        data.insertAttributeAt(new Attribute("DIED"), data.numAttributes());

        int dateDiedIndex = data.attribute("DATE_DIED").index();
        int diedIndex = data.attribute("DIED").index();

        for (Instance instance : data) {
            String date = instance.stringValue(dateDiedIndex);
            if(date.equals("9999-99-99")) 
                instance.setValue(diedIndex, "0");
            else 
                instance.setValue(diedIndex, "1");
        }

        // Remove unnessary columns
    }
}
