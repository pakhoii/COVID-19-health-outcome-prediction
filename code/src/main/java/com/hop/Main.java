package com.hop;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.*;

import static com.hop.utils.Utils.loadData;
import static com.hop.utils.Utils.saveARFF;

public class Main {
    public static void main(String[] args) throws Exception {
        Instances source = loadData("data/preprocess/covid_sample.csv");
        saveARFF(source, "data/preprocess/covid_sample.arff");
    }
}
