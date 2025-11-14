package com.hop.preprocess;

import com.hop.preprocess.Cleaner;

import weka.core.Instances;

public class Preprocess {
    public static void main(String[] args) {
        Cleaner cleaner = new Cleaner();
        Instances data = cleaner.loadData();
        System.out.println("   - Loaded " + data.numInstances() + " instances and " + data.numAttributes() + " attributes.");
    }
}
