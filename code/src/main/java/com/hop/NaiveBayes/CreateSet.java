package com.hop.NaiveBayes;

import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;
import weka.core.converters.CSVSaver;
import weka.filters.Filter;
import weka.filters.supervised.instance.StratifiedRemoveFolds;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.filters.unsupervised.attribute.Remove;

import java.io.File;
import java.util.Random;

public class CreateSet {

    public static void main(String[] args) throws Exception {
        String inputCsv = args.length > 0 ? args[0] : "data/preprocess/covid_cleaned.csv";
        String outTrainCsvPath = args.length > 1 ? args[1] : "data/naive_bayes/stage1_train.csv";
        String outTestCsvPath = args.length > 2 ? args[2] : "data/naive_bayes/stage1_test.csv";
        String outTrainArffPath = args.length > 3 ? args[3] : "data/naive_bayes/stage1_train.arff";
        String outTestArffPath = args.length > 4 ? args[4] : "data/naive_bayes/stage1_test.arff";
        int seed = 42;

        // Load CSV
        CSVLoader loader = new CSVLoader();
        loader.setSource(new File(inputCsv));
        Instances data = loader.getDataSet();

        // Ensure DIED attribute exists
        int diedIdx = data.attribute("DIED") != null ? data.attribute("DIED").index() : -1;
        if (diedIdx == -1) throw new IllegalArgumentException("DIED attribute not found");

        // If DIED is numeric, convert to nominal for stratification
        if (data.attribute(diedIdx).isNumeric()) {
            NumericToNominal n2n = new NumericToNominal();
            n2n.setAttributeIndices(String.valueOf(diedIdx + 1));
            n2n.setInputFormat(data);
            data = Filter.useFilter(data, n2n);
        }

        // Set class to DIED for stratified split
        data.setClassIndex(data.attribute("DIED").index());

        // Randomize
        data.randomize(new Random(seed));
        if (data.classAttribute().isNominal()) data.stratify(5); // prepare for 5-fold -> 20% test

        // Create test (fold 1) and train (remaining folds)
        StratifiedRemoveFolds srf = new StratifiedRemoveFolds();
        srf.setNumFolds(5);
        srf.setSeed(seed);
        srf.setFold(1);
        srf.setInvertSelection(false); // get test
        srf.setInputFormat(data);
        Instances testPool = Filter.useFilter(data, srf);

        srf = new StratifiedRemoveFolds();
        srf.setNumFolds(5);
        srf.setSeed(seed);
        srf.setFold(1);
        srf.setInvertSelection(true); // get train
        srf.setInputFormat(data);
        Instances trainPool = Filter.useFilter(data, srf);

        // Separate died==1 and cured==0 in training set
        Instances diedInst = new Instances(trainPool, 0);
        Instances curedInst = new Instances(trainPool, 0);
        int classIdx = trainPool.classIndex();

        for (int i = 0; i < trainPool.numInstances(); i++) {
            double v = trainPool.instance(i).value(classIdx);
            // If class is nominal, we compare by nominal index or string "1"
            if (trainPool.classAttribute().isNominal()) {
                String label = trainPool.classAttribute().value((int) v);
                if ("1".equals(label) || "yes".equalsIgnoreCase(label) || "true".equalsIgnoreCase(label)) diedInst.add(trainPool.instance(i));
                else curedInst.add(trainPool.instance(i));
            } else {
                if (v == 1.0) diedInst.add(trainPool.instance(i));
                else curedInst.add(trainPool.instance(i));
            }
        }

        // Sample cured to match died count (undersampling)
        int nDied = diedInst.numInstances();
        Instances curedSampled;
        if (curedInst.numInstances() > nDied) {
            curedInst.randomize(new Random(seed));
            curedSampled = new Instances(curedInst, 0, nDied);
        } else {
            curedSampled = new Instances(curedInst);
        }

        // Combine and shuffle
        Instances stage1Train = new Instances(diedInst);
        for (int i = 0; i < curedSampled.numInstances(); i++) stage1Train.add(curedSampled.instance(i));
        stage1Train.randomize(new Random(seed));

        // Select stage1 columns: SEVERITY_INDEX, AGE_GROUP, SEX, PNEUMONIA, DIED
        String[] cols = new String[] {"SEVERITY_INDEX", "AGE_GROUP", "SEX", "PNEUMONIA", "DIED"};
        stage1Train = keepOnlyAttributes(stage1Train, cols);
        testPool = keepOnlyAttributes(testPool, cols);

        // Save CSVs
        CSVSaver saver = new CSVSaver();
        saver.setInstances(stage1Train);
        saver.setFile(new File(outTrainCsvPath));
        saver.writeBatch();

        saver = new CSVSaver();
        saver.setInstances(testPool);
        saver.setFile(new File(outTestCsvPath));
        saver.writeBatch();

        System.out.println("Created stage1 train and test sets .csv successfully.");

        // Save ARFFs
        ArffSaver arffSaver = new ArffSaver();
        arffSaver.setInstances(data);
        arffSaver.setFile(new File(outTrainArffPath));
        arffSaver.writeBatch();

        arffSaver = new ArffSaver();
        arffSaver.setInstances(testPool);
        arffSaver.setFile(new File(outTestArffPath));
        arffSaver.writeBatch();

        System.out.println("Created stage1 train and test sets .arff successfully.");
    }

    private static Instances keepOnlyAttributes(Instances data, String[] keepNames) throws Exception {
        // Build a boolean mask of attributes to keep
        StringBuilder indicesToKeep = new StringBuilder();
        for (String name : keepNames) {
            if (data.attribute(name) == null) throw new IllegalArgumentException("Attribute not found: " + name);
            indicesToKeep.append(data.attribute(name).index() + 1).append(",");
        }
        // Remove trailing comma
        if (indicesToKeep.length() > 0) indicesToKeep.setLength(indicesToKeep.length() - 1);

        Remove remove = new Remove();
        remove.setAttributeIndices(indicesToKeep.toString());
        remove.setInvertSelection(true); // keep listed attributes
        remove.setInputFormat(data);
        Instances out = Filter.useFilter(data, remove);
        // keep class index if DIED present
        if (out.attribute("DIED") != null) out.setClassIndex(out.attribute("DIED").index());
        return out;
    }
}