package com.hop;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.*;

import static com.hop.CascadeTunnel.CascadeTunnelCrossValidation.crossValidateCascadeTunnel;
import static com.hop.NaiveBayes.Main.CrossValidateNaiveBayes;
import static com.hop.NaiveBayes.Main.CrossValidateNaiveBayesWithUndersampling;
import static com.hop.utils.Utils.*;
import static com.hop.utils.Utils.dataFullBalanceARFF;

public class Main {
    public static void main(String[] args) throws Exception {
        System.out.println("--- Testing Naive Bayes Classifier for Stage 1 ---");
        System.out.println("\nMethod 1: 10-Fold Cross-Validation with Strategic Undersampling");
        CrossValidateNaiveBayesWithUndersampling(dataFullARFF);
        System.out.println("\nMethod 2: 10-Fold Cross-Validation without Undersampling");
        CrossValidateNaiveBayes(dataFullARFF);
        System.out.println("\n--- End of Naive Bayes Testing ---");
        System.out.println("--- Tunnel ---");
        crossValidateCascadeTunnel(dataFullARFF);


    }
}
