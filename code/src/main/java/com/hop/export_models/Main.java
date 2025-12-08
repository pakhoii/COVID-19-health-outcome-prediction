package com.hop.export_models;

import static com.hop.utils.Utils.*;
import static com.hop.export_models.ModelExporter.*;

public class Main {
    public static void main(String[] args) {
        // NaiveBayes with no undersampling
        exportSingleModel(saveModelStage1, dataFullARFF, exportedNaiveBayesNoUndersampling);

        // NaiveBayes with undersampling
        exportSingleModel(saveModelStage1UnderSampling, dataFullARFF, exportedNaiveBayesWithUndersampling);

        // Cascade Tunnel
        exportCascadeModel(saveModelStage21Undersampling,
                saveModelStage22Undersampling,
                dataFullARFF, exportedCascadeTunnel);
    }
}
