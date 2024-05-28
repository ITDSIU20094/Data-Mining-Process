package PrintOutModel;
import weka.core.Instances;
import java.util.Random;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.Evaluation;
import java.io.PrintWriter;
import java.io.File;
import weka.classifiers.rules.ZeroR;

public class Folds_ZeroR {
    public static void main(String args[]) throws Exception{
        // Load dataset
        DataSource source = new DataSource("C:\\Users\\quanh\\OneDrive\\Tài liệu\\GitHub\\Data-Mining-Process\\Dataset_ARFF\\wind_data.arff");
        Instances dataset = source.getDataSet();
        // Set class index to the last attribute
        dataset.setClassIndex(dataset.attribute("RAIN").index());
        
        // Create and build the classifier
        ZeroR zeroR = new ZeroR();
        zeroR.buildClassifier(dataset);
        int seed = 1; // the seed for randomizing the data
        int folds = 20; // the number of folds for cross-validation

        // Randomize the data
        Random rand = new Random(seed);
        Instances randData = new Instances(dataset);
        randData.randomize(rand);
        //stratify the data
        if (randData.classAttribute().isNominal())
            randData.stratify(folds);
        
        // Create a file to write the output
        File file = new File("C:\\Users\\quanh\\OneDrive\\Tài liệu\\GitHub\\Data-Mining-Process\\Output\\Folds_ZeroR.txt");
        PrintWriter output = new PrintWriter(file);

        output.println("=== ZeroR ===");
        // Perform cross-validation
        for (int n = 0; n < folds; n++) {
            Evaluation eval = new Evaluation(randData);
            // generate the training and test set
            Instances train = randData.trainCV(folds, n);
            Instances test = randData.testCV(folds, n);
            // build and evaluate the classifier
            zeroR.buildClassifier(train);
            eval.evaluateModel(zeroR, test);

            // output evaluation
            output.println();
            output.println(eval.toSummaryString("=== " + folds + "-fold Cross-validation run " + (n+1) + " ===", false));
            output.println(eval.toMatrixString("=== Confusion matrix for fold " + (n+1) + "/" + folds + " ===\n"));
        }
        output.close();
    }

}
