import weka.core.Instances;
import java.util.Random;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.SMO;

public class Step5C_Folds_SMO {
    public static void main(String args[]) throws Exception{
        // Load dataset
        DataSource source = new DataSource("C:\\Users\\quanh\\OneDrive\\Tài liệu\\GitHub\\Data-Mining-Process\\Dataset_ARFF\\wind_data.arff");
        Instances dataset = source.getDataSet();
        // Set class index to the last attribute
        dataset.setClassIndex(dataset.attribute("RAIN").index());
        
        // Create and build the classifier
        SMO smo = new SMO();
        smo.buildClassifier(dataset);
        int seed = 1; // the seed for randomizing the data
        int folds = 10; // the number of folds for cross-validation

        // Randomize the data
        Random rand = new Random(seed);
        Instances randData = new Instances(dataset);
        randData.randomize(rand);
        //stratify the data
        if (randData.classAttribute().isNominal())
            randData.stratify(folds);

        System.out.println("=== SMO ===");
        // Perform cross-validation
        for (int n = 0; n < folds; n++) {
            Evaluation eval = new Evaluation(randData);
            // generate the training and test set
            Instances train = randData.trainCV(folds, n);
            Instances test = randData.testCV(folds, n);
            // build and evaluate the classifier
            smo.buildClassifier(train);
            eval.evaluateModel(smo, test);

            // output evaluation
            System.out.println();
            System.out.println(eval.toSummaryString("=== " + folds + "-fold Cross-validation run " + (n+1) + " ===", false));
            System.out.println(eval.toMatrixString("=== Confusion matrix for fold " + (n+1) + "/" + folds + " ===\n"));
        }
    }
}
