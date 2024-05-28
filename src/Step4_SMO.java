import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.functions.SMO;
import weka.classifiers.Evaluation;
import java.util.Random;

public class Step4_SMO {
    public static void main(String args[]) throws Exception{
        // Load training dataset
        DataSource source = new DataSource("C:\\Users\\quanh\\OneDrive\\Tài liệu\\GitHub\\Data-Mining-Process\\train_test_dataset\\Train_dataset.arff");
        Instances trainData = source.getDataSet();
        // Set class index to the last attribute
        trainData.setClassIndex(trainData.attribute("RAIN").index());

        // Load testing dataset
        DataSource source1 = new DataSource("C:\\Users\\quanh\\OneDrive\\Tài liệu\\GitHub\\Data-Mining-Process\\train_test_dataset\\Test_dataset.arff");
        Instances testData = source1.getDataSet();
        // Set class index to the last attribute
        testData.setClassIndex(testData.attribute("RAIN").index());
        
        // Create and build the classifier
        SMO smo = new SMO();
        smo.buildClassifier(trainData);
        System.out.println(smo.toString());

        // Evaluate the SMO model
        Evaluation eval = new Evaluation(trainData);
        // Built cross-validation = 10 folds
        int folds = 10;
        // use 10-fold cross-validation
        eval.crossValidateModel(smo, testData, folds, new Random(1));
        System.out.println(eval.toSummaryString("\nEvaluation results:", false));
        System.out.println("Detailed Accuracy for each class: \n" + eval.toClassDetailsString());
        System.out.println(eval.toMatrixString("\nConfusion Matrix:"));
    }

}
