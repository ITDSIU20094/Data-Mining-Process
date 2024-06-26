package PrintOutModel;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import java.util.Random;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.trees.RandomTree;
import weka.classifiers.functions.SMO;
import weka.classifiers.lazy.IBk;
import java.io.PrintWriter;
import java.io.File;
import weka.classifiers.rules.OneR;
import weka.classifiers.rules.ZeroR;

public class Classification {
    public static void main(String args[]) throws Exception{

        // Load dataset training
        DataSource source = new DataSource("C:\\Users\\quanh\\OneDrive\\Tài liệu\\GitHub\\Data-Mining-Process\\train_test_dataset\\Train_dataset.arff");
        Instances trainData = source.getDataSet();
        // Set class index to the RAIN attribute
        trainData.setClassIndex(trainData.attribute("RAIN").index());
        
        // Load dataset testing
        DataSource source1 = new DataSource("C:\\Users\\quanh\\OneDrive\\Tài liệu\\GitHub\\Data-Mining-Process\\train_test_dataset\\Test_dataset.arff");
        Instances testData = source1.getDataSet();
        // Set class index to the RAIN attribute
        testData.setClassIndex(testData.attribute("RAIN").index());

        // Create a file to write the output
        File file = new File("C:\\Users\\quanh\\OneDrive\\Tài liệu\\GitHub\\Data-Mining-Process\\Output\\Classification.txt");
        PrintWriter output = new PrintWriter(file);

///// Random Tree /////
        // Create and build the RandomTree classifier
        RandomTree tree = new RandomTree();
        tree.buildClassifier(trainData);

        // Evaluate the J48 model
        Evaluation eval = new Evaluation(trainData);
            // Built cross-validation = 10 folds
        Random rand = new Random(1);
        int folds = 10;
            // use 10-fold cross-validation
        eval.crossValidateModel(tree, testData, folds, rand);
        output.println(eval.toSummaryString("\nRandom Tree results:", false)); 
        output.println(eval.toMatrixString("\nRandom Tree Confusion Matrix:"));

///// NaiveBayes /////
        // Create and build the NaiveBayes classifier
        NaiveBayes nb = new NaiveBayes();
        nb.buildClassifier(trainData);

        // Evaluate the NaiveBayes model
        Evaluation eval1 = new Evaluation(trainData);
            // Built cross-validation = 10 folds
        Random rand1 = new Random(1);
        int folds1 = 10;
            // use 10-fold cross-validation
        eval1.crossValidateModel(nb, testData, folds1, rand1);
        output.println(eval1.toSummaryString("\nNaiveBayes results:", false));
        output.println(eval1.toMatrixString("\nNaiveBayes Confusion Matrix:"));

///// SMO /////
        // Create and build the SMO classifier
        SMO smo = new SMO();
        smo.buildClassifier(trainData);

        // Evaluate the SMO model
        Evaluation eval2 = new Evaluation(trainData);
            // Built cross-validation = 10 folds
        Random rand2 = new Random(1);
        int folds2 = 10;
            // use 10-fold cross-validation
        eval2.crossValidateModel(smo, testData, folds2, rand2);
        output.println(eval2.toSummaryString("\nSMO results:", false));
        output.println(eval2.toMatrixString("\nSMO Confusion Matrix:"));

///// IBK /////
        // Create and build the IBK classifier
        IBk ibk = new IBk();
        ibk.buildClassifier(trainData);

        // Evaluate the OneR model
        Evaluation eval3 = new Evaluation(trainData);
            // Built cross-validation = 10 folds
        Random rand3 = new Random(1);
        int folds3 = 10;
            // use 10-fold cross-validation
        eval3.crossValidateModel(ibk, testData, folds3, rand3);
        output.println(eval3.toSummaryString("\nIBK results:", false));
        output.println(eval3.toMatrixString("\nIBK Confusion Matrix:"));

///// OneR /////
        // Create and build the OneR classifier
        OneR oneR = new OneR();
        oneR.buildClassifier(trainData);

        // Evaluate the OneR model
        Evaluation eval4 = new Evaluation(trainData);
            // Built cross-validation = 10 folds
        Random rand4 = new Random(1);
        int folds4 = 10;
            // use 10-fold cross-validation
        eval4.crossValidateModel(oneR, testData, folds4, rand4);
        output.println(eval4.toSummaryString("\nOneR results:", false));
        output.println(eval4.toMatrixString("\nOneR Confusion Matrix:"));

///// ZeroR /////
        // Create and build the ZeroR classifier
        ZeroR zeroR = new ZeroR();
        zeroR.buildClassifier(trainData);

        // Evaluate the ZeroR model
        Evaluation eval5 = new Evaluation(trainData);
            // Built cross-validation = 10 folds
        Random rand5 = new Random(1);
        int folds5 = 10;
            // use 10-fold cross
        eval5.crossValidateModel(zeroR, testData, folds5, rand5);
        output.println(eval5.toSummaryString("\nZeroR results:", false));
        output.println(eval5.toMatrixString("\nZeroR Confusion Matrix:"));

        output.close();
    }    

}
