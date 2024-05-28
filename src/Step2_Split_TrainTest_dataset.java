import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import java.util.Random;
import weka.core.converters.ArffSaver;
import java.io.File;

public class Step2_Split_TrainTest_dataset {
    public static void main(String args[]) throws Exception{
        // Load dataset
        DataSource source = new DataSource("C:\\Users\\quanh\\OneDrive\\Tài liệu\\GitHub\\Data-Mining-Process\\Dataset_ARFF\\wind_data.arff");
        Instances dataset = source.getDataSet(); 
        // Set class index to the RAIN attribute
        dataset.setClassIndex(dataset.attribute("RAIN").index());

        // Randomize dataset
        Random rand = new Random(1);
        dataset.randomize(rand);
        
        // Split dataset into training and testing dataset
        int trainSize = (int) Math.round(dataset.numInstances() * 0.8); // 80% for training
        int testSize = dataset.numInstances() - trainSize;
        Instances trainDataset = new Instances(dataset, 0, trainSize);
        Instances testDataset = new Instances(dataset, trainSize, testSize);

        // Save training dataset
        ArffSaver saver = new ArffSaver();
        saver.setInstances(trainDataset);
        saver.setFile(new File("C:\\Users\\quanh\\OneDrive\\Tài liệu\\GitHub\\Data-Mining-Process\\train_test_dataset\\Train_dataset.arff"));
        saver.writeBatch();

        // Save testing dataset
        saver.setInstances(testDataset);
        saver.setFile(new File("C:\\Users\\quanh\\OneDrive\\Tài liệu\\GitHub\\Data-Mining-Process\\train_test_dataset\\Test_dataset.arff"));
        saver.writeBatch();

        System.out.println("Training dataset size: " + trainDataset.numInstances());
        System.out.println("Testing dataset size: " + testDataset.numInstances());
    }
}
