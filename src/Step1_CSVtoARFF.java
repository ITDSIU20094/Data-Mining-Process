import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.Filter;

import java.io.File;

public class Step1_CSVtoARFF {
    public static void main(String[] args) throws Exception {
        
        //Load CSV
        CSVLoader loader = new CSVLoader();
        loader.setSource(new File("C:\\Users\\quanh\\OneDrive\\Tài liệu\\GitHub\\Data-Mining-Process\\Dataset_CSV\\wind_dataset.csv"));
        Instances data = loader.getDataSet();

        //set up options to remove "DATE" attribute
        String[] opts = new String[]{"-R", "1"};
        //create Remove object(filter class)
        Remove remove = new Remove();
        remove.setOptions(opts);
        remove.setInputFormat(data);
        Instances newData = Filter.useFilter(data, remove);

        // print out the result
        System.out.println(newData.toSummaryString());

        //save ARFF
        ArffSaver saver = new ArffSaver();
        saver.setInstances(newData);
        //and save as ARFF
        saver.setFile(new File("C:\\Users\\quanh\\OneDrive\\Tài liệu\\GitHub\\Data-Mining-Process\\Dataset_ARFF\\wind_data.arff"));
        saver.writeBatch();
    }
}
