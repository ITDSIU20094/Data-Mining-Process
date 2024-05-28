import os
import matplotlib.pyplot as plt
#  get a list of file path
file_paths = [
    'C:/Users/PC 2024/OneDrive - VietNam National University - HCM INTERNATIONAL UNIVERSITY/Desktop/DSA project/Data-Mining-Process/Output/Folds_IBK.txt',
    'C:/Users/PC 2024/OneDrive - VietNam National University - HCM INTERNATIONAL UNIVERSITY/Desktop/DSA project/Data-Mining-Process/Output/Folds_NaiveBayes.txt',
    'C:/Users/PC 2024/OneDrive - VietNam National University - HCM INTERNATIONAL UNIVERSITY/Desktop/DSA project/Data-Mining-Process/Output/Folds_OneR.txt',
    'C:/Users/PC 2024/OneDrive - VietNam National University - HCM INTERNATIONAL UNIVERSITY/Desktop/DSA project/Data-Mining-Process/Output/Folds_RandomTree.txt',
    'C:/Users/PC 2024/OneDrive - VietNam National University - HCM INTERNATIONAL UNIVERSITY/Desktop/DSA project/Data-Mining-Process/Output/Folds_SMO.txt',
    'C:/Users/PC 2024/OneDrive - VietNam National University - HCM INTERNATIONAL UNIVERSITY/Desktop/DSA project/Data-Mining-Process/Output/Folds_ZeroR.txt'
]

def taking_accuracy_from_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    classification_accuracy_list = []
    for line in lines:
        if "Correctly Classified Instances" in line:
            classification_accuracy_list.append(float(line.split()[4]))
        

    return classification_accuracy_list
def taking_MAE_from_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    classification_MAE_list = []
    for line in lines:
        if "Mean absolute error" in line:
            classification_MAE_list.append(float(line.split()[3]))
        

    return classification_MAE_list
def taking_RMSE_from_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    classification_RMSE_list = []
    for line in lines:
        if "Root mean squared error" in line:
            classification_RMSE_list.append(float(line.split()[4]))
        

    return classification_RMSE_list

accuracy = []
rmse=[]
mae=[]
for i in range(len(file_paths)):
    accuracy.append(taking_accuracy_from_file(file_paths[i]))
    rmse.append(taking_RMSE_from_file(file_paths[i]))
    mae.append(taking_MAE_from_file(file_paths[i]))

def visualize(data,type):   
# Plotting the data
   plt.figure(figsize=(14, 8))
# Plot each list of accuracies
   list_model=['IBK','NaiveBayes','OneR','RandomTree','SMO','ZeroR']
   for i, acc_list in enumerate(data):
      plt.plot(acc_list, label=f'model {list_model[i]}', marker='o')

   plt.xlabel('k-folds')
   plt.ylabel(type)
   plt.title(type+'Evaluation')
   plt.legend()
   plt.grid(True)
   plt.show()
print(visualize(accuracy,'Accuracy ')) 
print(visualize(mae,'Mean absolute error ')) 
print(visualize(rmse,'Root mean square error ')) 


