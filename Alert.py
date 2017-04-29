"""
Alert.py

Author: Utsav Dixit
"""
import pandas as pd
import numpy as np
import sklearn as sk
from sklearn.preprocessing import LabelEncoder
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

output_file_name = 'model_output.csv'

def read_dataset_to_train_and_test(train_csv_path, test_csv_path, output_file_path):
    output_file_path = output_file_path + output_file_name
    train = pd.read_csv(train_csv_path)
    test  = pd.read_csv(test_csv_path)

    ID_col = ['ObsNum']
    target_col = ["IsAlert"]
    num_cols = ['P1','P2','P3','P4','P5','P6','P7','P8','E1','E2','E3','E4','E5','E6','E7','E8','E9','E10','E11','V1','V2','V3','V4','V5','V6','V7','V8','V9','V10','V11','TrialID']
    other_col=['Type']

    # Separating 75% data for training and testing randomly
    train['is_train'] = np.random.uniform(0, 1, len(train)) <= .75
    Train, validate = train[train['is_train']==True], train[train['is_train']==False]
    
    features=list(set(list(num_cols)))
    x_train = Train[list(features)].values
    y_train = Train["IsAlert"].values
    x_validate = validate[list(features)].values
    y_validate = validate["IsAlert"].values
    x_test=test[list(features)].values
    #print("check3")
    
    # Creating and using data structures to store important features.
    # Creating Dictionary
    imp_features = dict()
    # Creating List
    imp_list = list()
    # Creating helper List
    imp_list1 = list()
    
    # Returns object of Random Forest classifier.
    #rf = UseRandomForest()
    rf = RandomForestClassifier(n_estimators=100,oob_score =True)
    
    rf.fit(x_train, y_train)
    for feature,imp in zip(features,rf.feature_importances_):
        imp_features[feature] = imp
        print(feature, imp)
    
    imp_list = sorted(imp_features, key=imp_features.get)

    for item in range(20,30):
        # Printing top 10 features
        print(imp_list[item])
        # Selecting top 10 features
        imp_list1.append(imp_list[item])

    # Using Prediction on Validate dataset
    status = rf.predict(x_validate)
    
    # Printing the score matrix, which prints the accuracy of the model
    print(sk.metrics.classification_report(y_validate,status))
    
    
    # Predicting accuracy on Testing dataset
    final_status = rf.predict(x_test)
    test["IsAlert"]=final_status       
    
    num_cols = imp_list1
    #
    features=list(set(list(num_cols)))
    x_train = train[list(features)].values
    y_train = train["IsAlert"].values
    
    x_validate = validate[list(features)].values
    y_validate = validate["IsAlert"].values
    x_test=test[list(features)].values
    
    rf = RandomForestClassifier(n_estimators=100,max_features=5,oob_score =True)
    rf.fit(x_train, y_train)
    
    status_new = rf.predict(x_validate)
    print(sk.metrics.classification_report(y_validate,status_new))
    
    final_status_new = rf.predict(x_test)
    test["IsAlert"]=final_status_new            
    test.to_csv(output_file_path)


def path_to_file():
    print('Enter CSV files absolute path separatated by space.')
    path1, path2= input().split(" ")
    print('Enter Path for final output file')
    path3 = input()
    return path1, path2, path3

# main function
if __name__=="__main__":
    csv_file_path1, csv_file_path2, output_file_path = path_to_file()
    read_dataset_to_train_and_test(csv_file_path1, csv_file_path2, output_file_path)
    
    
    
