import numpy as np
import pandas as pd
import math
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA, FastICA
from sklearn.decomposition import TruncatedSVD 
from sklearn.random_projection import GaussianRandomProjection
from sklearn.random_projection import SparseRandomProjection
from sklearn.feature_extraction import DictVectorizer
from xgboost.sklearn import XGBClassifier

from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import time
start = time.time()


meta = pd.read_csv("./AutoKaggle - Metadata.csv")
arrOfRows = [64,360]
row = 360

def preprocessing(row):
    find_row = meta.loc[row]
    train = ''
    test = None
    check_test = True
    train_X = ''
    train_Y = ''
    test_X = None
    if meta['name'].loc[row] == 'kobe-bryant-shot-selection':
        train = pd.read_csv("./kobe-bryant-shot-selection/data/data.csv")
        check_test = False
    elif meta['name'].loc[row] == 'mercedes-benz-greener-manufacturing':
        train = pd.read_csv("./mercedes-benz-greener-manufacturing/data/train.csv")
        test = pd.read_csv("./mercedes-benz-greener-manufacturing/data/test.csv")
        
    train = train.dropna()
    if check_test:
        test = test.dropna()
    for c in train.columns:

        if train[c].dtype == 'object':    # deal with text
            lbl = LabelEncoder() 
            if check_test:
                lbl.fit(list(train[c].values) + list(test[c].values)) 
                train[c] = lbl.transform(list(train[c].values))
                test[c] = lbl.transform(list(test[c].values))
            else:
                lbl.fit(list(train[c].values))
                train[c] = lbl.transform(list(train[c].values))

    targetName = find_row['targetName']
    train_Y = train[targetName]
    train_X = train.drop(columns=targetName)
    if check_test:
        test_X = test
        return train_X, train_Y, test_X
    else:
        return train_X, train_Y, None





def feature_extraction(row,X_train, X_test):
    if type(meta["function call feature extraction"].loc[row]) is not str:
        print('not func')
        return X_train,X_test
    extraction_function_calls = str(meta["function call feature extraction"].loc[row])
    extraction_function_calls = extraction_function_calls.split(",")
    extraction_funtion_param = eval(meta["function parameters feature extraction"].loc[row])
    function_nums = len(extraction_function_calls)
    for i in range(function_nums):
        str1 = extraction_function_calls[i]
        str2 = extraction_funtion_param[i]
        l_str = str1.split("(")
        l_str.insert(1,"("+str2)
        str_call = ''
        str_call = str_call.join(l_str)
        str_call = 'extractor' + '=' + str_call
        exec(str_call, globals(), globals())
        extracted_train = extractor.fit_transform(X_train)
        n_comp = extracted_train.shape[1]
        for j in range(0, n_comp):
            X_train['extractor'+ str(i)+"_"+str(j)] = extracted_train[:, j]
        if X_test is not None:
            extracted_test = extractor.fit_transform(X_test)
            for j in range(0, n_comp):
                X_test['extractor'+ str(i)+"_"+str(j)] = extracted_test[:, j]
            return X_train,X_test
        else:
            return X_train,None

def feature_selection():
    pass

def estimation(row,X_train,X_test,Y_train):
    estimation_function_calls = meta["function calls estimation"].loc[row]
    estimation_function_calls = estimation_function_calls.split(",")
    print(type(meta["function parameters estimation"].loc[row]))
    print(meta["function parameters estimation"].loc[row])
    estimation_function_param = eval(meta["function parameters estimation"].loc[row])
    
    print(len(estimation_function_calls))
    if len(estimation_function_calls) == 1:
        l_str = estimation_function_calls[0].split("(")
        l_str.insert(1,'('+estimation_function_param[0])
        str_call = ''
        str_call = str_call.join(l_str)
        str_call = 'estimator' + '=' + str_call
        exec(str_call,globals(),globals())
#         estimator.fit(X_train,Y_train)
        print(cross_val_score(estimator, X_train, Y_train, cv=3, n_jobs=8))
    else:
        estimators = []
        n_estimators = len(estimation_function_calls)
        for i in range(n_estimators):
            str1 = extraction_function_calls
            str2 = extraction_funtion_param
            l_str = str1.split("(")
            l_str.insert(1,"("+str2)
            str_call = ''
            str_call = str_call.join(l_str)
            str_call = 'estimator' + '=' + str_call
            print(l_str)
            print(str_call)
            exec(str_call)
            estimators.append(estimator)
            postprocessing(estimators,stack = True)

X_train, Y_train, X_test = preprocessing(row)
X_train_selected, X_test_selected = feature_extraction(row,X_train,X_test)
Y_pred = estimation(row,X_train_selected,X_test,Y_train)

def postprocessing(estimators,stack):
    pass

end = time.time()
print(end - start)
