import numpy as np
import pandas as pd
import math
import string
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA, FastICA
from sklearn.decomposition import TruncatedSVD 
from sklearn.random_projection import GaussianRandomProjection
from sklearn.random_projection import SparseRandomProjection
from sklearn.feature_extraction import DictVectorizer
from xgboost.sklearn import XGBClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import *
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import time

start = time.time()

meta = pd.read_csv("./AutoKaggle - Metadata.csv", encoding='latin-1', error_bad_lines=False)
arrOfRows = [65,360,239]
nlp_rows = [239]
tabular_rows = [65,360,241]

row = 65
train = ''
data_name = meta['name'].loc[row]


def preprocessing(row):
    find_row = meta.loc[row]
    train = ''
    test = None
    check_pred = True # if true, there exists a test dataset to submit
    train_X = ''
    train_Y = ''
    pred_X = None
    print(meta['name'].loc[row])
    
    if meta['name'].loc[row] == 'kobe-bryant-shot-selection':
        train = pd.read_csv("./kobe-bryant-shot-selection/data/data.csv")
        check_pred = False
    elif meta['name'].loc[row] == 'mercedes-benz-greener-manufacturing':
        train = pd.read_csv("./mercedes-benz-greener-manufacturing/data/train.csv")
        pred = pd.read_csv("./mercedes-benz-greener-manufacturing/data/test.csv")
    elif meta['name'].loc[row] == 'uciml_sms-spam-collection-dataset':
        train = pd.read_csv("./uciml_sms-spam-collection-dataset/data/spam.csv",  encoding='cp1252', error_bad_lines=False)
        check_pred = False
    else:
        train = pd.read_csv("./nsharanh-1b-visa/h1b_kaggle.csv") 
        check_pred = False
    
    if check_pred:
        pred = pred.dropna()
    if row in nlp_rows:
        row = pd.read_csv("./uciml_sms-spam-collection-dataset/submission/row.csv")
        sms = train
        row_prepro = row['preprocessing function call'][0]
        prepro_ls = eval(row_prepro)
        sms = eval(prepro_ls[0])
        train = eval(prepro_ls[1])
        return train
    else:
        train = train.dropna()
        
        if type(meta["unwanted column"].loc[row]) is str:  # check if there's unwanted column
            column_list = eval(meta["unwanted column"].loc[row])
            train.drop(column_list,axis=1)
        
        if type(meta["numeric column"].loc[row]) is str:
            numeric=eval(meta["numeric column"].loc[row])
        
        for c in train.columns:
            if train[c].dtype == 'object':  #deal with text
                lbl = LabelEncoder() 
                if check_pred:
                    lbl.fit(list(train[c].values) + list(test[c].values)) 
                    train[c] = lbl.transform(list(train[c].values))
                    test[c] = lbl.transform(list(test[c].values))
                else:
                    lbl.fit(list(train[c].values))
                    train[c] = lbl.transform(list(train[c].values))
        targetName = find_row['targetName']
        train_Y = train[targetName]
        train_X = train.drop(columns=targetName)
        
        if type(meta["preprocessing function call"].loc[row]) is not str: #check if there is preprocessing functions
            print('No preprocessing')
        
        else:
            preprocessing_func = eval(meta["preprocessing function call"].loc[row])
            for call in preprocessing_func:
                print(call)
                exec(call)
        
        train_X,test_X,train_Y,test_Y = train_test_split(train_X, train_Y, test_size=0.2)

        if check_pred:
            pred_X = pred
            return train_X, train_Y,test_X,test_Y ,pred_X
        else:
            return train_X, train_Y, test_X,test_Y,None



def text_process(text):
    
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = [word for word in text.split() if word.lower() not in stopwords.words('english')]
    
    return " ".join(text)

def feature_extraction(row, X_train,X_test,X_pred):
    if row in nlp_rows:
        rowcsv = pd.read_csv("./uciml_sms-spam-collection-dataset/submission/row.csv")
        row_extract = rowcsv['featureExtractor function call'].loc[0]
        sms = X_train
        extract = eval(row_extract)
        sms['message'] = eval(extract[0])
        sms['message'] = eval(extract[1])
        text_feat = sms['message'].apply(str).copy()
        text_feat = eval(extract[2])
        vectorizer = eval(extract[3])
        features = eval(extract[4])
        features_train, features_test, labels_train, labels_test = train_test_split(features, sms['label'], test_size=0.3)
        return features_train, features_test, labels_train, labels_test
    else:
        if type(meta["featureExtractor function call"].loc[row]) is not str:
            print('not func')
            return X_train,X_test,X_pred
        
        extraction_function_calls = str(meta["featureExtractor function call"].loc[row])
        extraction_function_calls = extraction_function_calls.split(",")
        extraction_funtion_param = eval(meta["featureExtractor function call"].loc[row])
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
            extracted_test = extractor.fit_transform(X_test)
            n_comp = extracted_train.shape[1]
            for j in range(0, n_comp):
                X_train['extractor'+ str(i)+"_"+str(j)] = extracted_train[:, j]
                X_test['extractor'+ str(i)+"_"+str(j)] = extracted_test[:, j]
            if X_pred is not None:
                extracted_pred = extractor.fit_transform(X_pred)
                for j in range(0, n_comp):
                    X_test['extractor'+ str(i)+"_"+str(j)] = extracted_pred[:, j]
                return X_train,X_test,X_pred
            else:
                return X_train,X_test,None

def feature_selection():
    pass

def estimation(row,X_train,X_test,Y_train, Y_test):
    if row in nlp_rows:
        rowcsv = pd.read_csv("./uciml_sms-spam-collection-dataset/submission/row.csv")
        row_extract = eval(rowcsv['estimator1 function call'].loc[0])
        mnb = eval(row_extract[0])
        eval(row_extract[1])
        pred = eval(row_extract[2])
        if rowcsv['performanceMetric'].loc[0] == 'accuracy':
            return accuracy_score(Y_test, pred)
    else:
        estimation_function_calls = eval(meta["estimator1 function call"].loc[row])

        
        print(len(estimation_function_calls))
        if len(estimation_function_calls) == 1:
            str_call = estimation_function_calls[0]
            print(str_call)
            str_call = 'estimator' + '=' + str_call
            print(str_call)
            exec(str_call,globals(),globals())
            
            if meta["taskType"].loc[row] == 'classification':
                estimator.fit(X_train,Y_train)
                Y_pred = estimator.predict(X_test)
#                 print('here')
                print("Accuracy:", recall_score(Y_test,Y_pred,average='weighted'))
#                 print('here')
            elif meta["taskType"].loc[row] == 'regression':
                estimator.fit(X_train,Y_train)
                print("Accuracy:", r2_score(Y_test,Y_pred,average='weighted'))
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
#                 print(l_str)
#                 print(str_call)
                exec(str_call)
                estimators.append(estimator)
                postprocessing(estimators,stack = True)

if row in nlp_rows:
    train_set = preprocessing(row)
    X_train, X_test, Y_train, Y_test = feature_extraction(row, train_set, None, None)
    Y_pred = estimation(row, X_train, X_test, Y_train, Y_test)
    print("The accuracy of SMS Spam Collection Dataset is", Y_pred)
if row in tabular_rows:
    X_train,Y_train,X_test,Y_test,X_pred = preprocessing(row)
    X_train,X_test,X_pred = feature_extraction(row,X_train,X_test,X_pred)
    estimation(row,X_train,X_test,Y_train, Y_test)
    
        
    
    

def postprocessing(estimators,stack):
    pass

end = time.time()
print(end - start)



