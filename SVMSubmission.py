# -*- coding: utf-8 -*-

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from json import JSONDecoder, JSONDecodeError  # for reading the JSON data files
import re  # for regular expressions
import os  # for os related operations
from sklearn import svm, preprocessing
#from sklearn.cross_validation import StratifiedShuffleSplit
from matplotlib import style
style.use("ggplot")
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm as cm
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from mpl_toolkits.mplot3d import Axes3D
from tabulate import tabulate
from sklearn.model_selection import cross_val_score
from statistics import mean
from sklearn.metrics import confusion_matrix 
warnings.filterwarnings('ignore')
"""
Created on Tue Jul  9 11:34:24 2019

@author: mschaak
"""
def buildSets(features = ['TOTUSJH', "TOTPOT", "TOTUSJZ", "ABSNJZH", "USFLUX"]):
    train, test = preProc( 'fold1Training.json_avg_med.csv',"fold2Training.json_avg_med.csv",'fold3Training.json_avg_med.csv', features)
    X = np.array(train[features].values.tolist())
    Y = np.array(train["LABEL"].values.tolist())
   
    X_test = np.array(test[features].values.tolist())
    y_test = np.array(test["LABEL"].values.tolist())
    return X,Y, X_test, y_test

def preProc(file1, file2, file3, feature = ['TOTUSJH', "TOTPOT", "TOTUSJZ", "ABSNJZH", "USFLUX"]):
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    df3 = pd.read_csv(file3)
    features = ['TOTUSJH', "TOTPOT", "TOTUSJZ", "ABSNJZH", "USFLUX","LABEL"]
    df1 = df1[features]
    #df1 = df1.dropna()
    df2 = df2[features]
    #df2 = df2.dropna()
    df3 = df3[features]
    
    frames = [df1,df2]

    df_comp = pd.concat(frames)
    frames = [df_comp, df3]
    df_keys = pd.concat(frames, keys = ['x','y'])
    print("weeeee")
    #df_keys = df_keys[(df_keys[feature] != 0).all(1)]
    labels = df_keys['LABEL']
    df_keys.fillna(df3.mean(), inplace=True)
    df_keys = df_keys.drop(['LABEL'], axis = 1)
    print("test")
    tot_avg = df_keys.mean(axis = 0)
    tot_std = df_keys.std(axis = 0)
    tot_med = df_keys.median(axis = 0)
    tot_med_list = tot_med.tolist()
    tot_avg_list = tot_avg.tolist()
    tot_std_list = tot_std.tolist()
    counter = 0
    for feat in feature:
        df_keys[feat] = df_keys[feat]-tot_avg_list[counter]
        df_keys[feat] =df_keys[feat]/tot_std_list[counter]
        counter+=1
    df_keys["LABEL"] = labels
    return df_keys.loc['x'], df_keys.loc['y']
    pass


def Analysis():


    X,y, X_test = buildSets()

   

    clf= svm.SVC(kernel= "rbf", C = 4.0, gamma = .075)

    clf.fit(X,y) 

    correct_count = 0

    '''for x in range(1, test_size+1):

        if clf.predict([X[-x]])[0] == y[-x]:

            correct_count += 1'''

    pred_labels = clf.predict(X_test)
    
    pred_labels = pred_labels.astype('int32')

    print(len(pred_labels))
    
    temp = []
    for index in range(1, len(pred_labels) + 1):
        temp.append(index)
        
    df = pd.DataFrame({'Id': temp, 'ClassLabel': pred_labels})
    df.to_csv('SubmissionPrototype.csv', index = False)

    print(len(y_test))

    for x in range (len(pred_labels)):

        if pred_labels[-x] == y_test[-x]:

            correct_count+=1

    #TP, FP, TN, FN = perf_measure(y_test, pred_labels)

    TN, FP, FN, TP = confusion_matrix(y_test, pred_labels).ravel()

    print("TP: ", TP, "FP: ", FP, "TN: ", TN,"FN: ", FN)

    f1 = f1_score(y_test, pred_labels, average="macro")

    pre = precision_score(y_test, pred_labels, average="macro")

    recall = recall_score(y_test, pred_labels, average=None)

    N = TN + FP

    P = TP + FN

    HSS1 = [(TN + TP - P) / N, (TP + TN - N) / P]

    HSS2 = [(2 * ((TN * TP) - (FP * FN)) / (N * (FP + TP) + (TN + FN) * P)), (2 * ((TP * TN) - (FN * FP)) / (P * (FN + TN) + (TP + FP) * N))]

    CH = [((TN + FN) * (TN + FP)) / (N + P), ((TP + FP) * (TP + FN)) / (P + N)]

    GS = [(TN - CH[0]) / (TN + FN + FP - CH[0]), (TP - CH[1]) / (TP + FP + FN - CH[1])]

    TSS = [recall[0] - (FN / (FN + TP)), recall[1] - (FP / (FP + TN))]

    

    print("acc:" ,(correct_count/len(y_test))*100.00)

    print("f1: ",f1)

    print("pre: ", pre)

    print("recall: ", mean(recall))

    print("HSS1: ", mean(HSS1))

    print("HSS2: ", mean(HSS2))

    print("GS: ", mean(GS))

    print("TSS: ", mean(TSS))

    

    pass

def PCA():
    
    pass
