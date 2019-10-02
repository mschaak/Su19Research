# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 14:13:42 2019

@author: mschaak
"""
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
from tabulate import tabulate

from sklearn.metrics import confusion_matrix

from statistics import mean 
warnings.filterwarnings('ignore')
def buildCrossVal(features = ['TOTUSJH', "TOTPOT", "TOTUSJZ", "ABSNJZH"]):
    set1 , set2, set3 = preProc("fold1Training.json_avg_med.csv", "fold2Training.json_avg_med.csv", "fold3Training.json_avg_med.csv", features)
    sets = pd.concat([set1,set2])
    samples1 = np.array(sets[features].values.tolist())
    sample1label = np.array(sets["LABEL"].values.tolist())
    target1 = np.array(set3[features].values.tolist())
    target1label= np.array(set3["LABEL"].values.tolist())
    sets = pd.concat([set1,set3])
    samples2 = np.array(sets[features].values.tolist())
    sample2label = np.array(sets["LABEL"].values.tolist())
    target2 = np.array(set2[features].values.tolist())
    target2label= np.array(set2["LABEL"].values.tolist())
    sets = pd.concat([set2,set3])
    samples3 = np.array(sets[features].values.tolist())
    sample3label = np.array(sets["LABEL"].values.tolist())
    target3 = np.array(set3[features].values.tolist())
    target3label= np.array(set3["LABEL"].values.tolist())
    return samples1, sample1label, target1, target1label, samples2, sample2label,target2,target2label, samples3, sample3label,target3, target3label 
def preProc(file1, file2, file3, feature = ['TOTUSJH', "TOTPOT", "TOTUSJZ", "ABSNJZH"]):
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    df3 = pd.read_csv(file3)
    features = ['TOTUSJH', "TOTPOT", "TOTUSJZ", "ABSNJZH","LABEL"]
    df1 = df1[features]
    #df1 = df1.dropna()
    df2 = df2[features]
    #df2 = df2.dropna()
    df3 = df3[features]
    
    frames = [df1,df2,df3]

    df_keys = pd.concat(frames, keys = ['x','y','z'])
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
    return df_keys.loc['x'], df_keys.loc['y'], df_keys.loc['z']
    pass

def crossVal():
    samples1, sample1label, target1, target1label, samples2, sample2label,target2,target2label, samples3, sample3label,target3, target3label = buildCrossVal()
    clf= svm.SVC(kernel= "rbf", C = 15.0, gamma = .75)
    clf.fit(samples1, sample1label)
    pred_label = clf.predict(target1)
    print("fold 1 and 2 train test on 3")
    TN, FP, FN, TP = confusion_matrix(target1label, pred_label).ravel()

    print("TP: ", TP, "FP: ", FP, "TN: ", TN,"FN: ", FN)

    f1 = f1_score(target1label, pred_label, average="macro")

    pre = precision_score(target1label, pred_label, average="macro")

    recall = recall_score(target1label, pred_label, average=None)

    N = TN + FP

    P = TP + FN

    HSS1 = [(TN + TP - P) / N, (TP + TN - N) / P]

    HSS2 = [(2 * ((TN * TP) - (FP * FN)) / (N * (FP + TP) + (TN + FN) * P)), (2 * ((TP * TN) - (FN * FP)) / (P * (FN + TN) + (TP + FP) * N))]

    CH = [((TN + FN) * (TN + FP)) / (N + P), ((TP + FP) * (TP + FN)) / (P + N)]

    GS = [(TN - CH[0]) / (TN + FN + FP - CH[0]), (TP - CH[1]) / (TP + FP + FN - CH[1])]

    TSS = [recall[0] - (FN / (FN + TP)), recall[1] - (FP / (FP + TN))]

    


    print("f1: ",f1)

    print("pre: ", pre)

    print("recall: ", mean(recall))

    print("HSS1: ", mean(HSS1))

    print("HSS2: ", mean(HSS2))

    print("GS: ", mean(GS))

    print("TSS: ", mean(TSS))

    clf.fit(samples2, sample2label)
    pred_label = clf.predict(target2)
    print("fold 1 and 3 train test on 2")
    TN, FP, FN, TP = confusion_matrix(target2label, pred_label).ravel()

    print("TP: ", TP, "FP: ", FP, "TN: ", TN,"FN: ", FN)

    f1 = f1_score(target2label, pred_label, average="macro")

    pre = precision_score(target2label, pred_label, average="macro")

    recall = recall_score(target2label, pred_label, average=None)

    N = TN + FP

    P = TP + FN

    HSS1 = [(TN + TP - P) / N, (TP + TN - N) / P]

    HSS2 = [(2 * ((TN * TP) - (FP * FN)) / (N * (FP + TP) + (TN + FN) * P)), (2 * ((TP * TN) - (FN * FP)) / (P * (FN + TN) + (TP + FP) * N))]

    CH = [((TN + FN) * (TN + FP)) / (N + P), ((TP + FP) * (TP + FN)) / (P + N)]

    GS = [(TN - CH[0]) / (TN + FN + FP - CH[0]), (TP - CH[1]) / (TP + FP + FN - CH[1])]

    TSS = [recall[0] - (FN / (FN + TP)), recall[1] - (FP / (FP + TN))]

    


    print("f1: ",f1)

    print("pre: ", pre)

    print("recall: ", mean(recall))

    print("HSS1: ", mean(HSS1))

    print("HSS2: ", mean(HSS2))

    print("GS: ", mean(GS))

    print("TSS: ", mean(TSS))

    clf.fit(samples3, sample3label)
    pred_label = clf.predict(target3)
    print("fold 1 and 2 train test on 3")
    TN, FP, FN, TP = confusion_matrix(target3label, pred_label).ravel()

    print("TP: ", TP, "FP: ", FP, "TN: ", TN,"FN: ", FN)

    f1 = f1_score(target3label, pred_label, average="macro")

    pre = precision_score(target3label, pred_label, average="macro")

    recall = recall_score(target3label, pred_label, average=None)

    N = TN + FP

    P = TP + FN

    HSS1 = [(TN + TP - P) / N, (TP + TN - N) / P]

    HSS2 = [(2 * ((TN * TP) - (FP * FN)) / (N * (FP + TP) + (TN + FN) * P)), (2 * ((TP * TN) - (FN * FP)) / (P * (FN + TN) + (TP + FP) * N))]

    CH = [((TN + FN) * (TN + FP)) / (N + P), ((TP + FP) * (TP + FN)) / (P + N)]

    GS = [(TN - CH[0]) / (TN + FN + FP - CH[0]), (TP - CH[1]) / (TP + FP + FN - CH[1])]

    TSS = [recall[0] - (FN / (FN + TP)), recall[1] - (FP / (FP + TN))]

    


    print("f1: ",f1)

    print("pre: ", pre)

    print("recall: ", mean(recall))

    print("HSS1: ", mean(HSS1))

    print("HSS2: ", mean(HSS2))

    print("GS: ", mean(GS))

    print("TSS: ", mean(TSS))
