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
import math
warnings.filterwarnings('ignore')

df = pd.read_csv("testSet.json_avg_vals.csv")

print(df.isnull().sum())
#print(df)



df = df.drop(["LABEL"], axis = 1)

headers = list(df.columns.values)    

#df_n = df[pd.isnull(df[headers])]
na_free = df.dropna()
only_na = df[~df.index.isin(na_free.index)]
only_na = only_na.reset_index()
df_pred = pd.read_csv('prediction.csv')
print(only_na)
for ids in df_pred["Id"]:
    df_pred.at[ids, 'ClassLabel'] = 1
for ids in only_na["ID"]:
    
    for id1 in df_pred["Id"]:
        if (ids) == id1:
            print("__________________")
            print(df_pred.at[ids, 'ClassLabel'])
            df_pred.at[ids, 'ClassLabel'] = 0
            print(df_pred.at[ids, 'ClassLabel'])
            print("__________________")
df_pred.to_csv("replaced_ones_to_Zeroes.csv")
            
