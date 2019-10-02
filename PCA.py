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
from sklearn.decomposition import PCA
warnings.filterwarnings('ignore')

def build_df():
    df1 = pd.read_csv("fold1Training.json_avg_med.csv")
    df2 = pd.read_csv("fold2Training.json_avg_med.csv")
    df3 = pd.read_csv("fold3Training.json_avg_med.csv")
    frames = [df2,df2,df3]
    df_keys = pd.concat(frames, keys = ['x','y','z'])
    
    return df_keys
def preProc(feature = ['TOTUSJH', "TOTPOT", "TOTUSJZ", "ABSNJZH", "USFLUX","MEANPOT", "EPSZ"]):
    df = build_df()
    features = ['TOTUSJH', "TOTPOT", "TOTUSJZ", "ABSNJZH", "USFLUX","MEANPOT", "EPSZ","LABEL"]
    df = df[features]
    #df_keys = df_keys[(df_keys[feature] != 0).all(1)]
    labels = df['LABEL']
    df.fillna(df.mean(), inplace=True)
    df= df.drop(['LABEL'], axis = 1)
    print("test")
    tot_avg = df.mean(axis = 0)
    tot_std = df.std(axis = 0)
    tot_med = df.median(axis = 0)
    tot_med_list = tot_med.tolist()
    tot_avg_list = tot_avg.tolist()
    tot_std_list = tot_std.tolist()
    counter = 0
    for feat in feature:
        df[feat] = df[feat]-tot_avg_list[counter]
        df[feat] =df[feat]/tot_std_list[counter]
        counter+=1
    df["LABEL"] = labels
    return df
def build_set():
    feature = ['TOTUSJH', "TOTPOT", "TOTUSJZ", "ABSNJZH", "USFLUX","MEANPOT", "EPSZ"]
    df = preProc()
    x = df[feature]
    y = df['LABEL']
    return x,y
def PCABuild():
    x,y = build_set()
    print("WEEEEEE")
    pca = PCA(n_components=2)
    pc = pca.fit_transform(x)
    print(pc)
    principalDf = pd.DataFrame(pc ,columns = ['principal component 1', 'principal component 2'])
    finalDf = pd.concat([principalDf, y])
    print(finalDf)
    print(pca.explained_variance_ratio_)
    pass
