import csv

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

os.chdir("../Data")
data_dir = os.getcwd()

FEATURE_LIST = ['TOTUSJH', 'TOTBSQ', 'TOTPOT', 'TOTUSJZ', 'ABSNJZH', 'USFLUX', 'SAVNCPP']

scaler = StandardScaler()

# TODO: Sample flares and non-flares equally
# TODO: Random sample w/o replacement

fold_1 = pd.read_csv(data_dir + "\\fold1TopSevenDropNASTD.csv")
fold_2 = pd.read_csv(data_dir + "\\fold2TopSevenDropNASTD.csv")
fold_3 = pd.read_csv(data_dir + "\\fold3TopSevenDropNASTD.csv")

fold_1 = fold_1.dropna()
fold_2 = fold_2.dropna()
fold_3 = fold_3.dropna()

fold_1_X = fold_1.drop(['id', 'class_num'], axis=1)
fold_2_X = fold_2.drop(['id', 'class_num'], axis=1)
fold_3_X = fold_3.drop(['id', 'class_num'], axis=1)

fold_1_y = fold_1['class_num']
fold_2_y = fold_2['class_num']
fold_3_y = fold_3['class_num']

scaler = scaler.partial_fit(fold_1_X)
scaler = scaler.partial_fit(fold_2_X)
scaler = scaler.partial_fit(fold_3_X)

fold_1_X_transformed = scaler.transform(fold_1_X)
fold_2_X_transformed = scaler.transform(fold_2_X)
fold_3_X_transformed = scaler.transform(fold_3_X)

f1_index_list = np.zeros(len(fold_1_X_transformed), dtype=int)
f2_index_list = np.zeros(len(fold_2_X_transformed), dtype=int)
f3_index_list = np.zeros(len(fold_3_X_transformed), dtype=int)

for i in range(len(fold_1_X_transformed)):
    f1_index_list[i] = i

for i in range(len(fold_2_X_transformed)):
    f2_index_list[i] = i

for i in range(len(fold_3_X_transformed)):
    f3_index_list[i] = i


f1_indexes = pd.Series(f1_index_list)
f2_indexes = pd.Series(f2_index_list)
f3_indexes = pd.Series(f3_index_list)

fold_1_X_transformed = pd.DataFrame(data=fold_1_X_transformed,
                                    index=f1_indexes,
                                    columns=FEATURE_LIST)
fold_2_X_transformed = pd.DataFrame(data=fold_2_X_transformed,
                                    index=f2_indexes,
                                    columns=FEATURE_LIST)
fold_3_X_transformed = pd.DataFrame(data=fold_3_X_transformed,
                                    index=f3_indexes,
                                    columns=FEATURE_LIST)

f12_X = pd.concat([fold_1_X_transformed, fold_2_X_transformed], ignore_index=True)
f23_X = pd.concat([fold_2_X_transformed, fold_3_X_transformed], ignore_index=True)
f13_X = pd.concat([fold_1_X_transformed, fold_3_X_transformed], ignore_index=True)
f12_y = pd.concat([fold_1_y, fold_2_y], ignore_index=True)
f23_y = pd.concat([fold_2_y, fold_3_y], ignore_index=True)
f13_y = pd.concat([fold_1_y, fold_3_y], ignore_index=True)

fold_1_scatter = []
fold_2_scatter = []
fold_3_scatter = []

with open('KNNResults.csv', 'w', newline='') as results:
    for i in range(0, 9999, 500):
        output_row = ""
        model12 = KNeighborsClassifier(p=2, n_neighbors=i+1)
        model23 = KNeighborsClassifier(p=2, n_neighbors=i+1)
        model13 = KNeighborsClassifier(p=2, n_neighbors=i+1)

        model12.fit(f12_X, f12_y)
        model23.fit(f23_X, f23_y)
        model13.fit(f13_X, f13_y)

        pred3 = model12.predict(fold_3_X_transformed)
        pred1 = model23.predict(fold_1_X_transformed)
        pred2 = model13.predict(fold_2_X_transformed)

        print("N = " + str(i+1) + " fold 1: " + str(f1_score(fold_1_y, pred1, pos_label=1)))
        print("N = " + str(i+1) + " fold 2: " + str(f1_score(fold_2_y, pred2, pos_label=1)))
        print("N = " + str(i+1) + " fold 3: " + str(f1_score(fold_3_y, pred3, pos_label=1)))
        output_row += str(i + 1) + ","
        output_row += str(f1_score(fold_1_y, pred1, pos_label=1)) + ','
        output_row += str(f1_score(fold_2_y, pred2, pos_label=1)) + ','
        output_row += str(f1_score(fold_3_y, pred3, pos_label=1)) + '\n'

        results.write(output_row)
