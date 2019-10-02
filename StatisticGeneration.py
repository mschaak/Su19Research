import os
from typing import Dict, Any, Union
from datetime import datetime
from scipy.stats import kurtosis, skew
import time
import numpy as np
import json

import csv

SUMMARY_TYPE = 'TopSixDropNA'

# FEATURE_LIST = ['TOTUSJH', 'TOTBSQ', 'TOTPOT', 'TOTUSJZ', 'ABSNJZH', 'USFLUX', 'SAVNCPP', 'TOTFZ', 'MEANPOT', 'EPSZ',
#                 'MEANSHR', 'SHRGT45', 'MEANGAM', 'MEANGBT', 'MEANGBZ', 'MEANGBH', 'MEANJZH', 'TOTFY', 'MEANJZD',
#                 'MEANALP', 'TOTFX', 'EPSY', 'EPSX', 'R_VALUE', 'XR_MAX']
# FEATURE_LIST = ['R_VALUE']
FEATURE_LIST = ['TOTUSJH', 'TOTBSQ', 'TOTPOT', 'TOTUSJZ', 'ABSNJZH', 'USFLUX', 'SAVNCPP']


def compute_derivative_array(feature_array):
    output_array = np.zeros(len(feature_array) - 1)
    for index in range(len(feature_array) - 1):
        output_array[index] = feature_array[(index + 1)] - feature_array[index]
    return output_array


def compute_mean(feature_array):
    return np.mean(feature_array)


def compute_std(feature_array):
    return np.std(feature_array, dtype=np.float64)


def compute_skew(feature_array, mean, std):
    return skew(feature_array)


def compute_kurt(feature_array, mean_val, std_val):
    return kurtosis(feature_array)


def generate_feature_list():
    feature_list = ["id", "class_num"]
    sub_list = ["mean", "std", "skew", "kurt"]
    i = 0
    while i < 2:
        if i == 0:
            for feat in FEATURE_LIST:
                for post in sub_list:
                    feature_list.append(feat + "-" + post)
        # Separate Section For First Order Derivative
        else:
            for feat in FEATURE_LIST:
                for post in sub_list:
                    feature_list.append(feat + "-" + post + "-derivative")
        i += 1
    return feature_list


os.chdir("../Data")

now = datetime.now()
now = now.__format__('%m-%d-%y--%H-%M')
file_names = {'fold1DropNa.json': 'fold1Statistics' + SUMMARY_TYPE + '-' + str(now) + '.csv',
              'fold2DropNa.json': 'fold2Statistics' + SUMMARY_TYPE + '-' + str(now) + '.csv',
              'fold3DropNa.json': 'fold3Statistics' + SUMMARY_TYPE + '-' + str(now) + '.csv'
              }

fold_1_time = 0
fold_2_time = 0
fold_3_time = 0
test_time = 0
fold_cnt = 1

for fold in file_names:
    start_time = time.time()
    with open(fold, 'r', ) as f, open(file_names[fold], 'w', newline='') as out:
        entry_number = 0
        # df = pd.DataFrame()
        columns = []
        for entry in f:
            delete = False
            # print(entry_number)
            if entry_number == 0:
                columns = generate_feature_list()
            json_object = json.loads(entry)
            output_row: Dict[Any, Union[np.ndarray, float]] = {}
            for feature in FEATURE_LIST:
                feature_set = json_object['values'][feature]
                # if not len(feature_set)
                # TODO: Update for empty arrays
                numpy_array = np.zeros(len(feature_set))
                j = 0
                for item in feature_set:
                    numpy_array[j] = item
                    j += 1

                mean = compute_mean(numpy_array)
                std = compute_std(numpy_array)
                output_row[feature + "-mean"] = mean
                output_row[feature + "-std"] = std
                output_row[feature + "-skew"] = compute_skew(numpy_array, mean, std)
                output_row[feature + "-kurt"] = compute_kurt(numpy_array, mean, std)

                derivative_array = compute_derivative_array(feature_set)
                mean = compute_mean(derivative_array)
                std = compute_std(derivative_array)
                output_row[feature + "-mean-derivative"] = mean
                output_row[feature + "-std-derivative"] = std
                output_row[feature + "-skew-derivative"] = compute_skew(derivative_array, mean, std)
                output_row[feature + "-kurt-derivative"] = compute_kurt(derivative_array, mean, std)

            output_row['class_num'] = json_object['classNum']
            output_row['id'] = entry_number + 1

            writer = csv.DictWriter(out, fieldnames=columns)
            if entry_number == 0:
                writer.writeheader()
            writer.writerow(output_row)

            if entry_number % 5000 == 0:
                now = datetime.now()
                print("   File: " + str(fold) + "  " + str(entry_number) + " rows processed at " + str(now))
            entry_number += 1

        now = datetime.now()
        print("   File: " + str(fold) + "  " + str(entry_number) + " rows processed at " + str(now))
        print("  -------------------------------------------------------------------")

    if fold_cnt == 1:
        fold_1_time = time.time() - start_time
        print("Total Fold 1 Time: " + str(fold_1_time/60.0) + " minutes")
        print("  -------------------------------------------------------------------")

    elif fold_cnt == 2:
        fold_2_time = time.time() - start_time
        print("Total Fold 2 Time: " + str(fold_2_time/60.0) + " minutes")
        print("  -------------------------------------------------------------------")

    elif fold_cnt == 3:
        fold_3_time = time.time() - start_time
        print("Total Fold 3 Time: " + str(fold_3_time/60.0) + " minutes")
        print("  -------------------------------------------------------------------")

    elif fold_cnt == 4:
        test_time = time.time() - start_time
        print("Total Test Time: " + str(test_time/60.0) + " minutes")
        print("  -------------------------------------------------------------------")

    fold_cnt += 1

print("Total Fold 1 Time: " + str(fold_1_time/60.0) + " minutes")
print("Total Fold 2 Time: " + str(fold_2_time/60.0) + " minutes")
print("Total Fold 3 Time: " + str(fold_3_time/60.0) + " minutes")
print("Total Test Time: " + str(test_time/60.0) + " minutes")

print("Total Runtime: " + str((fold_1_time + fold_2_time + fold_3_time + test_time)/3600.0) + " hours")
