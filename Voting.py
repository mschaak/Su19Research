import pandas as pd
import os

data_dir = os.getcwd()

R_FOREST_FILENAME = data_dir + '\\rf_submissions.csv'
KNN_FILENAME = data_dir + '\\rf_submissions.csv'
SVM_FILENAME = data_dir + '\\svm_submissions.csv'
TEST_SET_SAMPLES = 173512

prediction_index = 0

with open(R_FOREST_FILENAME, 'r',) as r_forrest_file, \
        open(KNN_FILENAME, 'r',) as knn_file, \
        open(SVM_FILENAME, 'r',) as svm_file, \
        open('finalPrediction.csv', 'w', newline='') as output:
    svm_predictions = pd.read_csv(svm_file)
    knn_predictions = pd.read_csv(knn_file)
    rf_predictions = pd.read_csv(r_forrest_file)

    csv_writer = csv.writer(output, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(['Id', 'ClassLabel'])
    while prediction_index < TEST_SET_SAMPLES:
        pos_class = 0
        neg_class = 0

        if svm_predictions['ClassLabel'][prediction_index] == 1:
            pos_class += 1
        else:
            neg_class += 1
        if knn_predictions['ClassLabel'][prediction_index] == 1:
            pos_class += 1
        else:
            neg_class += 1
        if rf_predictions['ClassLabel'][prediction_index] == 1:
            pos_class += 1
        else:
            neg_class += 1

        if pos_class > neg_class:
            csv_writer.writerow([prediction_index + 1, 1])
        else:
            csv_writer.writerow([prediction_index + 1, 0])

        prediction_index += 1

        if prediction_index % 5000 == 0:
            print(prediction_index)
