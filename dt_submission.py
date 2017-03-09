import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer as DV
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.metrics import precision_recall_curve
from sklearn.cross_validation import StratifiedShuffleSplit
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
import sys
import csv


class MultiColumnLabelEncoder:
    def __init__(self, columns=None):
        self.columns = columns  # array of column names to encode

    def fit(self, X, y=None):
        return self  # not relevant here

    def transform(self, X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname, col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


def main():
    data_dir = './v1/'  # needs trailing slash

    # validation split, both files with headers and the Happy column
    train_file = data_dir + 'trainData.csv'
    test_file = data_dir + 'testData.csv'

    label_file = 'train_labels.csv'

    train = pd.read_csv(train_file)
    test = pd.read_csv(test_file)

    y_train = pd.read_csv(label_file)['Label']
    x_train = train.drop(['id'], axis=1)

    x_test = test.drop(['id'], axis=1)

    y_train_num = y_train
    x_train_num = x_train
    all_cols = list(train.columns.values)

    remove_cols = ['id']
    numeric_cols = [x for x in all_cols if x not in remove_cols]
    used_cols = [x for x in all_cols if x not in remove_cols]

    # handle numerical features
    x_num_train = train[numeric_cols].as_matrix()
    x_num_test = test[numeric_cols].as_matrix()

    x_train_count = x_num_train.shape[0]
    x_test_count = x_num_test.shape[0]

    x_num_combined = np.concatenate((x_num_train, x_num_test), axis=0)  # 0 -row 1 - col

    # scale numeric features to <0,1>
    max_num = np.amax(x_num_combined, 0)

    x_num_combined = np.true_divide(x_num_combined, max_num)  # scale by max. truedivide needed for decimals
    x_num_train = x_num_combined[0:x_train_count]
    x_num_test = x_num_combined[x_train_count:]


    dt_classifier = DecisionTreeClassifier(min_samples_split=10)
    dt_classifier.fit(x_train_num, y_train_num)
    tree.export_graphviz(dt_classifier, feature_names=used_cols, out_file="./v1/weighted_tree.dot")
    predicted = dt_classifier.predict(x_test)

    ids = test['id'].as_matrix()
    output = zip(ids,predicted)

    outFilePath = "./v1/predictions.csv"
    with open(outFilePath, 'w') as outFile:
        outFile.write("Id,Prediction\n")
        writer = csv.writer(outFile)
        writer.writerows(output)
        # for opLine in output:
        #     outFile.write(str(opLine))






if __name__ == '__main__':
    main()
