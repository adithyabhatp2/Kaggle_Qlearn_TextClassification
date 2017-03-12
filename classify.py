import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer as DV
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn import neighbors
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
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
    version_dir = './v6/'  # needs trailing slash

    # validation split, both files with headers and the Happy column
    train_file = version_dir + 'trainData.csv'
    test_file = version_dir + 'testData.csv'

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


    classifierType = "SVM"

    print "Classifier: "+classifierType

    if classifierType == "DT":
        classifier = DecisionTreeClassifier(min_samples_leaf=8)
        classifier.fit(x_train_num, y_train_num)
        tree.export_graphviz(classifier, feature_names=used_cols, out_file=version_dir+"sub_weighted_tree.dot")

    elif classifierType == "Ada":
        classifier = AdaBoostClassifier(n_estimators=200, learning_rate=0.01)
        classifier.fit(x_train_num, y_train_num)

    elif classifierType == "SVM":
        classifier = svm.SVC(probability=True, C=1, kernel='linear')
        classifier.fit(x_train_num, y_train_num)

    elif classifierType == "RF":
        classifier = RandomForestClassifier(n_estimators=200, n_jobs=-1)
        classifier.fit(x_train_num, y_train_num)

    elif classifierType == "NB":
        classifier = GaussianNB()
        classifier.fit(x_train_num, y_train_num)

    elif classifierType == "KNN":
        classifier = neighbors.KNeighborsClassifier(n_neighbors=5)
        classifier.fit(x_train_num, y_train_num)


    predicted = classifier.predict(x_test)

    ids = test['id'].as_matrix()
    output = zip(ids,predicted)

    outFilePath = version_dir+"predictions.csv"
    with open(outFilePath, 'w') as outFile:
        outFile.write("Id,Prediction\n")
        writer = csv.writer(outFile)
        writer.writerows(output)



if __name__ == '__main__':
    main()
