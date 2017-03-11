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
import os
from plot_confusion_matrix import plot_confusion_matrix

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
    version_dir = './v3/'  # needs trailing slash

    # validation split, both files with headers and the Happy column
    train_file = version_dir + 'trainData.csv'

    label_file = 'train_labels.csv'


    train = pd.read_csv(train_file)[0:2000]
    test = pd.read_csv(train_file)[2000:]

    y_train = pd.read_csv(label_file)[0:2000]['Label']
    x_train = train.drop(['id'], axis=1)

    # print enc_cols

    y_test = pd.read_csv(label_file)[2000:].Label
    x_test = test.drop(['id'], axis=1)

    # y_train_num = []
    # for i in range(len(y_train)):
    #     if y_train[i] == 'Pos':
    #         y_train_num.append(1)
    #     else:
    #         y_train_num.append(0)

    y_train_num = y_train
    x_train_num = x_train
    all_col_headers = list(train.columns.values)

    remove_cols = ['id']
    numeric_cols = [x for x in all_col_headers if x not in remove_cols]
    # remove_cols = ['label','instance weight','migration code-change in msa','migration code-change in reg','migration code-move within reg','migration prev res in sunbelt']

    cat_cols = [x for x in all_col_headers if x not in numeric_cols and x not in remove_cols]
    used_cols = [x for x in all_col_headers if x not in remove_cols]

    # handle numerical features
    x_num_train = train[numeric_cols].as_matrix()
    x_num_test = test[numeric_cols].as_matrix()

    x_train_count = x_num_train.shape[0]
    x_test_count = x_num_test.shape[0]

    x_num_combined = np.concatenate((x_num_train, x_num_test), axis=0)  # 0 -row 1 - col

    # scale numeric features to <0,1>
    max_num = np.amax(x_num_combined, 0)

    x_num_combined = np.true_divide(x_num_combined, max_num)  # scale by max. truedivide needed for decimals
    x_train_num_scaled = x_num_combined[0:x_train_count]
    x_test_num_scaled = x_num_combined[x_train_count:]

    y_test_num = y_test.as_matrix()
    y_train_num = y_train.as_matrix()

    class_labels = ['Abbr', 'Human', 'Loc', 'Desc', 'Entity', 'Num']

    classifierType = "SVM"
    print "Classifier: "+classifierType

    if classifierType == "DT":
        classifier = DecisionTreeClassifier(min_samples_leaf=10)
        classifier.fit(x_train_num, y_train_num)
        tree.export_graphviz(classifier, feature_names=used_cols, out_file=version_dir+"weighted_tree.dot")

    elif classifierType == "Ada":
        # 200,0.01 - v1
        # 50, 0.1 - v2
        classifier = AdaBoostClassifier(n_estimators=200, learning_rate=0.1)
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
        classifier = neighbors.KNeighborsClassifier(n_neighbors=1)
        classifier.fit(x_train_num, y_train_num)

    predicted_test = classifier.predict(x_test)
    predicted_train = classifier.predict(x_train)

    print "\nMetrics classification report - Test"
    print(metrics.classification_report(y_test_num, predicted_test))
    print "Confusion Matrix report - Test"
    test_confusion_matrix = metrics.confusion_matrix(y_test_num, predicted_test)
    print test_confusion_matrix
    print "Test Correct predictions: ", np.trace(test_confusion_matrix)

    plot_confusion_matrix(test_confusion_matrix, class_labels)

    print ""

    print "\nMetrics classification report - Train"
    print metrics.classification_report(y_train_num, predicted_train)
    print "Confusion Matrix report - Train"
    train_confusion_matrix = metrics.confusion_matrix(y_train_num, predicted_train)
    print train_confusion_matrix
    print "Train correct predictions: ", np.trace(train_confusion_matrix)


    outputHeaderRow = list(all_col_headers)
    outputHeaderRow.insert(0, "Predicted")
    outputHeaderRow.insert(0, "Actual")
    trainOutputData = np.column_stack((y_train_num, predicted_train, train.as_matrix()))
    testOutputData = np.column_stack((y_test_num, predicted_test, test.as_matrix()))

    trainOp_wthHeader = np.vstack((outputHeaderRow, trainOutputData))
    testOp_wthHeader = np.vstack((outputHeaderRow, testOutputData))

    np.savetxt(fname = version_dir+'train_preds.csv',X = trainOp_wthHeader , delimiter=',',fmt="%s")
    np.savetxt(version_dir+'test_preds.csv',testOp_wthHeader , delimiter=',', fmt="%s")


    print "\nFeature Importances"
    featImps = classifier.feature_importances_
    featNames = np.array(all_col_headers[1:]).flatten()
    featVals = np.column_stack((featImps, featNames))
    featVals = featVals[featVals[:,0].argsort()[::-1]]

    print featVals

    # sys.stdin.read(1)



if __name__ == '__main__':
    main()
