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
    all_cols = list(train.columns.values)

    remove_cols = ['id']
    numeric_cols = [x for x in all_cols if x not in remove_cols]
    # remove_cols = ['label','instance weight','migration code-change in msa','migration code-change in reg','migration code-move within reg','migration prev res in sunbelt']

    cat_cols = [x for x in all_cols if x not in numeric_cols and x not in remove_cols]
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

    y_conf = []
    y_test_num = []

    print len(y_test)
    print type(y_test)
    y_test = y_test.as_matrix()

    y_test_num = y_test


    dt_classifier = DecisionTreeClassifier(min_samples_split=10)
    dt_classifier.fit(x_train_num, y_train_num)
    tree.export_graphviz(dt_classifier, feature_names=used_cols, out_file="./v1/weighted_tree.dot")
    predicted = dt_classifier.predict(x_test)

    print(metrics.classification_report(y_test_num, predicted))

    probs = dt_classifier.predict_proba(x_test)

    # #Neg is 0 in probs and 0 in y_test
    # #pos is 1 in probs and 1 in y_test
    # print "Going to plot"

    for class_to_plot in [0, 1, 2, 3, 4, 5]:
        y_conf = []
        for i in range(len(y_test)):
            y_conf.append(probs[i][class_to_plot])
        precision, recall, thresholds = precision_recall_curve(y_test_num, y_conf, pos_label=class_to_plot)
        plt.plot(recall, precision)
        plt.axis([0, 1, 0, 1])
        plt.yticks(np.arange(0, 1.1, 0.1))

        print "Checking class", class_to_plot
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('DT without instance weights for class ' + str(class_to_plot))
        plt.grid(b=True, which='major', axis='both', color='black', linestyle='-', alpha=0.3)
        plt.xticks(np.arange(0, 1.1, 0.1))
        filename = "./v1/dt_weighted_" + str(class_to_plot) + ".png"
        plt.savefig(filename)
    # plt.clf()

    # #Learn without Instance weights
    plt.clf()

    temp = dt_classifier.feature_importances_
    print "NumCols : " + str(len(all_cols))
    print "Feautre Importances - length" + str(len(temp))
    print temp

    print "Feats"

    for i in range(0, len(temp)):
        print all_cols[i + 2], " ", temp[i]


if __name__ == '__main__':
    main()
