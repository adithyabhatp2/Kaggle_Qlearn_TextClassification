"""
================
Confusion matrix
================

Example of confusion matrix usage to evaluate the quality
of the output of a classifier on the iris data set. The
diagonal elements represent the number of points for which
the predicted label is equal to the true label, while
off-diagonal elements are those that are mislabeled by the
classifier. The higher the diagonal values of the confusion
matrix the better, indicating many correct predictions.

The figures show the confusion matrix with and without
normalization by class support size (number of elements
in each class). This kind of normalization can be
interesting in case of class imbalance to have a more
visual interpretation of which class is being misclassified.

Here the results are not as good as they could be as our
choice for the regularization parameter C was not the best.
In real life applications this parameter is usually chosen
using :ref:`grid_search`.

"""

import itertools
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets

from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cmatrix, class_names,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.imshow(cmatrix, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # if normalize:
    #     cmatrix = cmatrix.astype('float') / cmatrix.sum(axis=1)[:, np.newaxis]
    #     print("Normalized confusion matrix")
    # else:
    #     print('Confusion matrix, without normalization')
    #
    # print(cmatrix)

    thresh = cmatrix.max() / 2.
    for i, j in itertools.product(range(cmatrix.shape[0]), range(cmatrix.shape[1])):
        plt.text(j, i, cmatrix[i, j],
                 horizontalalignment="center",
                 color="white" if cmatrix[i, j] > thresh else "black")

    plt.tight_layout()

    plt.show()


def main():
    # Compute confusion matrix
    # cnf_matrix = confusion_matrix(y_test, y_pred)

    cnf_matrix = np.matrix([
    [35,2,1,3,0,3],
    [1,6,1,1,1,1],
    [2,3,29,1,1,2],
    [0,2,2,13,2,0],
    [0,2,4,0,11,2],
    [3,3,0,1,2,38]
    ]).transpose()

    # [41,18,37,19,17,46],

    np.set_printoptions(precision=2)

    class_names = ["0", "1", "2", "3", "4", "5"]

    classNames = ["airplane","butterfly", "flower", "grand_piano", "starfish", "watch"]

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, class_names=classNames,
                          title='Test Set Confusion matrix')

    plt.show()
    # # Plot normalized confusion matrix
    # plt.figure()
    # plot_confusion_matrix(cnf_matrix, class_names=class_names, normalize=True,
    #                       title='Normalized confusion matrix')

    plt.show()


if __name__ == '__main__':
    main()
