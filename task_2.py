# Naive Bayes

from task_1 import *
from Ults import *

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import  KNeighborsClassifier
from sklearn.svm import SVC


from sklearn.naive_bayes import MultinomialNB,GaussianNB
from sklearn.model_selection import learning_curve, ShuffleSplit
from sklearn.metrics import brier_score_loss, precision_score, recall_score,f1_score
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import LogisticRegression

filename = "Naive_BayesClassifier.pkl"

# Get the data
data = DataProcessing()

# Get the transformed data
X_train_scale,X_test_scale = data.standard_scaler()
# X_train_nor,X_test_nor = data.normalize()
# X_train_minmax,X_test_minmax = data.min_max_scaler()
# X_train_abs,X_test_abs = data.maxAbs_scaler()

def Evaluate_accuracy(clf,X_train,X_test):
    """ This function use to print out the accuracy score of classifier
        with and without feature scaling

        This function is for testing purpose only and will be remove in future
    """
    pred_train = clf.predict(X_train)
    print('\nPrediction accuracy for the training dataset')
    print('{:.2%}'.format(metrics.accuracy_score(data.y_train, pred_train)))

    pred_test = clf.predict(X_test)

    print('\nPrediction accuracy for the test dataset')
    print('{:.2%}\n'.format(metrics.accuracy_score(data.y_test, pred_test)))

def plot_transform(X_train_std):
    """ This function use to plot the range value of feature after scaling
        X_train_std is a array like

        This function is for testing purpose only and will be remove in future
    """
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 4))
    for l, c, m in zip(range(1, 4), ('blue', 'red', 'green'), ('^', 's', 'o')):
        ax1.scatter(data.X_train[data.y_train == l, 0], data.X_train[data.y_train == l, 1],
                    color=c,
                    label='class %s' % l,
                    alpha=0.5,
                    marker=m
                    )

    for l, c, m, in zip(range(1, 4), ('blue', 'red', 'green'), ('^', 's', 'o')):
        ax2.scatter(X_train_std[data.y_train == l, 0], X_train_std[data.y_train == l, 1],
                    color=c,
                    label='class %s' % l,
                    alpha=0.5,
                    marker=m
                    )

    ax1.set_title('Transformed NON-standardized training dataset after PCA')
    ax2.set_title('Transformed standardized training dataset after PCA')

    for ax in (ax1, ax2):
        ax.set_xlabel('1st principal component')
        ax.set_ylabel('2nd principal component')
        ax.legend(loc='upper right')
        ax.grid()
    plt.tight_layout()

    plt.show()

def test_scaling_features():
    """ This function for testing purpose only
        Will be removed in future
    """

    clf = SVC(kernel='linear',C=1)
    clf.fit(data.X_train, data.y_train)
    Evaluate_accuracy(clf,data.X_train,data.X_test)

    print "Standard Scaler"
    clf_std = SVC(kernel='linear',C=1)
    clf_std.fit(X_train_scale, data.y_train)
    Evaluate_accuracy(clf_std,X_train_scale,X_test_scale)

    print "Normalize"
    clf_nor = SVC(kernel='linear',C=1)
    clf_nor.fit(X_train_nor, data.y_train)
    Evaluate_accuracy(clf_nor,X_train_nor,X_test_nor)

    print "Min Max scaler"
    clf_minmax = SVC(kernel='linear',C=1)
    clf_minmax.fit(X_train_minmax, data.y_train)
    Evaluate_accuracy(clf_minmax,X_train_minmax,X_test_minmax)

    print "Max Abs scaler"
    clf_abs = SVC(kernel='linear',C=1)
    clf_abs.fit(X_train_abs, data.y_train)
    Evaluate_accuracy(clf_abs,X_train_abs,X_test_abs)

    #plot_transform(X_train_scale)


""" Result:
        Scaling the whole data set will give a very poor accuracy
            No feature Scaling:  45.75%
            Standard Scaler: 48.99%
            Normalize: 37.81%
            MinMax: 8.96%
            MaxAbs: 8.95%
        Only scale the first 10 features
            No feature Scaling: 45.75%
            Standard Scaler: 63%
            Normalize: 47.24% (l2)
            Normalize: 46.74% (l1)
            MinMax: 62.78%
            MaxAbs: 62.77%


    Overall: Standard scaler return the best result
            K-NearestNeighbour: normalize return the best accuracy score
            SVC take too long to fit the data

    Thing to try: Dimension Reduce
"""
#test_scaling_features()

clf = GaussianNB()
train_and_evaluate(clf,X_train_scale,data.y_train,X_test_scale,data.y_test)


"""
[ 0.62774358  0.62715639  0.63132745  0.62883696  0.6322919   0.63297291
  0.63339813  0.63339071  0.62902213  0.62823758]
Mean score:0.630 (+/-0.001)
Accuracy on testing set:
0.630037176427
Accuracy: 0.632

Classification report
             precision    recall  f1-score   support

          1       0.63      0.68      0.65     31946
          2       0.73      0.66      0.69     42414
          3       0.50      0.67      0.57      5336
          4       0.44      0.39      0.42       388
          5       0.17      0.21      0.19      1458
          6       0.33      0.29      0.31      2578
          7       0.25      0.26      0.26      3032

avg / total       0.64      0.63      0.63     87152


Confussion matrix
[[21566  7953   226     0   319    97  1785]
 [10234 27865  1827     0  1118   818   552]
 [    0  1066  3592   150    18   510     0]
 [    0     1   177   153     0    57     0]
 [    1  1059    53     0   312    33     0]
 [    0   465  1284    44    25   760     0]
 [ 2202    17    20     0     0     0   793]]
"""
Save_Classifier(clf,file_name=filename)