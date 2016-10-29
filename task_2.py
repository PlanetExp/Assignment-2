# Naive Bayes

from task_1 import DataProcessing
from task_1 import Save_Classifier
# from task_1 import Load_classifier
from Utils import train_and_evaluate

from sklearn.tree import DecisionTreeClassifier
# from sklearn.svm import SVC

from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier

from time import time
import itertools
import numpy as np
from sklearn import metrics

# from sklearn.neighbors import  KNeighborsClassifier

# unused
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.model_selection import learning_curve
# from sklearn.model_selection import ShuffleSplit
# from sklearn.metrics import brier_score_loss
# from sklearn.metrics import precision_score
# from sklearn.metrics import recall_score
# from sklearn.metrics import f1_score
# from sklearn.calibration import CalibratedClassifierCV
# from sklearn.calibration import calibration_curve
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import BaggingClassifier
# from sklearn.ensemble import AdaBoostClassifier
# from sklearn.ensemble import GradientBoostingClassifier

filename = 'Naive_BayesClassifier'

# --------------------------
# Loading data

data = DataProcessing()

# normalise
X_train_scale, X_test_scale = data.normalize()

# Principal component analysis
pca = PCA()
X_train_scale_pca = pca.fit_transform(X_train_scale)
X_test_scale_pca = pca.fit_transform(X_test_scale)

# Principal component analysis
# pca = PCA()
# X_transform_scale = pca.fit_transform(X_train_scale)
# X_test_transform_scale = pca.fit_transform(X_test_scale)

# X_transform2 = pca.fit_transform(data.X_train)
# X_test_transform2 = pca.fit_transform(data.X_test)

# lda = LinearDiscriminantAnalysis()
# X_transform3 = lda.fit(X_train_scale, data.y_train).transform(X_train_scale)
# X_test_transform3 = lda.fit(X_test_scale, data.y_test).transform(X_test_scale)

# X_transform4 = lda.fit(data.X_train, data.y_train).transform(data.X_train)
# X_test_transform4 = lda.fit(data.X_test, data.y_test).transform(data.X_test)

# qda = QuadraticDiscriminantAnalysis
# X_transform5 = qda.fit(X_train_scale, data.y_train).transform(X_train_scale)
# X_test_transform5 = qda.fit(X_test_scale, data.y_test).transform(X_test_scale)

# X_transform6 = qda.fit_transform(data.X_train)
# X_test_transform6 = qda.fit_transform(data.X_test)
# X_train_nor, X_test_nor = data.normalize()
# X_train_minmax, X_test_minmax = data.min_max_scaler()
# X_train_abs, X_test_abs = data.maxAbs_scaler()

# ------------------------------------------
# Running test

score = []

def benchmark(clf_class, params, name, X_train=data.X_train, X_test=data.X_test):
    print("parameters:", params)
    t0 = time()
    clf = clf_class(**params).fit(X_train, data.y_train)
    print 'done in %fs' % (time() - t0)

    if hasattr(clf, 'coef_'):
        print("Percentage of non zeros coef: %f"
              % (np.mean(clf.coef_ != 0) * 100))
    print 'Predicting the outcomes of the testing set'
    t0 = time()
    pred = clf.predict(X_test)
    accuracy = metrics.accuracy_score(data.y_test, pred)

    print 'Accuracy: {0:.3f}%\n'.format(accuracy)
    print 'done in {0}s'.format((time() - t0))

    print 'Classification report on test set for classifier:'
    print clf
    print ''
    print classification_report(data.y_test, pred)

    cm = confusion_matrix(data.y_test, pred)
    print 'Confusion matrix:'
    print cm

    # Show confusion matrix
    plt.figure()
    plt.imshow(cm, interpolation='nearest')

    # plot text
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    # plot labels
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    # plt.matshow(cm)
    plt.title('Classifier: {0}, Accuracy: {1}'.format(name, accuracy))
    plt.colorbar()


# print("Testbenching a linear classifier...")
# parameters = {
#     'loss': 'hinge',
#     'penalty': 'l2',
#     'n_iter': 50,
#     'alpha': 0.00001,
#     'fit_intercept': True,
# }

# benchmark(SGDClassifier, parameters, 'SGD')

print 'Naive Bayes'

print("Testbenching a BernoulliNB classifier...")
parameters = {'alpha': 0.01}
benchmark(BernoulliNB, parameters, 'MultinomialNB')

print("Testbenching a BernoulliNB normalised data")
parameters = {'alpha': 0.01}
benchmark(BernoulliNB, parameters, 'MultinomialNB normalised', X_train_scale, X_test_scale)

print("Testbenching a BernoulliNB with PCA")
parameters = {'alpha': 0.01}
benchmark(BernoulliNB, parameters, 'MultinomialNB with PCA', X_train_scale_pca, X_test_scale_pca)

# print("Testbenching a GaussianNB classifier...")
# parameters = {}
# benchmark(GaussianNB, parameters, 'GaussianNB')

print("Testbenching a GaussianNB classifier with normalised data")
parameters = {}
benchmark(GaussianNB, parameters, 'GaussianNB with normalised data', X_train_scale, X_test_scale)

print("Testbenching a GaussianNB with PCA")
parameters = {}
benchmark(GaussianNB, parameters, 'GaussianNB with PCA', X_train_scale_pca, X_test_scale_pca)

plt.show()








# print 'GaussianNB'
# clf0 = GaussianNB()
# train_and_evaluate(clf0, data.X_train, data.y_train, data.X_test, data.y_test)

# print 'GaussianNB normalised'
# clf1 = GaussianNB()
# train_and_evaluate(clf1, X_train_scale, data.y_train, X_test_scale, data.y_test)

# print 'Gaussian'
# clf3 = GaussianNB()
# train_and_evaluate(clf3, X_transform3, data.y_train, X_test_transform3, data.y_test)

# print 'GaussianNB '
# clf4 = GaussianNB()
# train_and_evaluate(clf4, X_transform4, data.y_train, X_test_transform4, data.y_test)


# clf1 = DecisionTreeClassifier()
# train_and_evaluate(clf1,data.X_train,data.y_train,data.X_test,data.y_test)
#
# clf2 = DecisionTreeClassifier()
# train_and_evaluate(clf2,X_train_scale,data.y_train,X_test_scale,data.y_test)
#
# print ("PCA")
# clf5 = DecisionTreeClassifier()
# train_and_evaluate(clf5,X_transform,data.y_train,X_test_transform,data.y_test)
#
# clf6 = DecisionTreeClassifier()
# train_and_evaluate(clf6,X_transform2,data.y_train,X_test_transform2,data.y_test)

# clf4 = GaussianNB()
# train_and_evaluate(clf4,X_transform_n,data.y_train,X_test_transform_n,data.y_test)





# print "Bagging"
# bagging = BaggingClassifier(GaussianNB(),n_estimators=20,random_state=9)
# train_and_evaluate(bagging,data.X_train,data.y_train,data.X_test,data.y_test)
#
# print "Ada"
# boosting = AdaBoostClassifier(GaussianNB(),n_estimators=20,random_state=9)
# train_and_evaluate(boosting,data.X_train,data.y_train,data.X_test,data.y_test)
#
# print "Gradient"
# gradient = GradientBoostingClassifier(GaussianNB(),n_estimators=20,random_state=9)
# train_and_evaluate(gradient,data.X_train,data.y_train,data.X_test,data.y_test)

# Save_Classifier(clf, file_name=filename)

# -------------------------------
# Results

"""
    Non-normalize: 0.456
    Normalize: 0.378
    LDA + non-normalize: 0.653
    LDA + normalize: 0.645
    non-normalize + PCA: 0.558
    normalize + PCA: 0.525
    PCA + non-normalize + whiten: 0.446
    normalize + PCA + whiten: 0.404
    PCA + normalize: 0.463
    PCA + normalize + whiten: 0.567
"""
"""
    Decision Tree
    Non-normalize: 0.859
    Normalize: 0.854

    PCA + non-normalize: 0.844
    PCA + normalize: 0.829

    Fix normalize: 0.853
"""


# ----------- Graveyard -------------

# def test_scaling_features():
#     """ This function for testing purpose only
#         Will be removed in future
#     """

#     clf = SVC(kernel='linear',C=1)
#     clf.fit(data.X_train, data.y_train)
#     Evaluate_accuracy(clf,data.X_train,data.X_test)

#     print "Standard Scaler"
#     clf_std = SVC(kernel='linear',C=1)
#     clf_std.fit(X_train_scale, data.y_train)
#     Evaluate_accuracy(clf_std,X_train_scale,X_test_scale)

#     print "Normalize"
#     clf_nor = SVC(kernel='linear',C=1)
#     clf_nor.fit(X_train_nor, data.y_train)
#     Evaluate_accuracy(clf_nor, X_train_nor, X_test_nor)

#     print "Min Max scaler"
#     clf_minmax = SVC(kernel='linear', C=1)
#     clf_minmax.fit(X_train_minmax, data.y_train)
#     Evaluate_accuracy(clf_minmax, X_train_minmax, X_test_minmax)

#     print "Max Abs scaler"
#     clf_abs = SVC(kernel='linear',C=1)
#     clf_abs.fit(X_train_abs, data.y_train)
#     Evaluate_accuracy(clf_abs, X_train_abs, X_test_abs)

    # plot_transform(X_train_scale)

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
# test_scaling_features()



    # def plot_transform(X_train_std):
#     """ This function use to plot the range value of feature after scaling
#         X_train_std is a array like

#         This function is for testing purpose only and will be remove in future
#     """
#     fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 4))
#     for l, c, m in zip(range(1, 4), ('blue', 'red', 'green'), ('^', 's', 'o')):
#         ax1.scatter(data.X_train[data.y_train == l, 0], data.X_train[data.y_train == l, 1],
#                     color=c,
#                     label='class %s' % l,
#                     alpha=0.5,
#                     marker=m
#                     )

#     for l, c, m, in zip(range(1, 4), ('blue', 'red', 'green'), ('^', 's', 'o')):
#         ax2.scatter(X_train_std[data.y_train == l, 0], X_train_std[data.y_train == l, 1],
#                     color=c,
#                     label='class %s' % l,
#                     alpha=0.5,
#                     marker=m
#                     )

#     ax1.set_title('Transformed NON-standardized training dataset after PCA')
#     ax2.set_title('Transformed standardized training dataset after PCA')

#     for ax in (ax1, ax2):
#         ax.set_xlabel('1st principal component')
#         ax.set_ylabel('2nd principal component')
#         ax.legend(loc='upper right')
#         ax.grid()
#     plt.tight_layout()

#     plt.show()

# print 'Decision Tree'
# clf2 = DecisionTreeClassifier()
# train_and_evaluate(clf2, X_train_scale, data.y_train, X_test_scale, data.y_test)
