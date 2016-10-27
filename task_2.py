# Naive Bayes

from task_1 import *
from Ults import *

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import  KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis,QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import MultinomialNB,GaussianNB
from sklearn.model_selection import learning_curve, ShuffleSplit
from sklearn.metrics import brier_score_loss, precision_score, recall_score,f1_score
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import  BaggingClassifier,AdaBoostClassifier,GradientBoostingClassifier

filename = "Naive_BayesClassifier.pkl"

# Get the data
data = DataProcessing()

# Get the transformed data
X_train_scale,X_test_scale = data.normalize()


# pca =PCA(whiten=True)
# X_transform = pca.fit_transform(X_train_scale)
# X_test_transform = pca.fit_transform(X_test_scale)
#
# X_transform2 = pca.fit_transform(data.X_train)
# X_test_transform2 = pca.fit_transform(data.X_test)
#
# normalizer = preprocessing.Normalizer(norm='l2')
# X_transform_n = normalizer.fit_transform(X_transform2)
# X_test_transform_n = normalizer.fit_transform(X_test_transform2)
#
# lda = LinearDiscriminantAnalysis()
# X_transform3 = lda.fit(X_train_scale,data.y_train).transform(X_train_scale)
# X_test_transform3 = lda.fit(X_test_scale,data.y_test).transform(X_test_scale)
#
# X_transform4 = lda.fit(data.X_train,data.y_train).transform(data.X_train)
#X_test_transform4 = lda.fit(data.X_test,data.y_test).transform(data.X_test)

# qda = QuadraticDiscriminantAnalysis
# X_transform5 = qda.fit(X_train_scale,data.y_train).transform(X_train_scale)
# X_test_transform5 = qda.fit(X_test_scale,data.y_test).transform(X_test_scale)
#
# X_transform6 = qda.fit_transform(data.X_train)
# X_test_transform6 = qda.fit_transform(data.X_test)
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

#clf1 = GaussianNB()
#train_and_evaluate(clf1,data.X_train,data.y_train,data.X_test,data.y_test)

clf2 = DecisionTreeClassifier()
train_and_evaluate(clf2,X_train_scale,data.y_train,X_test_scale,data.y_test)
#
# print ("LDA")
# clf3 = GaussianNB()
# train_and_evaluate(clf3,X_transform3,data.y_train,X_test_transform3,data.y_test)
#
# clf4 = GaussianNB()
# train_and_evaluate(clf4,X_transform4,data.y_train,X_test_transform4,data.y_test)


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

Save_Classifier(clf,file_name=filename)

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