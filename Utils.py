import numpy as np
import itertools
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import GridSearchCV

# unused
# from sklearn.model_selection import KFold
# from sklearn.model_selection import RandomizedSearchCV
# from sklearn.model_selection import StratifiedKFold

from scipy.stats import sem
from sklearn import metrics
# from sklearn.grid_search import GridSearchCV
# from sklearn.grid_search import RandomizedSearchCV


def plot_confusion_matrix(clf, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting 'normalize = True'

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

    plt.show()
    """

    plt.imshow(clf, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")
    print (cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# K-fold cross-validation
def evaluate_cross_validation(clf, X, y):

    # Create a k-fold cross validation iterator
    # Evaluate on training data and mean score
    # evaluate_cross_validation(svc_1,X_train,y_train,5)
    # cv = StratifiedKFold(n_splits = 10,shuffle=True,random_state=0)

    # print ('{}{:^61} {}'.format('Interation','Training set observations', 'Testing set observations'))
    # for interation,data in enumerate(clf,start=1):
    #     print ('{:^9}{}{:^25}'.format(interation,data[0],data[1]))

    scores = cross_val_score(clf, X, y, cv=10, n_jobs=-1)
    print scores
    print("Mean score:{0:.3f} (+/-{1:.3f})").format(np.mean(scores), sem(scores))


def measure_performance(clf, X, y, show_accuracy=True, show_classification_report=True, show_confussion_matrix=True, show_plot=False):

    """ Predict with test data set """
    y_pred = cross_val_predict(clf, X, y, cv=10)
    if(show_accuracy):
        print "Accuracy: {0:.3f}".format(metrics.accuracy_score(y, y_pred)), "\n"
    if(show_classification_report):
        print "Classification report"
        print metrics.classification_report(y, y_pred), "\n"
    if show_confussion_matrix:
        print "Confusion matrix"
        print metrics.confusion_matrix(y, y_pred), "\n"
    if show_plot:
        plot_cross_validation(y_pred, y)


def train_and_evaluate(clf, X_train, y_train, X_test, y_test):
    """ Function to perform training on the training set and
        evaluate the performance on the testing set """
    clf.fit(X_train, y_train)
    print "Accuracy on training set: "
    evaluate_cross_validation(clf, X_train, y_train)

    print "Accuracy on testing set: "
    print clf.score(X_test, y_test)

    measure_performance(clf, X_test, y_test)


def plot_cross_validation(y_pred, y):
    # y_pred = cross_val_predict(clf, X, y, cv=10)

    fig, ax = plt.subplots()
    ax.scatter(y, y_pred)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    plt.show()


def grid_seach(clf,tuned_parameters,cv,X_train,y_train,X_test,y_test,plot = False):
    """ Parameter estimation using grid search with cross validation
         Parameters
        ----------
        clf : object type that implements the "fit" and "predict" methods
            An object of that type which is cloned for each validation.

        title : string
            Title for the chart.

        X_train : array-like, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        X_test : array-like, shape (n_samples, n_features)
            Testing vector, where n_samples is the number of samples and
            n_features is the number of features.

        y_train : array-like, shape (n_samples) or (n_samples, n_features), optional
            Target relative to X for predict;

        y_test : array-like, shape (n_samples) or (n_samples, n_features),
            Target relative to X for testing;

        cv : int, cross-validation generator or an iterable, optional
            Determines the cross-validation splitting strategy.
            Possible inputs for cv are:
              - None, to use the default 3-fold cross-validation,
              - integer, to specify the number of folds.
              - An object to be used as a cross-validation generator.
              - An iterable yielding train/test splits.

            For integer/None inputs, if ``y`` is binary or multiclass,
            :class:`StratifiedKFold` used. If the estimator is not a classifier
            or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

            Refer :ref:`User Guide <cross_validation>` for the various
            cross-validators that can be used here.

        n_jobs : integer, optional
            Number of jobs to run in parallel (default 1).
        tuned_parameters : dictionary
            dictionary of tune parameter

    """

    clf = GridSearchCV(clf,param_grid= tuned_parameters, cv=cv,n_jobs=-1, pre_dispatch=2, refit=True,
                      error_score=0, verbose=2, scoring='accuracy')
    clf.fit(X_train, y_train)
    print("Best parameters set found on training set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on training set:")
    print()

    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(metrics.classification_report(y_true, y_pred))
    print()

    if(plot):
        plt.plot(clf.cv_results_['params'],means)
        plt.xlabel('Value of params')
        plt.ylabel("Cross validated Accuracy")


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

def plot_KNearNeibours(k_range,k_scores):
    plt.plot(k_range,k_scores)
    plt.xlabel('Value of K for KNN')
    plt.ylabel('Cross-validated Accuracy')

def simple_line_plot(X,y,figure_no):
    plt.figure(figure_no)
    plt.plot(X,y)
    plt.xlabel('X values')
    plt.ylabel('y values')
    plt.title('Simple Line')

def simple_dots(X,y,figure_no):
    plt.figure(figure_no)
    plt.plot(X,y,'or')
    plt.xlabel('X values')
    plt.ylabel('y values')
    plt.title('Simple dots')

def simple_scatter(X,y,figure_no):
    plt.figure(figure_no)
    plt.scatter(X,y)
    plt.xlabel('X values')
    plt.ylabel('y values')
    plt.title('Simple Scatter')

def simple_scatter_with_color(X,y,labels,figure_no):
    plt.figure(figure_no)
    plt.scatter(X,y,c=labels)
    plt.xlabel('X values')
    plt.ylabel('y values')
    plt.title('Scatter with color')




