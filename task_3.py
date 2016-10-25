# Decision Tree
import pickle
import numpy as np
import itertools
import matplotlib.pyplot as plt

from scipy.stats import  randint as sp_randint
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split,KFold
from time import time
from sklearn.tree import  DecisionTreeClassifier
from sklearn import metrics
from sklearn.pipeline import Pipeline
from Ults import  *
from task_1 import *

def report(results, n_top=3):
    """ Helper function to printout the report

    """
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")



# Get the data
data = DataProcessing()

# Get the transformed data
X_train_scale,X_test_scale = data.standard_scaler()

""" Decision Tree Classifier with entropy"""
clf_en = DecisionTreeClassifier(criterion='entropy')
train_and_evaluate(clf_en,X_train_scale,data.y_train,X_test_scale,data.y_test)

"""************************* Result *************************
Accuracy on trainning set:
[ 0.92164088  0.92208634  0.92605491  0.92419211  0.92382611  0.92420929
  0.92431053  0.92570619  0.92539943  0.92525465]
Mean score:0.924 (+/-0.000)
Accuracy on testing set:
0.926358545989
Accuracy: 0.833

Classification report
             precision    recall  f1-score   support

          1       0.83      0.83      0.83     31946
          2       0.86      0.86      0.86     42414
          3       0.80      0.81      0.81      5336
          4       0.65      0.64      0.65       388
          5       0.61      0.59      0.60      1458
          6       0.67      0.66      0.67      2578
          7       0.82      0.83      0.83      3032

avg / total       0.83      0.83      0.83     87152

Confussion matrix
[[26600  4783     9     0    67    15   472]
 [ 4767 36400   425     5   458   298    61]
 [    7   442  4299    87    29   472     0]
 [    0     2   102   250     0    34     0]
 [   70   486    28     0   864    10     0]
 [   27   311   480    40     7  1713     0]
 [  453    67     0     0     2     0  2510]]
"""

""" Decision Tree Classifier with gini"""
clf_gini = DecisionTreeClassifier(criterion='gini')
train_and_evaluate(clf_gini,X_train_scale,data.y_train,X_test_scale,data.y_test)

"""************************* Result *************************
Accuracy on trainning set:
[ 0.91789504  0.91582976  0.91795578  0.91556653  0.91546358  0.91319402
  0.9165148   0.91857852  0.91976996  0.9164051 ]
Mean score:0.917 (+/-0.001)
Accuracy on testing set:
0.919313383514
Accuracy: 0.821

Classification report
             precision    recall  f1-score   support

          1       0.82      0.82      0.82     31946
          2       0.85      0.85      0.85     42414
          3       0.79      0.78      0.79      5336
          4       0.65      0.68      0.66       388
          5       0.58      0.59      0.59      1458
          6       0.65      0.66      0.65      2578
          7       0.79      0.80      0.80      3032

avg / total       0.82      0.82      0.82     87152

Confussion matrix
[[26250  5040    17     0    76    17   546]
 [ 5115 35887   491     7   494   336    84]
 [   11   502  4184    86    29   524     0]
 [    0     4    80   262     0    42     0]
 [   71   480    25     0   865    16     1]
 [   18   320   472    47    18  1703     0]
 [  522    75     0     0     1     0  2434]]
"""

""" Todo
    Seem like Entropy give a better accuracy score
    * Run GridSearch with Criterion = 'entropy'
"""
# Build a pipeline
#pipeline = Pipeline([('clf',DecisionTreeClassifier)])

# use a full grid over all parameters
# param_grid = {"max_depth": [3, None],
#               "max_features": [1, 3, 10],
#               "min_samples_split": [1, 3, 10],
#               "min_samples_leaf": [1, 3, 10],
#               "criterion": ["gini", "entropy"]}

# # run randomized search
# n_iter_search = 20
# random_search = RandomizedSearchCV(estimator=clf, param_distributions=param_dist,n_jobs=-1,cv=5,
#                                    n_iter=n_iter_search)
# start = time()
# print("RandomizedSearchCV took %.2f seconds for %d candidates"
#       " parameter settings." % ((time() - start), n_iter_search))
# print("Best parameters set found on development set:")
# print()
# print(random_search.best_params_)

# run grid search
# grid_search = GridSearchCV(clf, param_grid=param_grid,n_jobs=-1,verbose=1,cv=10,scoring ='accuracy')
# start = time()
#
# # ZGit the grid with data
# grid_search.fit(X_train, y_train)
# report(grid_search.cv_results_)
# print("Best parameters set found on development set:")
# print()
# print(grid_search.best_params_)
# print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
#       % (time() - start, len(grid_search.cv_results_['params'])))

Save_Classifier(clf_en,file_name='DecisionTree_entropy')
Save_Classifier(clf_gini,file_name='DecisionTree_gini')



