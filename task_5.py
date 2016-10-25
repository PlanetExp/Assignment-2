# Support Vector machine

from Ults import *
from task_1 import  *
import sklearn as sk
import numpy as np
import  matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import  scale
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import  validation_curve

data = DataProcessing()

# Get the transformed data
X_train_scale,X_test_scale = data.standard_scaler()

# Gaussian Naive-Bayes with no calibration
clf = SVC(kernel='linear',C=1)



# # this dataset is way too high-dimensional. Better do PCA:
# pca = PCA(n_components=2)
#
# # Select feature
# selection = SelectKBest(k=1)
#
# # Build estimator from PCA and Univariate selection:
# combine_features = FeatureUnion([("pca",pca),("univ_select",selection)])
#
# # Use combine features to transform dataset:
# X_features = combine_features.fit(X_train,y_train)
#
# clf = SVC(kernel='linear')
#
#
# # Choose cross-validation iterator
# from sklearn.model_selection import ShuffleSplit
# cv = ShuffleSplit(n_splits=10,test_size=0.2,random_state=0)
#
# # Do grid search over k,n_components and C
# pipeline = Pipeline([(("features", combine_features),("clf",clf))])
#
# param_grid = dict(features__pca__n_components = [1,2,3],
#                   features__univ_select__k = [1,2],
#                   svm__C = [0.1,1,10],
#                   svm__gammas = np.logspace(-6,-1,10))
# gs = GridSearchCV(pipeline,cv=cv,param_grid=param_grid, verbose=10)
# gs.fit(X_train,y_train)
#
# print(gs.best_estimator_)
#
#
# # Debug algorithm with learning curve
# from sklearn.model_selection import learning_curve
# title = 'Learning Curves (SVM, Linear kernel, $\gamma=%.6f$)' % gs.cv_results_['param_gamma']
# estimator = SVC(kernel='linear',gamma=gs.cv_results_['param_gamma'])
# plot_learning_curve(estimator,title,X_train,y_train,cv=cv)
# plt.show()
# gs.score(X_test,y_test)









