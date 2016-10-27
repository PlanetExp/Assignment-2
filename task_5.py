# Support Vector machine

from Ults import *
from task_1 import  *
import sklearn as sk
import numpy as np
import  matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import  GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import  scale
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import  validation_curve

data = DataProcessing()

# Get the transformed data
X_train_scale,X_test_scale = data.standard_scaler()

clf_ = SVC(kernel='rbf')

# Choose cross-validation iterator
from sklearn.model_selection import ShuffleSplit
cv = ShuffleSplit(n_splits=10,test_size=0.2,random_state=0)

# Set up C and gamma
Cs = [0.1, 1,10,100,1000]
Gammas = np.logspace(-6,-1,10)

param_grid = dict(svm__C = Cs,
                svm__gammas = Gammas)

grid_seach(clf_,param_grid,cv,X_train_scale,data.y_train,X_test_scale,data.y_test)



"""Plot the score base on C and gamma"""
# scores = [x[1] for x in gs.cv_results_['mean_test_score']]
# scores_std = [x[1] for x in gs.cv_results_['std_test_score']]
# scores = np.array(scores).reshape(len(Cs),len(Gammas))
# scores_std = np.array(scores_std).reshape(len(Cs),len(Gammas))

# for ind,i in enumerate(Cs):
#     plt.subplot(2,1,1)
#     plt.plot(Gammas,scores[ind],'-o',label = 'C: ' +str(i))
#     plt.subplot(2,1,2)
#     plt.plot(Gammas, scores_std[ind],'-o', label='C: ' + str(i))
# plt.legend()
# plt.xlabel('Gamma')
# plt.ylabel('Mean score')
# plt.show()

# # Debug algorithm with learning curve
# from sklearn.model_selection import learning_curve
# title = 'Learning Curves (SVM, Linear kernel, $\gamma=%.6f$)' % gs.cv_results_['param_gamma']
# estimator = SVC(kernel='linear',gamma=gs.cv_results_['param_gamma'])
# plot_learning_curve(estimator,title,X_train,y_train,cv=cv)
# plt.show()
# gs.score(X_test,y_test)










