# Nearest Neighbour
from sklearn.neighbors import  KNeighborsClassifier
from sklearn.model_selection import  ShuffleSplit
from sklearn import metrics
from sklearn import datasets
import cPickle
import numpy as np
from Ults import *
from task_1 import *


filename = "KNeighborsClassifier.pkl"

# Get the data
data = DataProcessing()

# Get the transformed data
X_train_scale,X_test_scale = data.standard_scaler()


for algorithms in ['auto', 'kd_tree', 'brute']:
    print "*************** %s *****************\n" %algorithms
    clf = KNeighborsClassifier(n_neighbors=15,algorithm=algorithms,n_jobs=-1)
    train_and_evaluate(clf, X_train_scale, data.y_train, X_test_scale, data.y_test)
    Save_Classifier(clf, file_name=algorithms)


#Save_Classifier(clf,file_name=filename)