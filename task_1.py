from __future__ import division

import pickle
from sklearn.model_selection import train_test_split,cross_val_score,KFold, StratifiedShuffleSplit
from sklearn import  svm
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.externals import joblib
from sklearn.preprocessing import normalize
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import scipy.stats.stats as st
from sklearn.base import TransformerMixin,BaseEstimator
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_array,as_float_array
from scipy import linalg
from sklearn.decomposition import  PCA

""" Import data set"""
with open('forest_data.pickle', 'rb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    data = pickle.load(f)
    target = pickle.load(f)

""" Data statistics"""

# Shape and type
print(data.shape, data.dtype)
print(target.shape, target.dtype)


class DataProcessing():
    # Test size use to split the data
    test_size = 0.15

    # Random seed
    seed = 42;

    def __init__(self):
        # Let us shuffle the data set
        #np.random.shuffle()
        self.X_train,self.X_test,self.y_train,self.y_test = train_test_split(data,target,test_size=self.test_size,random_state=self.seed)


    def shuffle_dataset(self):

        # Shuffle the dataset
        np.random.shuffle(input_dataset)

        # Prepare a stratisfied train and test split
        train_size = 0.85
        test_size = 1 - train_size

        # For ease we merge the data and target
        input_dataset = np.column_stack([data, target])

        stratified_split = StratifiedShuffleSplit(input_dataset[:,-1],test_size=test_size,n_splits=1)

        for train_indx, test_indx in stratified_split:
            X_train = input_dataset[train_indx,:-1]
            y_train = input_dataset[train_indx,-1]
            X_test = input_dataset[test_indx,:-1]
            y_test = input_dataset[test_indx,-1]

        return  X_train,y_train,X_test,y_test

    def check_missing_value(self):
        """ Check missing value of data set
            Return: True if no value missing
                    False otherwise
        """
        imp = preprocessing.Imputer(missing_values='NaN',strategy='mean',axis=0,copy=False)
        x = imp.fit_transform(data)
        return (data==x).all()

    def plot_bar(self,labels,value,subplot,title,xLabel,yLabel,save_png = False):

        # You typically want your plot to be ~1.33x wider than tall
        plt.figure(figsize=(12,9))

        # Remove the plot frame lines
        ax = plt.subplot(subplot)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Ensure that the axis ticks show up on the bottom and left of the plot
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()

        # Set x and y label
        plt.xlabel(xLabel)
        plt.ylabel(yLabel)

        # Plot the classes distribution
        ax.bar(labels,value,facecolor = "#3F5D7D")

        # Add text on top of the bar
        for x, y in zip(labels, value):
            plt.text(x + 0.4, y + 0.05, '%s' % y, ha='center', va='bottom')

        # Save the plot to png file
        if(save_png):
            plt.savefig("%s.png"%title, bbox_inches="tight");
        plt.show()

    def plot_features(self):
        """This function use to plot the feature distribution of data set"""
        unique, counts = np.unique(target, return_counts=True)
        self.plot_bar(unique,counts,111,'Class Distribution','Classes','Count',True)

    def plot_value_subfigures(self):
        """ This function use to plot the value distribution of each feature"""

        """List of features
            Since Widerness_Area and Soil_Types are binary data
            We will not plot them
        """
        feats = {'Elevation':0,
                 'Aspect':1,
                 'Slope':2,
                 'Horizontal_Distance_To_Hydrology':3,
                 'Vertical_Distance_To_Hydrology':4,
                 'Horizontal_Distance_To_Roadways':5,
                 'Hillshade_9am':6,
                 'Hillshade_Noon':7,
                 'Hillshade_3pm':8,
                 'Horizontal_Distance_To_Fire_Points':9,
                 }

        # Set figures sie
        plt.figure(figsize=(20,10))

        # Plot each feature figure
        for index,(name,positon) in enumerate (feats.items()):
            # Due to wide range of feature value. We select bin = 10
            n_bin = 10;

            # Get the unique value of current column
            unique = np.unique(data[:, index])
            x = np.arange(len(unique))

            ax = plt.subplot(3,4,index)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.get_xaxis().tick_bottom()
            ax.get_yaxis().tick_left()
            plt.xticks()
            plt.yticks(range(5000, 30001, 5000))
            ax.hist(unique, bins=n_bin, facecolor="#3F5D7D", edgecolor='white')
            plt.title("%s" % name)


        plt.subplots_adjust(left =.02, right =.98,top = .95, bottom = .05)
        plt.show()

    def standard_scaler(self):
        """ Scaling feature using Standard scaler
            Transform X_train and X_test then return them
        """
        scaler = preprocessing.StandardScaler()
        X_train_scale = scaler.fit_transform(self.X_train)
        X_test_scale = scaler.transform(self.X_test)
        return X_train_scale, X_test_scale

    def normalize(self):
        """ Scaling feature using Normalize
            Transform X_train and X_test then return them
        """
        normalizer = preprocessing.Normalizer(norm='l2')
        X_train_scale = normalizer.fit_transform(self.X_train)
        X_test_scale = normalizer.transform(self.X_test)
        return X_train_scale, X_test_scale

    def min_max_scaler(self):
        """ Scaling feature using min max scaler
            Transform X_train and X_test then return them
        """
        min_max_scaler = preprocessing.MinMaxScaler()
        X_train_scale = min_max_scaler.fit_transform(self.X_train)
        X_test_scale = min_max_scaler.transform(self.X_test)
        return X_train_scale, X_test_scale

    def maxAbs_scaler(self):
        """ Scaling feature using Max_Abs_scaler
            Transform X_train and X_test then return them
        """
        maxAbs = preprocessing.MaxAbsScaler()
        X_train_scale = maxAbs.fit_transform(self.X_train)
        X_test_scale = maxAbs.transform(self.X_test)
        return X_train_scale, X_test_scale

def Save_Classifier(clf,file_name):
    """ This function use to save the classifier model into file
        clf: classifier model
        filename: is a string which will be the name of saved file
    """
    joblib.dump(clf,filename=file_name)

def Load_Classifier(file_name):
    """ This function use to save the classifier model
        file_name: is a string of model want to load
    """
    return joblib.load(file_name)

a = DataProcessing()

# Plot the class distribution
#a.plot_features()

# Plot value distribution of each features
#a.plot_value_subfigures()