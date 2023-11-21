# import the necessary libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly as py
# normalize the dataset
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
import time
import psutil
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report



# conserve only percentage of data for each label but not the time step!!!
def downsample_data(data, label_column, target_percentage):
    """
    Downsample the data while conserving the same percentage of each label.

    Parameters:
    - data: Pandas DataFrame containing the dataset.
    - label_column: Name of the column containing the labels.
    - target_percentage: Target percentage to conserve for each label.

    Returns:
    - downsampled_data: Pandas DataFrame with downsampled data.
    """
    # Calculate the target count for each label
    target_counts = (data[label_column].value_counts() * target_percentage).round().astype(int)

    # Downsample each label
    downsampled_data = pd.DataFrame()
    for label, count in target_counts.items():
        label_data = data[data[label_column] == label]
        downsampled_label = label_data.sample(count, replace=False)  # Change replace to True for sampling with replacement
        downsampled_data = pd.concat([downsampled_data, downsampled_label])

    # order the data by time
    downsampled_data = downsampled_data.sort_values(by=['Time'])

    return downsampled_data



# function to read the data
def load_data(path,downsample=False,downsample_percentage=0.01):
    """
    Load the data from a csv file.

    Parameters:
    - path: Path to the csv file.
    - downsample: Boolean to indicate if the data should be downsampled.

    Returns:
    - data: Pandas DataFrame containing the dataset.
    """
    # read the data
    data = pd.read_csv(path, encoding='utf-8')

    # remove the spaces from the column names
    data.columns = data.columns.str.replace(' ', '')

    # downsample the data
    if downsample:
        data = downsample_data(data, 'Label', downsample_percentage)

    return data
    

# function to normalize the data
def normalize_data(data,column_normalisation=[]):
    """
    Normalize the data.

    Parameters:
    - data: Pandas DataFrame containing the dataset.

    Returns:
    - data: Pandas DataFrame containing the normalized dataset.
    """
    # normalize the data
    if column_normalisation != []:
        scaler.fit(data[column_normalisation])
        data[column_normalisation] = scaler.transform(data[column_normalisation])
        
    return data

# function to handle the nan values
def handle_nan(data,options='mode'):
    """
    Handle the nan values in the data.

    Parameters:
    - data: Pandas DataFrame containing the dataset.
    - options: String containing the options for handling the nan values mode or mean.

    Returns:
    - data: Pandas DataFrame containing the dataset without nan values.
    """
    # handle the nan by filling it with the options
    if options == 'mode':
        data = data.fillna(data.mode().iloc[0])
    elif options == 'mean':
        data = data.fillna(data.mean())
    else:
        print('Please enter a valid option for handling the nan values.')
    
    return data


# preprocessing the data
def preprocessing(data,column_normalisation=[]):
    
    """
    Preprocess the data.

    Parameters:
    - data: Pandas DataFrame containing the dataset.

    Returns:
    - data: Pandas DataFrame containing the preprocessed dataset.
    """

    # handle the nan values
    data = handle_nan(data)

    # normalize the data
    data = normalize_data(data,column_normalisation)
        
    return data


# function to apply one hot encoding to the data for the specified columns
# this function is used to apply one hot encoding to test our classifiers ( especially svm )
def one_hot_encoding(data,columns):
    """
    Apply one hot encoding to the data for the specified columns.

    Parameters:
    - data: Pandas DataFrame containing the dataset.
    - columns: List of columns to apply one hot encoding to.

    Returns:
    - data: Pandas DataFrame containing the dataset with one hot encoding.
    """
    # apply one hot encoding to the columns
    data = pd.get_dummies(data, columns=columns)
    
    return data


##################################################################

## classifier functions

# knn

class KNNClassifierWrapper:
    def __init__(self, n_neighbors=5, metric='minkowski', p=2):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.p = p
        self.classifier = KNeighborsClassifier(n_neighbors=self.n_neighbors, metric=self.metric, p=self.p)
    
    def train(self, X_train, y_train, show_time=False, show_memory=False):
        self.start_time = time.time()
        self.process = psutil.Process()

        # train the classifier
        self.classifier.fit(X_train, y_train)

        self.fit_time = time.time() - self.start_time
        if show_time:
            print('Training time: ', self.fit_time)
        
        if show_memory:
            print(f"RAM Usage after Fit: {self.process.memory_info().rss / (1024 ** 2):.2f} MB")

    def predict(self, X_test, show_time=False, show_memory=False):
        # predict the labels
        self.pred_start_time = time.time()
        self.y_pred = self.classifier.predict(X_test)
        self.predict_time = time.time() - self.start_time - self.fit_time

        if show_time:
            print('Prediction time: ', self.predict_time)

        if show_memory:
            print(f"RAM Usage after Predict: {self.process.memory_info().rss / (1024 ** 2):.2f} MB")
    
    def evaluate(self, y_test):
        # calculate the accuracy
        accuracy = accuracy_score(y_test, self.y_pred)

        # confusion matrix
        cm = confusion_matrix(y_test, self.y_pred)

        # classification report
        class_report = classification_report(y_test, self.y_pred)

        return accuracy, cm, class_report
    







# random forest

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import time
import psutil

class RandomForestClassifierWrapper:
    def __init__(self, n_estimators=10, criterion='entropy', random_state=0):
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.random_state = random_state
        self.classifier = RandomForestClassifier(n_estimators=self.n_estimators, criterion=self.criterion, random_state=self.random_state)

    def train(self, X_train, y_train, show_time=False, show_memory=False):
        self.start_time = time.time()
        self.process = psutil.Process()

        # train the classifier
        self.classifier.fit(X_train, y_train)

        self.fit_time = time.time() - self.start_time
        if show_time:
            print('Training time: ', self.fit_time)
        
        if show_memory:
            print(f"RAM Usage after Fit: {self.process.memory_info().rss / (1024 ** 2):.2f} MB")
    
    def predict(self, X_test, show_time=False, show_memory=False):
        # predict the labels
        self.pred_start_time = time.time()
        self.y_pred = self.classifier.predict(X_test)
        self.predict_time = time.time() - self.start_time - self.fit_time

        if show_time:
            print('Prediction time: ', self.predict_time)

        if show_memory:
            print(f"RAM Usage after Predict: {self.process.memory_info().rss / (1024 ** 2):.2f} MB")
        
    def evaluate(self, y_test):
        # calculate the accuracy
        accuracy = accuracy_score(y_test, self.y_pred)

        # confusion matrix
        cm = confusion_matrix(y_test, self.y_pred)

        # classification report
        class_report = classification_report(y_test, self.y_pred)

        return accuracy, cm, class_report


# svm this one is for binary classification used with label_n



class SVMClassifierWrapper:
    def __init__(self, kernel='rbf', random_state=0):
        self.kernel = kernel
        self.random_state = random_state
        self.classifier = SVC(kernel=self.kernel, random_state=self.random_state)

    def train(self, X_train, y_train, show_time=False, show_memory=False):
        self.start_time = time.time()
        self.process = psutil.Process()

        # train the classifier
        self.classifier.fit(X_train, y_train)

        self.fit_time = time.time() - self.start_time
        if show_time:
            print('Training time: ', self.fit_time)
        
        if show_memory:
            print(f"RAM Usage after Fit: {self.process.memory_info().rss / (1024 ** 2):.2f} MB")

    def predict(self, X_test, show_time=False, show_memory=False):
        # predict the labels
        self.pred_start_time = time.time()
        self.y_pred = self.classifier.predict(X_test)
        self.predict_time = time.time() - self.start_time - self.fit_time

        if show_time:
            print('Prediction time: ', self.predict_time)

        if show_memory:
            print(f"RAM Usage after Predict: {self.process.memory_info().rss / (1024 ** 2):.2f} MB")
    
    def evaluate(self, y_test):
        # calculate the accuracy
        accuracy = accuracy_score(y_test, self.y_pred)

        # confusion matrix
        cm = confusion_matrix(y_test, self.y_pred)

        # classification report
        class_report = classification_report(y_test, self.y_pred)

        return accuracy, cm, class_report
    





# xgboost

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

class XGBoostClassifierWrapper:
    def __init__(self, n_estimators=10, random_state=0):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.classifier = XGBClassifier(n_estimators=self.n_estimators, random_state=self.random_state)

    def train(self, X_train, y_train, show_time=False, show_memory=False):
        self.start_time = time.time()
        self.process = psutil.Process()

        # train the classifier
        self.classifier.fit(X_train, y_train)

        self.fit_time = time.time() - self.start_time
        if show_time:
            print('Training time: ', self.fit_time)
        
        if show_memory:
            print(f"RAM Usage after Fit: {self.process.memory_info().rss / (1024 ** 2):.2f} MB")

    def predict(self, X_test, show_time=False, show_memory=False):
        # predict the labels
        self.pred_start_time = time.time()
        self.y_pred = self.classifier.predict(X_test)
        self.predict_time = time.time() - self.start_time - self.fit_time

        if show_time:
            print('Prediction time: ', self.predict_time)

        if show_memory:
            print(f"RAM Usage after Predict: {self.process.memory_info().rss / (1024 ** 2):.2f} MB")
    
    def evaluate(self, y_test):
        # calculate the accuracy
        accuracy = accuracy_score(y_test, self.y_pred)

        # confusion matrix
        cm = confusion_matrix(y_test, self.y_pred)

        # classification report
        class_report = classification_report(y_test, self.y_pred)

        return accuracy, cm, class_report


# decision tree

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

class DecisionTreeClassifierWrapper:
    def __init__(self, criterion='entropy', random_state=0):
        self.criterion = criterion
        self.random_state = random_state
        self.classifier = DecisionTreeClassifier(criterion=self.criterion, random_state=self.random_state)

    def train(self, X_train, y_train, show_time=False, show_memory=False):
        self.start_time = time.time()
        self.process = psutil.Process()

        # train the classifier
        self.classifier.fit(X_train, y_train)

        self.fit_time = time.time() - self.start_time
        if show_time:
            print('Training time: ', self.fit_time)
        
        if show_memory:
            print(f"RAM Usage after Fit: {self.process.memory_info().rss / (1024 ** 2):.2f} MB")

    def predict(self, X_test, show_time=False, show_memory=False):
        # predict the labels
        self.pred_start_time = time.time()
        self.y_pred = self.classifier.predict(X_test)
        self.predict_time = time.time() - self.start_time - self.fit_time

        if show_time:
            print('Prediction time: ', self.predict_time)

        if show_memory:
            print(f"RAM Usage after Predict: {self.process.memory_info().rss / (1024 ** 2):.2f} MB")
        
    def evaluate(self, y_test):
        # calculate the accuracy
        accuracy = accuracy_score(y_test, self.y_pred)

        # confusion matrix
        cm = confusion_matrix(y_test, self.y_pred)

        # classification report
        class_report = classification_report(y_test, self.y_pred)

        return accuracy, cm, class_report
    

# naive bayes

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

class NaiveBayesClassifierWrapper:
    def __init__(self):
        self.classifier = GaussianNB()

    def train(self, X_train, y_train, show_time=False, show_memory=False):
        self.start_time = time.time()
        self.process = psutil.Process()

        # train the classifier
        self.classifier.fit(X_train, y_train)

        self.fit_time = time.time() - self.start_time
        if show_time:
            print('Training time: ', self.fit_time)
        
        if show_memory:
            print(f"RAM Usage after Fit: {self.process.memory_info().rss / (1024 ** 2):.2f} MB")

    def predict(self, X_test, show_time=False, show_memory=False):
        # predict the labels
        self.pred_start_time = time.time()
        self.y_pred = self.classifier.predict(X_test)
        self.predict_time = time.time() - self.start_time - self.fit_time

        if show_time:
            print('Prediction time: ', self.predict_time)

        if show_memory:
            print(f"RAM Usage after Predict: {self.process.memory_info().rss / (1024 ** 2):.2f} MB")
        
    def evaluate(self, y_test):
        # calculate the accuracy
        accuracy = accuracy_score(y_test, self.y_pred)

        # confusion matrix
        cm = confusion_matrix(y_test, self.y_pred)

        # classification report
        class_report = classification_report(y_test, self.y_pred)

        return accuracy, cm, class_report
    

# mlp

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

class MLPClassifierWrapper:
    def __init__(self, hidden_layer_sizes=(100,), activation='relu', solver='adam', alpha=0.0001, batch_size='auto', random_state=0):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.batch_size = batch_size
        self.random_state = random_state
        self.classifier = MLPClassifier(hidden_layer_sizes=self.hidden_layer_sizes, activation=self.activation, solver=self.solver, alpha=self.alpha, batch_size=self.batch_size, random_state=self.random_state)

    def train(self, X_train, y_train, show_time=False, show_memory=False):
        self.start_time = time.time()
        self.process = psutil.Process()

        # train the classifier
        self.classifier.fit(X_train, y_train)

        self.fit_time = time.time() - self.start_time
        if show_time:
            print('Training time: ', self.fit_time)
        
        if show_memory:
            print(f"RAM Usage after Fit: {self.process.memory_info().rss / (1024 ** 2):.2f} MB")

    def predict(self, X_test, show_time=False, show_memory=False):
        # predict the labels
        self.pred_start_time = time.time()
        self.y_pred = self.classifier.predict(X_test)
        self.predict_time = time.time() - self.start_time - self.fit_time

        if show_time:
            print('Prediction time: ', self.predict_time)

        if show_memory:
            print(f"RAM Usage after Predict: {self.process.memory_info().rss / (1024 ** 2):.2f} MB")
        
    def evaluate(self, y_test):
        # calculate the accuracy
        accuracy = accuracy_score(y_test, self.y_pred)

        # confusion matrix
        cm = confusion_matrix(y_test, self.y_pred)

        # classification report
        class_report = classification_report(y_test, self.y_pred)

        return accuracy, cm, class_report
    

# usage :
# mlpClassifier = MLPClassifierWrapper()
# mlpClassifier.train(X_train, y_train_encoded, show_time=True, show_memory=True)
# mlpClassifier.predict(X_test, show_time=True, show_memory=True)
# accuracy, cm, class_report = mlpClassifier.evaluate(y_test_encoded)
# 
# print('Accuracy: ', accuracy)
# print('Confusion Matrix: \n', cm)
# print('Classification Report: \n', class_report)