import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.externals import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

import csv

class bikeshareTrainer:
    def __init__(self, traindf):
        self.df = traindf # this is from the train.csv file provided by kaggle

        # inputs to be used by predictor
        self.X = self.prepX()

        # outputs to be used by predictor
        # assumes that casual and registered users follow different models (to get the total, simply sum casual and registered)
        self.y_casual = self.df.values[:,-3]
        self.y_registered = self.df.values[:,-2]
        self.y_total = self.df.values[:,-1]
        
        self.casual_model , self.registered_model = self.train()
        
    def prepX(self):
        # Takes advantage of separate date variables
        # inputs: self (should already have a dataframe)
        # output: x2 --> should be in a format that can be used with scikit-learn's RF

        # Categorical data
        column_labels = {'hour':1, 'month':2, 'year':3, 'season':4, 'weather':7}
        cat_data = self.df.values[:, sorted(column_labels.values())]

        # Prepping categorical data for scikit-learn
        ohe = OneHotEncoder()
        ohe.fit(cat_data)
        cat_data_prepped = ohe.transform(cat_data)

        # Extract numerical data from df
        # Combine cat_data_prepped and numerical data to make a giant matrix to feed into the ML estimator
        numerical_data_labels = np.array([5, 6, 8, 9, 10, 11])
        numerical_data = self.df.values[:, numerical_data_labels]

        # Entire data matrix
        return np.column_stack((cat_data, numerical_data))

    def trainTestPredict(self):
        # train test splitting
        X_train, X_test, y_casual_train, y_casual_test, y_registered_train, y_registered_test, y_total_train, y_total_test = train_test_split(X, 
                                                                                                                                              self.y_casual, 
                                                                                                                                              self.y_registered, 
                                                                                                                                              self.y_total, 
                                                                                                                                              test_size = 0.3)

        # TODO: Should these stay local or be class variables or??
        # instantiate RF regressors for casual and registered bikers
        rf_casual = RandomForestRegressor(n_jobs=-1)
        rf_registered = RandomForestRegressor(n_jobs=-1)

        # fitting for the casual and registered bikers
        rf_casual.fit(X_train, y_casual_train)
        rf_registered.fit(X_train, y_registered_train)

        # predictions for the casual and registered bikers
        y_casual_pred = rf_casual.predict(X_test)
        y_registered_pred = rf_registered.predict(X_test)

        # total predictions is predicted # of casual bikers + predicted # of registered bikers 
        y_total_pred = y_casual_pred + y_registered_pred
        y_total_pred = [int(x) for x in y_total_pred]
        
        # printing MAE for now
        # TODO: put all of the y variables into a dictionary, save as a class variable
        # TODO: make additional class methods for visualizing performances/model comparisons
        print "MAE for bikes rented per hour by casual riders: {}".format(np.mean(np.abs(y_casual_pred - y_casual_test)))
        print "MAE for bikes rented per hour by registered riders: {}".format(np.mean(np.abs(y_registered_pred - y_registered_test)))
        print "MAE for bikes rented per hour by all riders: {}".format(np.mean(np.abs(y_total_pred - y_total_test)))

        return rf_casual, rf_registered

    def train(self):
        # instantiate RF regressors for casual and registered bikers
        model_casual = RandomForestRegressor(n_jobs=-1)
        model_registered = RandomForestRegressor(n_jobs=-1)

        # fitting for the casual and registered bikers
        model_casual.fit(self.X, self.y_casual)
        model_registered.fit(self.X, self.y_registered)

        return model_casual, model_registered
    
    def saveModels(self):
        joblib.dump(self.casual_model, 'estimators/casual_model', compress=3)
        joblib.dump(self.registered_model, 'estimators/registered_model', compress=3)

class bikeshareTester(bikeshareTrainer):
    def __init__(self, testdf):
        self.df = testdf # this testdf is the test.csv file provided by kaggle

        # inputs to be used by predictor
        self.X = self.prepX()

    def loadModels(self):
        # Loads the models from the /estimators folder ; we created these with the bikeshareTrainer
        self.casual_model = joblib.load('estimators/casual_model')
        self.registered_model = joblib.load('estimators/registered_model')

    def predict(self):
        # Runs the model on the test data!

        # predictions for the casual and registered bikers
        self.y_casual_pred = self.casual_model.predict(self.X)
        self.y_registered_pred = self.registered_model.predict(self.X)

        # total predictions is predicted # of casual bikers + predicted # of registered bikers 
        self.y_total_pred = self.y_casual_pred + self.y_registered_pred
        self.y_total_pred = [int(x) for x in self.y_total_pred]

    def prepForSubmit(self):
        with open("submission1.csv", 'w') as csvfile:
            # Writes results to a csv which can be submitted to kaggle
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['datetime','count'])
            rows = zip(self.df['datetime'].values, self.y_total_pred)
            csvwriter.writerows(rows)




