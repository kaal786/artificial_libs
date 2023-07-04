from random import seed
from random import randrange
from csv import reader
from math import sqrt
import sys
sys.path.append("..")
from ..utils._stats import mean,variance,covariance

class LinearRegression :
    def __init__(self,):

        self.coeffiecients=[0,0]

    def fit(self,X,y):
        x_mean, y_mean = mean(X), mean(y)
        b1 = covariance(X, x_mean, y, y_mean) / variance(X, x_mean)
        b0 = y_mean - b1 * x_mean
        print(self.coeffiecients)
        self.coeffiecients=[b0, b1]
        print(self.coeffiecients)
        

    def predict(self,testX):
        predictions = list()
        print(self.coeffiecients)
        b0, b1 = self.coeffiecients
        for row in testX:
            yhat = b0 + b1 * row
            predictions.append(yhat)
        return predictions
