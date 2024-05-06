from random import seed
from random import randrange
from csv import reader
from math import sqrt

import os

from utils.stats import mean,variance,covariance


class LinearRegression :
    def __init__(self,lr=0.1):
        self.lr=lr,
        self.coeffiecients=[]

    #for single variable regression
    # def fit(self,X,y):
    #     x_mean, y_mean = mean(X), mean(y)
    #     b1 = covariance(X, x_mean, y, y_mean) / variance(X, x_mean)
    #     b0 = y_mean - b1 * x_mean
    #     print(self.coeffiecients)
    #     self.coeffiecients=[b0, b1]
    #     print(self.coeffiecients)
        

    # def predict(self,testX):
    #     predictions = list()
    #     print(self.coeffiecients)
    #     b0, b1 = self.coeffiecients
    #     for row in testX:
    #         yhat = b0 + b1 * row
    #         predictions.append(yhat)
    #     return predictions

    def fit(self,X,y):
        if len(X.shape) < 2:
            print('Error : please reshape the input data in (-1,1) format')

        self.coeffiecients = [0.0 for i in range(2)]
        self.coeffiecients = [0.0 for i in range(len(X[0]))]

        for row in range(len(X)):
            yhat = self.predict(X[row])
            error = yhat - y[row]
            self.coeffiecients[0] = self.coeffiecients[0] - self.lr * error
            for i in range(len(row)-1):
                self.coeffiecients[i + 1] = self.coeffiecients[i + 1] - self.lr * error * row[i]


    def predict(self,testX) :
        predictions = list()
        for row in testX :
            yhat=self.coeffiecients[0]
            for i in range(len(row)):
                yhat += coefficients[i + 1] * row[i]
            predictions.append(yhat)
        return predictions