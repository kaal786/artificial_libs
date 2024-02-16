from math import sqrt


def mean(values):
    return sum(values) / float(len(values))


def variance(values, mean):
    """
    To calculate variance of feature
    """
    return sum([(x-mean)**2 for x in values])

def covariance(x, mean_x, y, mean_y):
    """ 
        To calculate covariance between two features
        x : feature1
        mean_x : mean of x
        y :  feature2
        mean_y : mean of y
    """
    covar = 0.0
    for i in range(len(x)):
        covar += (x[i] - mean_x) * (y[i] - mean_y)
    return covar


def euclidean_distance(row1, row2):
    """ X1 : List , start point
        X2 : List , end point 
    """
    distance = 0.0
    for i in range(len(row1)):
        distance += (row1[i] - row2[i])**2
    return sqrt(distance)