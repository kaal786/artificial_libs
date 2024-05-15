import numpy as np
import pandas as pd

from models.linear import LinearRegression
from models.ensemble import RandomForestClassfier,RandomForestRegressor,BaggingRegressor,DecisionTreeRegressor
import sys
from math import ceil
from utils.plots import distribution,correlation
from utils.metrics import mae,mse
#sys.path.append("..")
def res():
		#from temprunner import LinearRegression
	datapath='/home/kali/Desktop/Eval/machine_learning/data/regression/'
	df=pd.read_csv(datapath+'sample_doublevariant.csv')
	features=df.columns
	X=df[['X','temp']].values
	y=df.y.values
	# #X=X.reshape(-1,1)
	# print(X)
	print('running')

	model=BaggingRegressor()
	model.fit(X,y)
	print('mse',mse(model.predict(X),y))
	print('mae',mae(model.predict(X),y))


res()