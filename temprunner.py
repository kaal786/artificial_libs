
from models.linear import LinearRegression
import pandas as pd
import sys
from math import ceil
from utils.plots import distribution,correlation
import numpy as np
#sys.path.append("..")
def res():
		#from temprunner import LinearRegression
	datapath='/home/kali/Desktop/Eval/machine_learning/data/regression/'
	df=pd.read_csv(datapath+'sample_doublevariant.csv')
	features=df.columns
		# X=df[['X','temp']].values
		# y=df.y.values
		# #X=X.reshape(-1,1)
		# print(X)
		# lr=LinearRegression()
		# lr.fit(X,y)
		# print('running')
	print(correlation(df))
	print(distribution(df,features=features))

res()