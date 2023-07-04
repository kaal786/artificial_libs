
from models._linear import LinearRegression
import pandas as pd
import sys
sys.path.append("..")
def res():
		from temprunner import LinearRegression
		datapath='/home/kali/Desktop/Eval/machine_learning/data/regression/'
		df=pd.read_csv(datapath+'sample_singlereg.csv')
		X=df.X.values
		y=df.y.values
		lr=LinearRegression()
		lr.fit(X,y)
		print('running')

res()