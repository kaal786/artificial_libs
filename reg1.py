
# external libs for use 
import pandas as pd
import numpy as np



class reg1():
    def __init__(self,data,index=None,**kwargs):
        self.data = data
        self.index=index
        self.df=pd.read_csv(self.data,index_col=self.index)

    def basics(self):
        print(f"\n 1st 10 values : \n\n {self.df.head()} \n\n")
        print(f"\n last 10 values : \n\n {self.df.tail()} \n\n")
        print(f"\n Describe : \n\n {self.df.describe()}\n\n")
        self.df.info()

        
    def features_insights(self):
        for i in self.df.columns:
            print(f"{i}:{len(self.df[i].unique())}")
            if (len(self.df[i].unique()) < 10) :
                print(f"{self.df[i].unique()}\n")







if __name__ == '__main__' :
    bs=reg1('train.csv','row_id')
    bs.features_insights()