
# external libs for use 
import pandas as pd
import numpy as np





class regtask():
    def __init__(self,df,**kwargs):
   
        self.df=df

    def basics(self):
        print(f"\n 1st 10 values : \n\n {self.df.head()} \n\n")
        print(f"\n last 10 values : \n\n {self.df.tail()} \n\n")
        print(f"\n Describe : \n\n {self.df.describe()}\n\n")
        self.df.info()

        
    def features_insights(self):
        df1=pd.DataFrame()
        print("\n\n")
        df1['featues']=self.df.columns
        df1['count']=list(map(lambda x : self.df[x].count(),self.df.columns))
        df1['null_values']=list(map(lambda x :self.df[x].isnull().sum(),self.df.columns))
        df1['uniques']=list(map(lambda x :len(self.df[x].unique()),self.df.columns))
        df1['values']=list(map(lambda x : self.df[x].unique() if len(self.df[x].unique()) < 10 else 'large spread', self.df.columns))
        df1['value_counts']=list(map(lambda x : self.df[x].value_counts().values  if len(self.df[x].unique()) < 10 else 'large spread',self.df.columns ))
        print(df1)

    def feature_plot(self):
        pass



# if __name__ == '__main__' :
    # df=pd.read_csv('~/Downloads/datasets/data/train.csv')
    # bs=regtask(df)
    # bs.features_insights()
    # pass
