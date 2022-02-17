
# external libs for use 
from abc import abstractmethod
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from abc import ABC, abstractmethod

class cucore(ABC):
    def __init__(self,df,value,**kwargs):
        self.df=df
        self.value=value
    @classmethod
    def features(self,numeric=False,object=False,datetime=False):
        df1=pd.DataFrame()
        self.features=np.array(self.df.columns)
        if numeric :
            s = (self.df.dtypes != 'object'  )
            self.features = np.array(s[s].index) 
        if object :
            s = (self.df.dtypes == 'object' ) 
            self.features = np.array(s[s].index) 
        if datetime :
            s = (self.df.dtypes == 'datetime64' ) 
            self.features = np.array(s[s].index) 

        df1['featues']=self.features  
        df1['count']=list(map(lambda x : self.df[x].count(),self.features))
        df1['null_values']=list(map(lambda x :self.df[x].isnull().sum(),self.features))
        df1['uniques']=list(map(lambda x :len(self.df[x].unique()),self.features))
        df1['values']=list(map(lambda x : self.df[x].unique() if len(self.df[x].unique()) < 10 else 'large spread', self.features))
        df1['value_counts']=list(map(lambda x : self.df[x].value_counts().values  if len(self.df[x].unique()) < 10 else 'large spread',self.features ))
       
        return df1


    def __add__(self,xyz):
        return self.value + xyz.value


    def features_plot(self,plot_type):
        if plot_type=='correlation' :
            plt.subplots(figsize = (30, 20))
            sns.heatmap(self.df.corr(),annot = True,center = 0)




    #class method : it will give the access of class to the instance
    @classmethod
    def method1(cls,data):
        cls.df= data


    
    #static method : To Access the class method directly without creating instance.
    @staticmethod
    def method2(data2):
        print('you data is ' , data2)



    #abstract method : order the child class to implements this method
    @abstractmethod
    def method3(self):
        print(self.value*self.value)






class classification(cucore):
    def __init__(self, df, **kwargs):
        super().__init__(df, **kwargs)


    def features_plot(self,plot_type):
        if plot_type=='confusion':
            pass


class regression(cucore):
    def __init__(self, df, value,**kwargs):
        super().__init__(df,value ,**kwargs)

    def method3(self):
        print(self.value*self.value)



if __name__ == '__main__' :
    df=pd.read_csv('https://raw.githubusercontent.com/pycaret/datasets/main/data/common/automobile.csv')
    bs=regression(df,1).method3()
    # print(bs.features(object=True))
    # bs.features_plot('correlation')
    
    # a1=cucore(df,12) + cucore(df,19)
    # print(a1)

    
