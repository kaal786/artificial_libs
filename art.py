# external libs for use 
from abc import abstractmethod
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
# from tensorflow.keras.utils import to_categorical
from keras.utils.np_utils import to_categorical


class common():
    def __init__(self,df,**kwargs):
        self.df=df
    def features(self,num_col=False,obj_col=False,dt_col=False):
        ddf=self.df
        df1=pd.DataFrame()
        self.features=np.array(ddf.columns.tolist())
        if num_col :
            s = (self.df.dtypes != 'object'  )
            self.features = np.array(s[s].index) 
        if obj_col :
            s = (self.df.dtypes == 'object' ) 
            self.features = np.array(s[s].index) 
        if dt_col :
            s = (self.df.dtypes == 'datetime64' ) 
            self.features = np.array(s[s].index) 

        df1['featues']=self.features  
        df1['count']=list(map(lambda x : self.df[x].count(),self.features))
        df1['null_values']=list(map(lambda x :self.df[x].isnull().sum(),self.features))
        df1['uniques']=list(map(lambda x :len(self.df[x].unique()),self.features))
        df1['values']=list(map(lambda x : self.df[x].unique() if len(self.df[x].unique()) < 10 else 'large spread', self.features))
        df1['value_counts']=list(map(lambda x : self.df[x].value_counts().values  if len(self.df[x].unique()) < 10 else 'large spread',self.features ))
        return df1
        

    
    def data_process(self,func=None,target='target'):
        if func!=None :
            self.df=func(self.df)

        X=self.df.drop(target,axis=1)
        y=self.df[target]
      
        std=StandardScaler()
        X=std.fit_transform(np.array(X))

        y=to_categorical(y,2)
        print(X.shape,y.shape)  

        from sklearn.model_selection import train_test_split
        train_X,val_X,train_y,val_y=train_test_split(X,y,test_size=0.2)
        for i in [train_X,val_X,train_y,val_y]:
            print(i.shape)

        return train_X,val_X,train_y,val_y


    def features_plot(self,plot_type):
        if plot_type=='correlation' :
            plt.subplots(figsize = (16, 9))
            sns.heatmap(self.df.corr(),annot = True,center = 0)

        



class classification(common):
    def __init__(self, df, **kwargs):
        super().__init__(df, **kwargs)




    def features_plot(self,plot_type):
        if plot_type=='confusion':
            pass


class regression(common):
    def __init__(self, df, value,**kwargs):
        super().__init__(df,value ,**kwargs)





if __name__=='__main__':
    print("Hello")
