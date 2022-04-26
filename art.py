# external libs for use 
from abc import abstractmethod
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import torch
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler,MinMaxScaler

from keras.utils.np_utils import to_categorical
from sklearn.metrics import accuracy_score

#models
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

#set seeds
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
torch.manual_seed(0)

class common():
    """Basic methods : 
    features,
    features_plot,
    data_process"""
    def __init__(self,df,data_prepare=None,target='target',mtype='simple',method='zscore',**kwargs):
        self.df=df
        self.data_prepare=data_prepare
        self.target=target
        self.mtype=mtype
        self.method=method
    
    def features(self,num_col=False,obj_col=False,dt_col=False):
        """Basic overviews of features
        for specific features : num_col,obj_col,dt_col"""
        ddf=self.df
        df1=pd.DataFrame()
        self.features=np.array(ddf.columns.tolist())
        if num_col :
            s = (self.df.dtypes != 'object'  )
            self.features = np.array(s[s].index)
            df1['skewness']=list(map(lambda x : self.df[x].skew(),self.features))
            df1['kurtos']=list(map(lambda x : self.df[x].kurt(),self.features))
        if obj_col :
            s = (self.df.dtypes == 'object' ) 
            self.features = np.array(s[s].index) 
        if dt_col :
            s = (self.df.dtypes == 'datetime64' ) 
            self.features = np.array(s[s].index) 
        df1['features']=self.features  
        df1['dtype']=list(map(lambda x : self.df[x].dtype,self.features))
        df1['count']=list(map(lambda x : self.df[x].count(),self.features))
        df1['null_values']=list(map(lambda x :self.df[x].isnull().sum(),self.features))
        df1['uniques']=list(map(lambda x :len(self.df[x].unique()),self.features))
        df1['values']=list(map(lambda x : self.df[x].unique() if len(self.df[x].unique()) < 10 else self.df[x].values[:4].tolist(), self.features))
        df1['value_counts']=list(map(lambda x : self.df[x].value_counts().values  if len(self.df[x].unique()) < 10 else self.df[x].value_counts().values[:4].tolist(),self.features ))
        df1.set_index('features',inplace=True)
        return df1


    def numcol_prepare(self,df):
        """Basic preparation of data :
        removing object features
        removing nan values"""
        df.drop((df.dtypes=='object')[df.dtypes =='object'].index.tolist(),axis=1,inplace=True)
        df.dropna(axis=0,inplace=True)
        return df
    
    def data_process(self):
        """
        To process the data to feed model
        func = your function that process raw data to universel data
        target = default : 'target' ,you can provide your target value
        """
        
        if self.data_prepare!=None :
            self.df=self.data_prepare(self.df)
        else :
            self.df=self.numcol_prepare(self.df)
            
        X=self.df.drop(self.target,axis=1)
        y=self.df[self.target]


        methods={'zscore':StandardScaler(),'minmax':MinMaxScaler()}
        nm=methods[self.method]
        print(nm)
        X=nm.fit_transform(np.array(X))

        if self.mtype=='deep' :
            y=to_categorical(y,len(self.df[self.target].unique()))
     
        
        train_X,val_X,train_y,val_y=train_test_split(X,y,test_size=0.2,random_state=42)
        print(f"train set : {train_X.shape,train_y.shape} val set : {val_X.shape,val_y.shape}")
        return train_X,val_X,train_y,val_y
    
    
    
    
    
    def features_plot(self,ptype='basic'):
        """ptype : default plot : basic || 
            plots : corelation ,cmatrix,"""
        fig=plt.figure(figsize=(16,9))
        ax=fig.gca()
        
        if ptype=='basic' :    
            self.df.hist(ax=ax,bins=10,)

        if ptype=='corelation' :
            sns.heatmap(self.df.corr(),ax=ax,cmap='Dark2')

        plt.show()



class classification(common):
    def __init__(self, df, **kwargs):
        super().__init__(df, **kwargs)
    
    
    
    
    
    def train(self,data,model=RandomForestClassifier()):
        """training basic model
            model=your custom model function
            mtype= defalut: simple
                    simple : sklearn model
                    deep : neural model"""
        
        train_X,val_X,train_y,val_y=data
        if self.mtype=='simple':
            model.fit(train_X,train_y) 
        
        if self.mtype=='deep':
            model.fit(train_X,train_y,validation_data=(val_X,val_y),epochs=1,batch_size=32,verbose=0)
        
        val_yhat=model.predict(val_X)
        train_yhat=model.predict(train_X)
        if self.mtype=='simple':
            train_score=accuracy_score(train_y,train_yhat)
            val_score=accuracy_score(val_y,val_yhat)
        
        if self.mtype=='deep':
            train_score=accuracy_score(np.argmax(train_y,axis=1),np.argmax(train_yhat,axis=1))
            val_score=accuracy_score(np.argmax(val_y,axis=1),np.argmax(val_yhat,axis=1))
            return print(f"train score : {train_score} val score : {val_score}")
        return print(f"train score : {train_score} val score : {val_score}")

class regression(common):
    def __init__(self, df, value,**kwargs):
        super().__init__(df,**kwargs)

class auto(classification):
    def __init__(self,df,model=RandomForestClassifier(),**kwargs):
        super().__init__(df ,**kwargs)
        data=self.data_process()
        return self.train(data,model)


if __name__=='__main__':
    print("Hello")
