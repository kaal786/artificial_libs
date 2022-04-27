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
# from tensorflow.keras.utils import to_categorical
from keras.utils.np_utils import to_categorical
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error,mean_absolute_error
#models
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from math import ceil
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
    def __init__(self,df,target='target',ptype='clf',mtype='simple',method=None,metric='mse',**kwargs):
        print("\n\n=>Attributes  [df,target='target',ptype='clf',mtype='simple',method='zscore',metric='mse'] \n\n=>Methods \n1.features() \n2.feature_plot(ptype='corelation') \n3.data=data_process()  \n4.train(data=data,model=model)\n\n")
        self.df=df
        self.target=target
        self.mtype=mtype
        self.method=method
        self.metric=metric
        self.ptype=ptype
    def features(self,more_col=False):
        """Basic overviews of features
        for specific features : num_col,obj_col,dt_col"""
        ddf=self.df
        features_df=pd.DataFrame()
        self.features=np.array(ddf.columns.tolist())
        if more_col :
            s = (self.df.dtypes != 'object'  )
            self.features = np.array(s[s].index)
            features_df['skewness']=list(map(lambda x : self.df[x].skew(),self.features))
            features_df['kurtos']=list(map(lambda x : self.df[x].kurt(),self.features))

        features_df['features']=self.features  
        features_df['dtype']=list(map(lambda x : self.df[x].dtype,self.features))
        features_df['count']=list(map(lambda x : self.df[x].count(),self.features))
        features_df['null_values']=list(map(lambda x :self.df[x].isnull().sum(),self.features))
        features_df['uniques']=list(map(lambda x :len(self.df[x].unique()),self.features))
        features_df['values']=list(map(lambda x : self.df[x].unique() if len(self.df[x].unique()) < 10 else self.df[x].values[:4].tolist(), self.features))
        features_df['value_counts']=list(map(lambda x : self.df[x].value_counts().values  if len(self.df[x].unique()) < 10 else self.df[x].value_counts().values[:4].tolist(),self.features ))
        features_df.set_index('features',inplace=True)
        return features_df

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

   
    
    def data_process(self,data_prepare=None):
        """
        To process the data to feed model
        func = your function that process raw data to universel data
        target = default : 'target' ,you can provide your target value
        """
        if data_prepare!=None :
            self.df=self.data_prepare(self.df)
        else :
            self.df.drop((self.df.dtypes=='object')[self.df.dtypes =='object'].index.tolist(),axis=1,inplace=True)
            self.df=self.df.dropna(axis=0,inplace=True)   
        X=self.df.drop(self.target,axis=1)
        y=self.df[self.target]

        if self.method !=None:
            methods={'zscore':StandardScaler(),'minmax':MinMaxScaler()}
            nm=methods[self.method]
            print(nm)
            X=nm.fit_transform(np.array(X))
        if self.mtype=='deep' :
            y=to_categorical(y,len(self.df[self.target].unique()))
        return X,y
        
    def validate(self,X,y,type='holdout'):
        if type=='holdout':
            train_X,val_X,train_y,val_y=train_test_split(X,y,test_size=0.2,random_state=42)
            print(f"train set : {train_X.shape,train_y.shape} val set : {val_X.shape,val_y.shape}")
            return train_X,val_X,train_y,val_y
        if type=='kfold':
            pass


    def fitting(self,train_X,train_y):
        if self.mtype=='simple':
            model.fit(train_X,train_y) 
        if self.mtype=='deep':
            model.fit(train_X,train_y,epochs=1,batch_size=32,verbose=0)         
    
    def predicting(self,train_X):
        val_yhat=model.predict(val_X)
        train_yhat=model.predict(train_X)
    
    def scoring(self,actual,yhat):
        if self.ptype=='clf':
            if self.mtype=='simple':
                train_score=accuracy_score(actual,yhat)
            if self.mtype=='deep':
                train_score=accuracy_score(np.argmax(actual,axis=1),np.argmax(yhat,axis=1))
        if self.ptype=='reg':
            metrics={'mse':mean_squared_error,'mae':mean_absolute_error}
            score_=metrics[self.metric]
            if self.mtype=='simple':    
                train_score=score_(actual,yhat)
            if self.mtype=='deep':
                train_score=score_(actual,np.argmax(yhat,axis=1))
        print(f" score for {actual} : {train_score} ")

    
    
    
    


class classification(common):
    def __init__(self, df, **kwargs):
        print('what"s your class ?')
        super().__init__(df, **kwargs)
       
    def train(self,data,model=RandomForestClassifier()):
        """training basic model
        model=your custom model function
        mtype= defalut: simple
        simple : sklearn model
        deep : neural model"""
        X,y=self.data_process()
        train_X,val_X,train_y,val_y=self.validate()
        self.fitting()
        self.predicting()
              
        val_yhat=model.predict(val_X)
        train_yhat=model.predict(train_X)
        self.scoring(train_y,train_yhat)
        self.scoring(val_y,val_yhat)
        
        
class regression(common):
    def __init__(self, df, **kwargs):
        super().__init__(df,**kwargs)
    
    
  
    def train(self,data,model=RandomForestRegressor()):
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
        
        


class timeseries(common):
    def __init__(self,df,**kwargs):
        print('time is yours')
        super().__init__(df,**kwargs)
         
    def data_process(self):
        """
        To process the data to feed model
        func = your function that process raw data to universel data
        target = default : 'target' ,you can provide your target value
        """
        print('time is yours')
        if self.data_prepare!=None :
            self.df=self.data_prepare(self.df)
        else :
            self.df=self.numcol_prepare(self.df)
            
        X=self.df.drop(self.target,axis=1)
        y=self.df[self.target]

        if self.method !=None:
            methods={'zscore':StandardScaler(),'minmax':MinMaxScaler()}
            nm=methods[self.method]
            print(nm)
            X=nm.fit_transform(np.array(X))

        if self.mtype=='deep' :
            y=to_categorical(y,len(self.df[self.target].unique()))
        index_=ceil(X.shape[0]*0.8)
        train_X,val_X,train_y,val_y=X[:index_],X[index_],y[:index_],y[index_:]
        
        return train_X,val_X,train_y,val_y









class auto(classification,regression):
    def __init__(self,df,model=RandomForestClassifier(),**kwargs):
        print("Extra Attributes : ptype :['clf','reg','timeseries']")
        super().__init__(df ,**kwargs)
        problem={'clf':classification ,'reg': regression}
        
        data=self.data_process()
        problem[self.ptype].train(self,data=data,model=model)
    


if __name__=='__main__':
    print("Hello")
