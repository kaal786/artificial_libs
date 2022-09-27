# external libs for use 
from abc import abstractmethod
from logging import exception
from turtle import shape
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
plt.rcParams['figure.figsize']=(12,6)
plt.style.use('fivethirtyeight')

from math import ceil
import random 



import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error,mean_absolute_error
#models
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression


import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Dense ,Dropout
from tensorflow.keras.models import Model



#set seeds
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
torch.manual_seed(42)




class art():
    """Basic methods : 
    features,
    features_plot,
    data_process"""
    def __init__(self,df,target='target',ptype='clf',mtype='simple',method=None,metric='mse',validation=True,**kwargs):
        """
        =>Attributes  [df,target='target',ptype='clf',mtype='simple',method='zscore',metric='mse'] 
        =>Methods
        1.features() 
        2.feature_plot(ptype='corelation') 
        3.data=data_process()  
        4.data=validate(X,y,type='holdout')
        5.fitting(train_X,train_y)
        6.scoring(self,actual,yhat)
        """
        self.df=df
        self.target=target
        self.mtype=mtype
        self.method=method
        self.metric=metric
        self.ptype=ptype
        self.validation=validation
       

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
        features_df['values']=list(map(lambda x : self.df[x].unique() if len(self.df[x].unique()) < 20 else self.df[x].values[:4].tolist(), self.features))
        features_df['value_counts']=list(map(lambda x : self.df[x].value_counts().values  if len(self.df[x].unique()) < 20 else self.df[x].value_counts().values[:4].tolist(),self.features ))
        features_df.set_index('features',inplace=True)
        return features_df

    def features_plot(self,plotype='basic'):
        """ptype : default plot : basic || 
            plots : corelation ,cmatrix,features"""
        if plotype=='basic' : 
            try:   
                self.df.hist()
            except exception as e:
                print(e)
        if plotype=='corelation' :
            sns.heatmap(self.df.corr(),cmap='Dark2')
        if plotype=='features':
            fig = plt.figure(constrained_layout = True, figsize = (16,len(self.df.columns.tolist())*7 ),edgecolor='black',facecolor='grey',)
            grid = matplotlib.gridspec.GridSpec(ncols = 1, nrows = len(self.df.columns.tolist()), figure = fig)           
            for i,feature in enumerate(self.df.columns.tolist()):
                try :
                    locals()["ax"+str(i)]= fig.add_subplot(grid[i, 0])
                    locals()["ax"+str(i)].set_title(feature)
                    locals()["ax"+str(i)].plot(self.df[feature])  
                except :
                    continue   
    
    def data_process(self,data_prepare=None):
        """
        To process the data to feed model
        func = your function that process raw data to universel data
        target = default : 'target' ,you can provide your target value
        return features , label
        """
        if data_prepare!=None :
            self.df=data_prepare(self.df)
        else :
            self.df.drop((self.df.dtypes=='object')[self.df.dtypes =='object'].index.tolist(),axis=1,inplace=True)
            self.df=self.df.dropna(axis=0)   
        X=self.df.drop(self.target,axis=1)
        y=self.df[self.target]
        if self.method !=None:
            methods={'zscore':StandardScaler(),'minmax':MinMaxScaler()}
            nm=methods[self.method]
            print(nm)
            X=nm.fit_transform(np.array(X))
        if self.mtype=='deep' & self.ptype=='clf':
            y=to_categorical(y,len(self.df[self.target].unique()))
        return self.X,self.y
        
    def validate(self,X,y,type='holdout'):
        if type=='holdout':
            train_X,val_X,train_y,val_y=train_test_split(X,y,test_size=0.2,random_state=42)
            print(f"train set : {train_X.shape,train_y.shape} val set : {val_X.shape,val_y.shape}")
            return train_X,train_y,val_X,val_y
        if type=='kfold':
            pass
        if type=='groupKfold':
            pass

    def fitting(self,model,train_X,train_y):
        if self.mtype=='simple':
            model.fit(train_X,train_y) 
        if self.mtype=='deep':
            self.model.fit(train_X,train_y,epochs=1,batch_size=32,verbose=0)         

    def scoring(self,actual,yhat):
        if self.ptype=='clf':
            if self.mtype=='simple':
                train_score=accuracy_score(actual,yhat)
            if self.mtype=='deep':
                train_score=accuracy_score(np.argmax(actual,axis=1),np.argmax(yhat,axis=1))
            print(f" accuracy score : {train_score} ")
        if self.ptype=='reg':
            metrics={'mse':mean_squared_error,'mae':mean_absolute_error,'accuracy':accuracy_score}
            score_=metrics[self.metric]
            if self.mtype=='simple':    
                train_score=score_(actual,yhat)
            if self.mtype=='deep':
                train_score=score_(actual,np.argmax(yhat,axis=1))
            print(f"  {self.metric} score : {train_score} ")
   
    def train(self,model=None,data_prepare=None):
        """training basic model
        model=your custom model function
        mtype= defalut: simple
        simple : sklearn model
        deep : neural model"""
       

        self.model.fit(self.X,self.y)
        if self.validation and self.ptype!='timeseries' :
            val_yhat=self.model.predict(val_X)
            # self.scoring(val_y,val_yhat)
     
        train_yhat=self.model.predict(X)
        # self.scoring(y,train_yhat)
        
    def network_(self,input_sizequery,output_shape):
        """deep model networks
        input_sizequery=tuple e.g (30,)
        ouput_shape=no of output 


        """
        input=Input(shape=input_sizequery)
        x=Dense(units=256,activation='relu')(input)
        x=Dense(units=128,activation='relu')(x)
        x=Dense(units=64,activation='relu')(x)
        output=Dense(units=output_shape)(x)
        model=Model(inputs=input,outputs=output)
        if self.ptype=='clf':
            model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
        if self.ptype=='reg':
            model.compile(optimizer='adam',loss='mae',metrics='mse')
        print(model.summary())
        return model
    

class clf(art):
    def __init__(self, df, **kwargs):
        print('what"s your class ?')
        super().__init__(df, **kwargs)
        self.ptype='clf'
    
class reg(art):
    def __init__(self, df,model=None, **kwargs):
        super().__init__(df,**kwargs)
        self.ptype='reg'
        X,y=self.data_process()
        # X,y,val_X,val_y=self.validate(X,y)
        # input_sizequery=self.X.shape[1:]
        # output_shape=self.y.shape
        # print(X.shape,y.shape)
        # self.models={'clf':{'simple': RandomForestClassifier(),
        #                 'deep':self.network_(input_sizequery=input_sizequery,output_shape=output_shape)},
        #             'reg':{'simple':RandomForestRegressor(),
        #                 'deep' :self.network_(input_sizequery=input_sizequery,output_shape=output_shape)},
        #             'timeseries':{'simple':LinearRegression(),}
        #             }
        # if model==None :
        #     self.model=self.models[self.ptype][self.mtype]
        # else:
        #     self.model=model
        
        # self.train()


        
    
class timeseries(art):
    def __init__(self,df,**kwargs):
        print('time is yours')
        super().__init__(df,**kwargs)
        self.ptype='timeseries'


# class auto(classification,regression):
#     def __init__(self,df,model=RandomForestClassifier(),**kwargs):
#         print("Extra Attributes : ptype :['clf','reg','timeseries']")
#         super().__init__(df ,**kwargs)
#         problem={'clf':classification ,'reg': regression}
        
#         data=self.data_process()
#         problem[self.ptype].train(self,data=data,model=model)
    

if __name__=='__main__':
    print("Hello")
