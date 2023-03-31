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


class eda():
    """Basic methods : 
    features,
    features_plot,
    data_process"""
    def __init__(self,df,target=None,df2=None,df3=None,**kwargs):
        """
        =>Attributes  [df,target='target'] 
        =>Methods
        1.features() 
        2.feature_plot() 
        3.auto_eda()
        """
        self.df=df
        self.target=target
        self.df2=df2
        self.df3=df3
        self.checkdf2=type(self.df2)!=type(None)
        self.checkdf3=type(self.df3)!=type(None)
        self.dfs={'df1':self.df}
        if self.checkdf2:
            self.dfs={'df1':self.df,'df2':self.df2}
        if self.checkdf3:
            self.dfs={'df1':self.df,'df2':self.df2,'df3':self.df3}
        
    def features(self):
        """Basic overviews of features
            for specific features : num_col,obj_col,dt_col"""
        ddf=self.df
        describe_df=pd.DataFrame()
        self.features=np.array(ddf.columns.tolist())

        describe_df['features']=self.features  
        describe_df['dtype']=list(map(lambda x : self.df[x].dtype,self.features))
        describe_df['count']=list(map(lambda x : self.df[x].count(),self.features))
        describe_df['null_values']=list(map(lambda x :self.df[x].isnull().sum(),self.features))
        describe_df['uniques']=list(map(lambda x :len(self.df[x].unique()),self.features))
        describe_df['values']=list(map(lambda x : self.df[x].unique() if len(self.df[x].unique()) < 20 else self.df[x].values[:4].tolist(), self.features))
        describe_df['value_counts']=list(map(lambda x : self.df[x].value_counts().values  if len(self.df[x].unique()) < 20 else self.df[x].value_counts().values[:4].tolist(),self.features ))
        describe_df['skewness']=list(map(lambda x : self.df[x].skew() if self.df[x].dtype!='object' else 'NaN',self.features))
        describe_df['kurtos']=list(map(lambda x : self.df[x].kurt() if self.df[x].dtype!='object' else 'NaN',self.features))
        describe_df=pd.concat([describe_df.set_index('features'),self.df.describe().T],axis=1)
        
        return describe_df

    def features_plot(self):
        """ptype : default plot : basic || 
            plots : corelation ,cmatrix,features"""
        feature=0
        features=self.df.columns.tolist()
        num_cols = list((self.df.dtypes != 'object')[(self.df.dtypes != 'object')].index) 
        obj_cols=list((self.df.dtypes == 'object')[(self.df.dtypes == 'object')].index)
        cat_cols=[feature for feature in features if self.df[feature].nunique() <=10]
        #histogram    
        n_cols = 2
        n_rows = ceil(len(features)/n_cols)
        fig1, ax = plt.subplots(n_rows, n_cols, figsize=(20, n_rows*4))
        ax = ax.flatten()
        for i, feature in enumerate(features):
            for item in self.dfs:
                try :
                    sns.histplot(self.dfs[item][feature],ax=ax[i],bins=50,label=item)
                except :
                    continue
            # remove axes to show only one at the end
            plot_axes = [ax[i]]
            handles = []
            labels = []
            for plot_ax in plot_axes:
                handles += plot_ax.get_legend_handles_labels()[0]
                labels += plot_ax.get_legend_handles_labels()[1]
                plot_ax.legend().remove()
        for i in range(i+1, len(ax)):
            ax[i].axis('off')
        fig1.suptitle(f'Dataset Feature Distributions-[hist]', ha='center',  fontweight='bold', fontsize=25)   
        fig1.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.95), fontsize=25, ncol=2)
         
        #kde comparison
        n_rows = ceil(len(num_cols)/n_cols)
        fig2, ax = plt.subplots(n_rows, n_cols, figsize=(20, n_rows*4))
        ax = ax.flatten()
        for i, column in enumerate(num_cols):
            plot_axes = [ax[i]]
            for item in self.dfs:
                try:
                    sns.kdeplot(self.dfs[item][column], label=item,ax=ax[i],)
                except :
                    continue
            ax[i].set_title(f'{column}')
            ax[i].set_xlabel(None)
            # remove axes to show only one at the end
            plot_axes = [ax[i]]
            handles = []
            labels = []
            for plot_ax in plot_axes:
                handles += plot_ax.get_legend_handles_labels()[0]
                labels += plot_ax.get_legend_handles_labels()[1]
                plot_ax.legend().remove()
        for i in range(i+1, len(ax)):
            ax[i].axis('off')
        fig2.suptitle(f'Comparison Dataset Feature Distributions\n\n\n\n\n\n', ha='center',  fontweight='bold', fontsize=25)
        fig2.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.95), fontsize=25, ncol=2)
        plt.tight_layout()    
        
        #correlation 
        fig3, ax = plt.subplots(len(self.dfs),1,figsize=(20, len(self.dfs)*10))
        ax = ax.flatten() if len(self.dfs) >1 else [ax]
        for i,item in enumerate(self.dfs):
            mask=np.triu(np.ones_like(self.dfs[item].corr()))
            sns.heatmap(self.dfs[item].corr(),mask=mask,cmap='Dark2',annot=True,ax=ax[i])
            ax[i].set_title(f'{item} data');
        fig3.suptitle(f'Correalation between features \n\n', ha='center',  fontweight='bold', fontsize=25)
        
        #categorical features with target
        try :
            assert isinstance(self.target,str)
            n_rows=ceil(len(cat_cols)/n_cols)
            fig4, ax = plt.subplots(n_rows, n_cols, figsize=(20, n_rows*4), dpi=150)
            ax = ax.flatten()
            for i, ft in enumerate(cat_cols):
                sns.histplot(data=self.df,x=self.target, hue=ft,multiple="stack",edgecolor=".3",linewidth=.5,ax=ax[i],cbar=True)    
                ax[i].set_title(f'{ft}')
                ax[i].set_xlabel(None)
            fig4.suptitle(f'Target distribution with categorical features\n\n', ha='center',  fontweight='bold', fontsize=25)
            
        except :
            pass
        

        
        
        
        
        
    def auto_eda(self):
        display(self.features())
        self.features_plot()
    




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
        if self.ptype=="clf" and  self.mtype=='deep':
            y=to_categorical(y,len(self.df[self.target].unique()))
        return X,y
        
    def validate(self,X,y,type='holdout'):
        if type=='holdout':
            train_X,val_X,train_y,val_y=train_test_split(X,y,test_size=0.2,random_state=42)
            print(f"train set : {train_X.shape,train_y.shape} val set : {val_X.shape,val_y.shape}")
            return train_X,train_y,val_X,val_y
        if type=='kfold':
            pass
        if type=='groupKfold':
            pass

    def validate(self,X):
        pass

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
   
    def train(self,model=None,data_prepare=None,epochs=10):
        """training basic model
        model=your custom model function
        mtype= defalut: simple
        simple : sklearn model
        deep : neural model"""
        for epoch in range(self.epochs):
            self.model.fit(self.X,self.y)
        if self.validation and self.ptype!='timeseries' :
            val_yhat=self.model.predict(self.val_X)
            self.scoring(self.val_y,val_yhat)
     
        train_yhat=self.model.predict(self.X)
        self.scoring(self.y,train_yhat)
        
    def network_(self,input_shape,output_shape):
        """deep model networks
        input_shape=tuple e.g (30,)
        ouput_shape=no of output 


        """
        input=Input(shape=input_shape)
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
    

# class clf(art):
#     def __init__(self, df,model=None, **kwargs):
#         print('what"s your class ?')
#         super().__init__(df, **kwargs)
#         self.ptype='clf'
#         X,y=self.data_process()
#         self.X,self.y,self.val_X,self.val_y=self.validate(X,y)
#         input_shape=X.shape[1:]
#         output_shape=1
#         print(X.shape,y.shape)
#         self.models={'clf':{'simple': RandomForestClassifier(),
#                         'deep':self.network_(input_shape=input_shape,output_shape=output_shape)},
#                     'reg':{'simple':RandomForestRegressor(),
#                         'deep' :self.network_(input_shape=input_shape,output_shape=output_shape)},
#                     'timeseries':{'simple':LinearRegression(),}
#                     }
#         if model==None :
#             self.model=self.models[self.ptype][self.mtype]
#         else:
#             self.model=model
        
#         self.train()
    
# class reg(art):
#     def __init__(self, df,model=None, **kwargs):
#         super().__init__(df,**kwargs)
#         self.ptype='reg'
#         X,y=self.data_process()
#         self.X,self.y,self.val_X,self.val_y=self.validate(X,y)
#         input_shape=X.shape[1:]
#         output_shape=1
#         print(X.shape,y.shape)
#         self.models={'clf':{'simple': RandomForestClassifier(),
#                         'deep':self.network_(input_shape=input_shape,output_shape=output_shape)},
#                     'reg':{'simple':RandomForestRegressor(),
#                         'deep' :self.network_(input_shape=input_shape,output_shape=output_shape)},
#                     'timeseries':{'simple':LinearRegression(),}
#                     }
#         if model==None :
#             self.model=self.models[self.ptype][self.mtype]
#         else:
#             self.model=model
        
#         self.train(self.epochs)


        
    
# class timeseries(art):
#     def __init__(self,df,**kwargs):
#         print('time is yours')
#         super().__init__(df,**kwargs)
#         self.ptype='timeseries'


# class auto(classification,regression):
#     def __init__(self,df,model=RandomForestClassifier(),**kwargs):
#         print("Extra Attributes : ptype :['clf','reg','timeseries']")
#         super().__init__(df ,**kwargs)
#         problem={'clf':classification ,'reg': regression}
        
#         data=self.data_process()
#         problem[self.ptype].train(self,data=data,model=model)
    