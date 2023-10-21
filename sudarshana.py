
# external libs for us.all 
from abc import abstractmethod
from logging import exception
from turtle import shape
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib
plt.rcParams['figure.figsize']=(12,6)
plt.style.use('fivethirtyeight')

import pandas as pd
pd.options.display.max_columns = 25
pd.options.display.max_rows = 25


from math import ceil
import random 

import warnings
warnings.simplefilter('ignore')
warnings.filterwarnings('ignore')
from tqdm import tqdm

import torch
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.preprocessing import LabelEncoder,StandardScaler,MinMaxScaler
from sklearn.metrics import accuracy_score,mean_squared_error,mean_absolute_error,roc_auc_score,f1_score
from sklearn.utils import all_estimators



import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Dense ,Dropout
from tensorflow.keras.models import Model


from _estimators import groupbased_estimators,extrareg_estimators,extraclf_estimators,sensitive_estimators
from xgboost import XGBRegressor,XGBClassifier
from lightgbm import LGBMRegressor,LGBMClassifier
from catboost import CatBoostRegressor,CatBoostClassifier


from utils._dataprocess import reduce_mem



#set seeds
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
torch.manual_seed(42)


class eda():
    def __init__(self,df,target: str =None,df2=None,df3=None,**kwargs):
        """
        =>Attributes  
        ===============
        df : pandas dataframe
        target : str,'class name'
        df2 : test/val dataframe
        df3 : other dataframe

        => Methods
        ===============
        1.Info() 
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
        
        self.features=self.df.columns.tolist()
        self.num_cols = list(((self.df.dtypes != '<M8[ns]') & (self.df.dtypes != 'object'))[(self.df.dtypes != '<M8[ns]') & (self.df.dtypes != 'object')].index)
        self.obj_cols=list((self.df.dtypes == 'object')[(self.df.dtypes == 'object')].index)
        self.cat_cols=[feature for feature in self.features if self.df[feature].nunique() <=10]
        self.datetime_cols=list((self.df.dtypes == '<M8[ns]')[(self.df.dtypes == '<M8[ns]')].index)
    def info(self):
        """Basic overviews of features
            for specific features : num_col,obj_col,dt_col"""
        describe_df=pd.DataFrame()

        describe_df['features']=self.features  
        describe_df['dtype']=list(map(lambda x : self.df[x].dtype,self.features))
        describe_df['count']=list(map(lambda x : self.df[x].count(),self.features))
        describe_df['null_values']=list(map(lambda x :self.df[x].isnull().sum(),self.features))
        describe_df['uniques']=list(map(lambda x :len(self.df[x].unique()),self.features))
        describe_df['values']=list(map(lambda x : self.df[x].unique() if len(self.df[x].unique()) < 20 else self.df[x].values[:4].tolist(), self.features))
        describe_df['value_counts']=list(map(lambda x : self.df[x].value_counts().values  if len(self.df[x].unique()) < 20 else self.df[x].value_counts().values[:4].tolist(),self.features ))
        describe_df['skewness']=list(map(lambda x : self.df[x].skew() if x in self.num_cols else 'NaN',self.features))
        describe_df['kurtos']=list(map(lambda x : self.df[x].kurt() if x in self.num_cols else 'NaN',self.features))
        describe_df=pd.concat([describe_df.set_index('features'),self.df.describe().T],axis=1)
        
        return describe_df

    def plots(self):
        """ptype : default plot : basic || 
            plots : corelation ,cmatrix,features"""
        feature=0
 
        #histogram    
        n_cols = 2
        n_rows = ceil(len(self.features)/n_cols)
        fig1, ax = plt.subplots(n_rows, n_cols, figsize=(20, n_rows*4))
        ax = ax.flatten()
        for i, feature in enumerate(self.features):
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
        n_rows = ceil(len(self.num_cols)/n_cols)
        fig2, ax = plt.subplots(n_rows, n_cols, figsize=(20, n_rows*4))
        ax = ax.flatten()
        for i, column in enumerate(self.num_cols):
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
        if self.target is not None:
            try :
                assert isinstance(self.target,str)
                n_rows=ceil(len(self.cat_cols)/n_cols)
                fig4, ax = plt.subplots(n_rows, n_cols, figsize=(20, n_rows*4), dpi=150)
                ax = ax.flatten()
                for i, ft in enumerate(self.cat_cols):
                    sns.histplot(data=self.df,x=self.target, hue=ft,multiple="stack",edgecolor=".3",linewidth=.5,ax=ax[i],cbar=True)    
                    ax[i].set_title(f'{ft}')
                    ax[i].set_xlabel(None)
                fig4.suptitle(f'Target distribution with categorical features\n\n', ha='center',  fontweight='bold', fontsize=25)
                
            except :
                pass
        
    def auto_eda(self):
        display(self.info())
        self.plots()



class tabular_supervised :
    def __init__(self,
                    X,
                    y=None,
                    df2=None,
                    type_filter='regressor',
                    drop_col=[],
                    validation_data=None,
                    metrics=None,
                    model_selection='train_test_split',
                    **kwargs):
        """
        PARAMETERS
        ==========
        
        X : DataFrame,numpy-array ,features
        y :DataFrame,numpy-array,str ,target

        validation_data = [val_X,val_y],DataFrame,numpy-array
        type_filter=default 'regressor' ,str ,['regressor',classifier]
        metrics = [accuracy_score,roc_auc_score,f1_score]
        """
        
        self.type_filter=type_filter
        self.model_selection=model_selection
        self.le={}
        
        self.X=X
        self.y=y

  
       
        self.validation_data=validation_data
        self.groupbased_estimators=groupbased_estimators

        self.sklearn_estimators = all_estimators(type_filter=self.type_filter)

        if self.type_filter=='regressor':
            self.extra_estimators = extrareg_estimators
            self.params={
                'CatBoostRegressor':{'logging_level':"Silent"},
            }
            self.metrics = metrics if metrics is not None \
            else [mean_squared_error,mean_absolute_error]
            self.scoring='neg_root_mean_squared_error'
            self.ascending=True

        if self.type_filter=='classifier' :
            self.extra_estimators = extraclf_estimators
            self.params={
                'CatBoostClassifier':{'logging_level':"Silent"},
                'LogisticRegression':{'max_iter':2000},
            }
            self.metrics = metrics if metrics is not None \
            else [accuracy_score,f1_score,roc_auc_score]  
            self.scoring='accuracy'
            self.ascending=False
    

    def crossval(self,cv: int=5,scoring:str= None):
        self.scoring=scoring if scoring is not None else self.scoring
        temp_dict={'models':[],'MeanScore':[],'std':[]}     
        for name,model in self.sklearn_estimators + self.extra_estimators:
            try :
                if name in self.groupbased_estimators :
                    continue
                model=model(**self.params[name] if name in self.params.keys() else {})
            except :
                model=model(**self.params[name] if name in self.params.keys() else {})                    
            cv_score=cross_val_score(estimator=model,X=self.X,y=self.y, cv=cv, scoring=self.scoring)
            temp_dict['MeanScore'].append(np.mean(cv_score))
            temp_dict['std'].append(np.std(cv_score))
            temp_dict['models'].append(name)
        score_df=pd.DataFrame(temp_dict)
        print('scoring metrics : {} | cv={}'.format(self.scoring,cv))
        return score_df.sort_values(by=['MeanScore'],ascending=self.ascending).reset_index().drop(['index'],axis=1)



    def train(self,):
        metris=[]
        self.trained_model={}
        for name,model in tqdm(self.sklearn_estimators + self.extra_estimators):
            print(name)
            try : 
                if name in self.groupbased_estimators+sensitive_estimators:
                    continue
                cmodel=model(**self.params[name] if name in self.params.keys() else {})       
                cmodel.fit(self.X,self.y)
            except Exception as e:
                print(name,e) 
                continue

            m_metris=[]
            for metric in self.metrics:
                m_metris.append(metric(self.y,cmodel.predict(self.X)))
                if self.validation_data is not None:
                    m_metris.append(metric(self.validation_data[1],cmodel.predict(self.validation_data[0])))
            metris.append(m_metris) 
            self.trained_model[name]=cmodel
            print(metris)
        # score_df=pd.DataFrame(metris,columns=[f'{i}{m.__name__}'for m in self.metrics for i in ['train_','val_']])
        # score_df.insert(0,'model',self.trained_model.keys())
        # self.score_df=score_df.sort_values(by=['val_'+self.metrics[0].__name__,'train_'+self.metrics[0].__name__],ascending=self.ascending)

        # return self.score_df
    
    def stacked(self,test_X=None,method: str='ensemble',top: int=5,proba: str =False,cv:int=5):
        """
        PARAMETERS
        ==========
        test_X : default None , return output on test data
        method : default ensemble , [ensemble,mean]
                        ensemble : group of best estimators
                        mean : mean value of probability of top best estimators
        top : default 5 , no of estimators to choose
        proba :bool, default False,use for probability of each class

        cv:int=5                
        """
        try : 
            test_input=test_X if test_X is not None else self.validation_data[0]
            if method=='mean':
                score=[]
                yhat=[]
                if self.type_filter=='regressor':
                    for model in self.score_df['model'][:top]:
                            yhat.append((self.trained_model[model].predict(test_input)))
                    if test_X is None :
                        score=self.metrics[0](np.array(self.validation_data[1]),np.array(yhat).mean(axis=0))
                    else :
                        return np.array(yhat).mean(axis=0)

                if self.type_filter=='classifier':
                    for model in self.score_df['model'][:top]:
                            yhat.append((self.trained_model[model].predict_proba(test_input))[:,1])
                    if test_X is None:
                        class_yhat=np.where(np.array(yhat).T.mean(axis=1)> 0.5,1,0)  # probability to class 
                        score=self.metrics[0](self.validation_data[1],class_yhat)
                    else : 
                        return np.array(yhat).T.mean(axis=1)

                print("TOP{}Mean_{} => {}".format(top,self.metrics[0].__name__,score))
        
                    

            if method=='ensemble':
                top_estimators=[(model,self.trained_model[model]) for model in self.score_df['model'][:top]]
                from sklearn.ensemble import StackingClassifier,VotingClassifier, StackingRegressor,VotingRegressor
                from sklearn.linear_model import LogisticRegression,LinearRegression
                if self.type_filter=='classifier':
                    stakers=[('VotingClassifier',VotingClassifier(estimators=top_estimators,voting="soft")),('StackingClassifier',StackingClassifier(estimators=top_estimators, final_estimator=LogisticRegression()))]
                if self.type_filter=='regressor':
                    stakers=[('VotingRegressor',VotingRegressor(estimators=top_estimators)),('StackingRegressor',StackingRegressor(estimators=top_estimators))]

                for name,model in stakers:
                    model.fit(self.X,self.y)
                    if test_X is None:
                        score=self.metrics[0](np.array(self.validation_data[1]),model.predict(test_input))
                    else :
                        if proba==True and self.type_filter=='classifier' :
                            return model.predict_proba(test_X)
                        else:
                            return model.predict(test_X)    
                    print("{}_{} => {}".format(name,self.metrics[0].__name__,score))
                    
        except AttributeError :
            print(f'please run the train() method first to get top{top} train models')




class smartrun: 
    def __init__(self,
                    X,
                    target: str,
                    validation_data=None,
                    testx=None,
                    type_filter='regressor',
                    drop_col=[],
                    num_col=[],
                    cat_col=[],
                    imputation_method='none',
                 
                    model_selection_method='train_test_split',
                    metrics=None,
                    **kwargs):

        """
        X : pandas dataframe
        y : str format of target col
        drop_col : list of feature to drop,
        num_col : list of features to handle as numericle,
        cat_col : list of features to handle as categorical or bool,
        imputation_method : 'none','drop','topmost'
        
        
        """
     
        self.X=X 
        self.target=target
        self.testx=testx
        self.drop_col=drop_col
        self.num_col=num_col
        self.cat_col=cat_col
        self.imputation_method=imputation_method
        self.model_selection_method=model_selection_method
        
        
   
    def imputation(self,X):
        if len(X.isnull().sum()[X.isnull().sum() > 0].index.values) > 0:
            if self.imputation_method=='drop':
                X=X.dropna()
            if self.imputation_method=='topmost':
                for col in train_df.isnull().sum()[train_df.isnull().sum() > 0].index.values : 
                    if col in num_col:
                        X.fillna(X[col].mean())
                    if col in cat_col:
                        value=X[col].value_counts().index.values[0]
                        X.fillna(value)
            return X           
        else : return X
            
    def model_selection(self,X):
        if self.model_selection_method=='train_test_split':
            trainx,valx,trainy,valy=train_test_split(X.drop([self.target],axis=1).values,X[target].values)
            print('3. trainx :{} , triany :{}\n valx :{} ,valy :{}\n'.format(trainx.shape,trainy.shape,valx.shape,valy.shape))
            return trainx,valx,trainy,valy
        if self.model_selection_method=='kfold':
            kf=KFold(n_splits=5)
            tidx=kf.split(train_df,).__next__()[0]
            vidx=kf.split(train_df,).__next__()[1]
            trainx,valx,trainy,valy=X.loc[tidx],X.loc[vidx],X[target].loc[tidx],X[target].loc[vidx]
            print('3. trainx :{} , triany :{}\n valx :{} ,valy :{}\n'.format(trainx.shape,trainy.shape,valx.shape,valy.shape))
            return trainx,valx,trainy,valy 
    
    def dataprocess(self):
        self.X=reduce_mem(self.X)
        self.X=self.X.drop(self.drop_col,axis=1)
        print('2. features with NaN values :{} with respected null percentage are :{}\n'.format(
                self.X.isnull().sum()[self.X.isnull().sum() > 0],
                list(map(lambda x : (x/self.X.shape[0])*100,self.X.isnull().sum()[self.X.isnull().sum() > 0].values.tolist()))
                ))
        if self.imputation_method != 'none':
            self.X=self.imputation(self.X)
        return self.model_selection(self.X)





if __name__=='__main__':
    pass