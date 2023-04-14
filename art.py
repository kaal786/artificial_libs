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
pd.options.display.float_format = '{:.2f}'.format

from math import ceil
import random 

import warnings
warnings.simplefilter('ignore')
warnings.filterwarnings('ignore')

import torch
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.preprocessing import LabelEncoder,StandardScaler,MinMaxScaler
from sklearn.metrics import accuracy_score,mean_squared_error,mean_absolute_error,roc_auc_score,f1_score
from sklearn.utils import all_estimators



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
            
        except exception as e:
            pass
        
    def auto_eda(self):
        display(self.info())
        self.plots()
    


    



#     def data_process(self,data_prepare=None):
#         """
#         To process the data to feed model
#         func = your function that process raw data to universel data
#         target = default : 'target' ,you can provide your target value
#         return features , label
#         """
#         if data_prepare!=None :
#             self.df=data_prepare(self.df)
#         else :
#             self.df.drop((self.df.dtypes=='object')[self.df.dtypes =='object'].index.tolist(),axis=1,inplace=True)
#             self.df=self.df.dropna(axis=0)   
#         X=self.df.drop(self.target,axis=1)
#         y=self.df[self.target]
#         if self.method !=None:
#             methods={'zscore':StandardScaler(),'minmax':MinMaxScaler()}
#             nm=methods[self.method]
#             print(nm)
#             X=nm.fit_transform(np.array(X))
#         if self.ptype=="clf" and  self.mtype=='deep':
#             y=to_categorical(y,len(self.df[self.target].unique()))
#         return X,y
        
#     def validate(self,X,y,type='holdout'):
#         if type=='holdout':
#             train_X,val_X,train_y,val_y=train_test_split(X,y,test_size=0.2,random_state=42)
#             print(f"train set : {train_X.shape,train_y.shape} val set : {val_X.shape,val_y.shape}")
#             return train_X,train_y,val_X,val_y
#         if type=='kfold':
#             pass
#         if type=='groupKfold':
#             pass

#     def validate(self,X):
#         pass

#     def scoring(self,actual,yhat):
#         if self.ptype=='clf':
#             if self.mtype=='simple':
#                 train_score=accuracy_score(actual,yhat)
#             if self.mtype=='deep':
#                 train_score=accuracy_score(np.argmax(actual,axis=1),np.argmax(yhat,axis=1))
#             print(f" accuracy score : {train_score} ")
#         if self.ptype=='reg':
#             metrics={'mse':mean_squared_error,'mae':mean_absolute_error,'accuracy':accuracy_score}
#             score_=metrics[self.metric]
#             if self.mtype=='simple':    
#                 train_score=score_(actual,yhat)
#             if self.mtype=='deep':
#                 train_score=score_(actual,np.argmax(yhat,axis=1))
#             print(f"  {self.metric} score : {train_score} ")
   
#     def train(self,model=None,data_prepare=None,epochs=10):
#         """training basic model
#         model=your custom model function
#         mtype= defalut: simple
#         simple : sklearn model
#         deep : neural model"""
#         for epoch in range(self.epochs):
#             self.model.fit(self.X,self.y)
#         if self.validation and self.ptype!='timeseries' :
#             val_yhat=self.model.predict(self.val_X)
#             self.scoring(self.val_y,val_yhat)
     
#         train_yhat=self.model.predict(self.X)
#         self.scoring(self.y,train_yhat)
        
#     def network_(self,input_shape,output_shape):
#         """deep model networks
#         input_shape=tuple e.g (30,)
#         ouput_shape=no of output 


#         """
#         input=Input(shape=input_shape)
#         x=Dense(units=256,activation='relu')(input)
#         x=Dense(units=128,activation='relu')(x)
#         x=Dense(units=64,activation='relu')(x)
#         output=Dense(units=output_shape)(x)
#         model=Model(inputs=input,outputs=output)
#         if self.ptype=='clf':
#             model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
#         if self.ptype=='reg':
#             model.compile(optimizer='adam',loss='mae',metrics='mse')
#         print(model.summary())
#         return model
    

class clf():

    def __init__(self,X,y,validation_data=None,metrics=None,random_state=2023,**kwargs):
        """
        PARAMETERS
        ==========
        
        X : features
        y : target
        validation_data =[val_X,val_y]
        metrics = [accuracy_score,roc_auc_score,f1_score]
        """
        from xgboost import XGBClassifier
        from lightgbm import LGBMClassifier
        from catboost import CatBoostClassifier
        

        self.X=X
        self.y=y
        self.random_state=random_state
        self.validation_data=validation_data
        self.groupbased_estimators=['ClassifierChain','MultiOutputClassifier','OneVsOneClassifier','OneVsRestClassifier','OutputCodeClassifier','RadiusNeighborsClassifier','StackingClassifier','VotingClassifier']
#classifierchain,  MultiOutputClassifier' ,OneVsOneClassifier ,  'OneVsRestClassifier','OutputCodeClassifier' : are use for multi label output
        
        
        self.sklearn_estimators = all_estimators(type_filter='classifier')
        self.extra_estimators = [
            (XGBClassifier().__class__.__name__,XGBClassifier),
            (LGBMClassifier().__class__.__name__,LGBMClassifier),
            (CatBoostClassifier().__class__.__name__,CatBoostClassifier)   ]
        
        self.params={
            'CatBoostClassifier':{'logging_level':"Silent"},
            'LogisticRegression':{'max_iter':2000},
        
        }
        
        self.metrics = metrics if metrics is not None \
        else [accuracy_score,f1_score,roc_auc_score]

    def crossval(self,cv: int=5,scoring: str='accuracy'):
        temp_dict={'models':[],'MeanScore':[],'std':[]}     
        estimators = all_estimators(type_filter='classifier')
        for name,model in self.sklearn_estimators + self.extra_estimators:
            try :
                if name in self.groupbased_estimators :
                    continue
                model=model(random_state=self.random_state,**self.params[name] if name in self.params.keys() else {})
            except :
                model=model(**self.params[name] if name in self.params.keys() else {})                 
                  
            cv_score=cross_val_score(estimator=model,X=self.X,y=self.y, cv=cv, scoring=scoring)
            temp_dict['MeanScore'].append(np.mean(cv_score))
            temp_dict['std'].append(np.std(cv_score))
            temp_dict['models'].append(name)

                
        score_df=pd.DataFrame(temp_dict)
        print('scoring metrics : {} | cv={}'.format(scoring,cv))
        return score_df.sort_values(by=['MeanScore'],ascending=False).reset_index().drop(['index'],axis=1)

    def train(self):
        metris=[]
        self.trained_model={}
        for name,model in self.sklearn_estimators + self.extra_estimators:
            try : 
                if name in self.groupbased_estimators:
                    continue
                model=model(random_state=self.random_state,**self.params[name] if name in self.params.keys() else {})   
            except :
                model=model(**self.params[name] if name in self.params.keys() else {})   
            try :
                model.fit(self.X,self.y)
            except :
                continue
            m_metris=[]
            for metric in self.metrics:
                m_metris.append(metric(self.y,model.predict(self.X)))
                m_metris.append(metric(self.validation_data[1],model.predict(self.validation_data[0])))
            metris.append(m_metris)                
            self.trained_model[name]=model
        score_df=pd.DataFrame(metris,columns=[f'{i}{m.__name__}'for m in self.metrics for i in ['train_','val_']])
        score_df.insert(0,'model',self.trained_model.keys())
        self.score_df=score_df.sort_values(by=['val_'+self.metrics[0].__name__,'train_'+self.metrics[0].__name__],ascending=False)
        return self.score_df
    
    def stacked(self,test_X=None,method: str='ensemble',top: int=5,output: str =None):
        """
        PARAMETERS
        ==========
        test_X : default None , return output on test data
        method : default ensemble , [ensemble,mean]
                        ensemble : group of best estimators
                        mean : mean value of probability of top best estimators
        top : default 5 , no of estimators to choose
        output : default None , will return labels
                        proba : use for probability of each class
        """
        if method=='mean':
            score=[]
            yhat=[]
            if test_X is not None :
                for model in self.score_df['model'][:top]:
                    yhat.append((self.trained_model[model].predict_proba(test_X))[:,1])
                return np.array(yhat).T.mean(axis=1)
            else :
                for model in self.score_df['model'][:top]:
                    yhat.append((self.trained_model[model].predict_proba(self.validation_data[0]))[:,1])
                print(f'stacked mean output probability : {np.array(yhat).T.mean(axis=1)}')
                print(np.where(np.array(yhat).T.mean(axis=1)> 0.5,1,0))
                score=self.metrics[0](self.validation_data[1],np.where(np.array(yhat).T.mean(axis=1)> 0.5,1,0))
                print(score)
        if method=='ensemble':
            top_estimators=[(model,self.trained_model[model]) for model in self.score_df['model'][:top]]
#             print(top_estimators)
            from sklearn.ensemble import StackingClassifier,VotingClassifier
            from sklearn.linear_model import LogisticRegression
            stakers=[('VotingClassifier',VotingClassifier(estimators=top_estimators,voting="soft")),('StackingClassifier',StackingClassifier(estimators=top_estimators, final_estimator=LogisticRegression()))]
            for name,model in stakers:
                score=cross_val_score(model,self.X,self.y,scoring="accuracy", cv=4)
                print(f'{name} mean score : {score.mean()}')
            if test_X is not None:
                model.fit(self.X,self.y)
                if output =='proba':
                    return model.predict_proba(test_X)
                else:
                    return model.predict(test_X)



class reg():
    def __init__(self,X,y,validation_data=None,metrics=None,random_state=2023,**kwargs):
        """
        PARAMETERS
        ==========
        
        X : features
        y : target
        validation_data =[val_X,val_y]
        metrics = [accuracy_score,roc_auc_score,f1_score]
        """
        from xgboost import XGBRegressor
        from lightgbm import LGBMRegressor
        from catboost import CatBoostRegressor

        self.X=X
        self.y=y
        self.random_state=random_state
        self.validation_data=validation_data
        self.groupbased_estimators=['RegressorChain','MultiOutputRegressor','StackingRegressor','VotingRegressor','RadiusNeighborsRegressor']
        
        #fit MultiTaskElasticNet fit MultiTaskElasticNetCV fit MultiTaskLasso fit MultiTaskLassoCV IsotonicRegression
        self.sklearn_estimators = all_estimators(type_filter='regressor')
        self.extra_estimators = [
            (XGBRegressor().__class__.__name__,XGBRegressor),
            (LGBMRegressor().__class__.__name__,LGBMRegressor),
            (CatBoostRegressor().__class__.__name__,CatBoostRegressor)   ]

        self.params={
            'CatBoostRegressor':{'logging_level':"Silent"},
        }
        
        self.metrics = metrics if metrics is not None \
        else [mean_squared_error,mean_absolute_error]

    def crossval(self,cv: int=5,scoring: str='neg_root_mean_squared_error'):
        temp_dict={'models':[],'MeanScore':[],'std':[]}     
        estimators = all_estimators(type_filter='classifier')
        for name,model in self.sklearn_estimators + self.extra_estimators:
            try :
                if name in self.groupbased_estimators :
                    continue
                model=model(random_state=self.random_state,**self.params[name] if name in self.params.keys() else {})
            except :
                model=model(**self.params[name] if name in self.params.keys() else {})                    
            cv_score=cross_val_score(estimator=model,X=self.X,y=self.y, cv=cv, scoring=scoring)
            temp_dict['MeanScore'].append(np.mean(cv_score))
            temp_dict['std'].append(np.std(cv_score))
            temp_dict['models'].append(name)
        score_df=pd.DataFrame(temp_dict)
        print('scoring metrics : {} | cv={}'.format(scoring,cv))
        return score_df.sort_values(by=['MeanScore'],ascending=True).reset_index().drop(['index'],axis=1)

    def train(self):
        metris=[]
        self.trained_model={}
        for name,model in self.sklearn_estimators + self.extra_estimators:
            try : 
                if name in self.groupbased_estimators:
                    continue
                model=model(random_state=self.random_state,**self.params[name] if name in self.params.keys() else {})       
            except :
                model=model(**self.params[name] if name in self.params.keys() else {})   
            try :
                model.fit(self.X,self.y)
            except :
                continue
            m_metris=[]
            for metric in self.metrics:
                m_metris.append(metric(self.y,model.predict(self.X)))
                m_metris.append(metric(self.validation_data[1],model.predict(self.validation_data[0])))
            metris.append(m_metris)              
            self.trained_model[name]=model
        score_df=pd.DataFrame(metris,columns=[f'{i}{m.__name__}'for m in self.metrics for i in ['train_','val_']])
        score_df.insert(0,'model',self.trained_model.keys())
        self.score_df=score_df.sort_values(by=['val_'+self.metrics[0].__name__,'train_'+self.metrics[0].__name__],ascending=True)
        return self.score_df
      



        
    
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
    


if __name__=='__main__':
    pass