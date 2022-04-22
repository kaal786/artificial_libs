# external libs for use 
from abc import abstractmethod
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder


class common():
    def __init__(self,df,**kwargs):
        self.df=df
    
    def features(self,num_col=False,obj_col=False,dt_col=False):
        df1=pd.DataFrame()
        self.features=np.array(df.columns.tolist())
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
        



    def data_prepare(self,encoder='oh_en',encode_features=None,drop_features=None,target_='target',split_method='hold_out'):
        """handling empty data"""
        if True in np.array(self.df.iloc[:1].isnull().any()):
            if True in np.array(self.df.iloc[:1].isnull().any()):
                 self.df=self.df.dropna(axis=0)
            self.df=self.df.fillna(method='bfill',axis=1)
        self.df=self.df.fillna(method='ffill',axis=1)

        """encoding data"""
        if encode_features==None:
            print("please give us features for encoding")
        else : 
            if encoder=='oh_en':
                oh_df=pd.get_dummies(self.df[[encode_features]])
                self.df=pd.concat([self.df,oh_df],axis=1)
            
            if encoder=='ord_en':
                
                le = LabelEncoder()
                le.fit_transform(self.df[[encode_features]])
                
        if drop_features !=None:    
            self.df=self.df.drop(encode_features+drop_features,axis=1)
        else :
            self.df=self.df.drop(encode_features,axis=1)


        """spliting data"""
        X=self.df.drop([target_],axis=1)
        y=self.df[target_]
        if split_method=='hold_out':
            train_X,val_X,train_y,val_y=train_test_split(X,y,test_size=0.2)
        






    def features_plot(self,plot_type):
        if plot_type=='correlation' :
            plt.subplots(figsize = (16, 9))
            sns.heatmap(self.df.corr(),annot = True,center = 0)

        


if __name__=='__main__':
    df=pd.read_csv('../data/regression/titanic_survived/test.csv')
    # print(df)

    # f1=common(df)
    # print(f1.features())

    f2=common(df)

