import numpy as np
from random import seed,randrange
from math import sqrt,log2

class DecisionTreeClassifier :
    def __init__(self,max_depth=5,min_size=10):
        self.tree=None
        self.max_depth = max_depth
        self.min_size = min_size

    def test_split(self,index, value, dataset):
            left, right = list(), list()
            for row in dataset:
                if row[index] < value:
                    left.append(row)
                else:
                    right.append(row)
            return left, right

    def gini_index(self,groups, classes):
        # count all samples at split point
        n_instances = float(sum([len(group) for group in groups]))
        #print('n',n_instances)
        # sum weighted Gini index for each group
        gini = 0.0
        for group in groups:
            size = float(len(group))

            # avoid divide by zero
            if size == 0:
                continue
            score = 0.0
            # score the group based on the score for each class
            for class_val in classes:
                p = [row[-1] for row in group].count(class_val) / size
                #print('no of P in group',p)
                score += p * p
            # weight the group score by its relative size
            gini += (1.0 - score) * (size / n_instances)
        return gini

    def get_split(self,dataset):
        class_values = list(set(row[-1] for row in dataset))
        b_index, b_value, b_score, b_groups = 999, 999, 999, None
        for index in range(len(dataset[0])-1):
            #print('index',index)
            for row in dataset:
                #print('row',row)
                groups = self.test_split(index, row[index], dataset)
                #print('groups',len(groups),len(groups[0]),len(groups[1]))
                gini = self.gini_index(groups, class_values)
                #print('X%d < %.3f Gini=%.3f' % ((index+1), row[index], gini))
                if gini < b_score:
                    b_index, b_value, b_score, b_groups = index, row[index], gini, groups

        return {'index':b_index, 'value':b_value, 'groups':b_groups}

    def to_terminal(self,group):
        outcomes = [row[-1] for row in group]
        #print('termination condition leaf node' ,max(set(outcomes), key=outcomes.count))
        return max(set(outcomes), key=outcomes.count)
   
    def split(self,node, depth):
        left, right = node['groups']
        del(node['groups'])
        # check for a no split
        if not left or not right:
            node['left'] = node['right'] = self.to_terminal(left + right)
            return

        # check for max depth
        if depth >= self.max_depth:
            node['left'], node['right'] = self.to_terminal(left), self.to_terminal(right)
            return
        # process left child
        if len(left) <= self.min_size:
            node['left'] = self.to_terminal(left)
        else:
            node['left'] = self.get_split(left)
            self.split(node['left'], depth+1)
        # process right child
        if len(right) <= self.min_size:
            node['right'] = self.to_terminal(right)
        else:
            node['right'] = self.get_split(right)
            self.split(node['right'], depth+1)

    def pred_util(self,node, row):
        if row[node['index']] < node['value']:
            if isinstance(node['left'], dict):
                return self.pred_util(node['left'], row)
            else:
                return node['left']
        else:
            if isinstance(node['right'], dict):
                return self.pred_util(node['right'], row)
            else:
                return node['right']

    def build_tree(self,data):
        root = self.get_split(data)
        self.split(root,1)
        return root

    def fit(self,X,y):
        X=np.array(X)
        y=np.array(y)
        data=np.concatenate((X,y.reshape(-1,1)),axis=1).tolist()
        self.tree = self.build_tree(data)

    def predict(self,test):
        test=np.array(test)
        predictions = list()
        for row in test:
                prediction = self.pred_util(self.tree, row)
                predictions.append(prediction)
        return (predictions)

class BaggingClassifier(DecisionTreeClassifier) :
    def __init__(self,n_estimators=10,max_depth=5,min_size=10,sample_size = 0.50):
        self.trees=None
        self.max_depth = max_depth
        self.min_size = min_size
        self.sample_size = sample_size
        self.n_estimators=n_estimators
        DecisionTreeClassifier.__init__(self,max_depth,min_size)


    def subsample(self,dataset):
        sample = list()
        n_sample = round(len(dataset) * self.sample_size)
        while len(sample) < n_sample:
            index = randrange(len(dataset))
            sample.append(dataset[index])
        return sample

    def bagging_predict(self,row):
        predictions = [self.pred_util(tree, row) for tree in self.trees]
        return max(set(predictions), key=predictions.count)

    def fit(self,X,y):
        X=np.array(X)
        y=np.array(y)
        data=np.concatenate((X,y.reshape(-1,1)),axis=1).tolist()
        self.trees = list()
        for i in range(self.n_estimators):
            sample = self.subsample(data)
            tree = self.build_tree(sample)
            self.trees.append(tree)

    def predict(self,test):
        test=np.array(test)
        predictions = [self.bagging_predict(row) for row in test]
        return (predictions)

class RandomForestClassfier(BaggingClassifier) :
    def __init__(self,n_estimators=10,max_depth=5,min_size=10,sample_size = 0.50,n_features='sqrt'):
        self.trees=None
        self.max_depth = max_depth
        self.min_size = min_size
        self.sample_size = sample_size
        self.n_estimators=n_estimators
        BaggingClassifier.__init__(self,n_estimators,max_depth,min_size,sample_size)
        self.n_features=n_features
    #for features bootstrapping
    def get_split(self,dataset):
        class_values = list(set(row[-1] for row in dataset))
        b_index, b_value, b_score, b_groups = 999, 999, 999, None
        features = list()
        while len(features) < self.n_features:
            index = randrange(len(dataset[0])-1)
            if index not in features:
                features.append(index)
            for index in features:
                for row in dataset:
                    groups = self.test_split(index, row[index], dataset)
                    gini = self.gini_index(groups, class_values)
                    if gini < b_score:
                        b_index, b_value, b_score, b_groups = index, row[index], gini, groups
        return {'index':b_index, 'value':b_value, 'groups':b_groups}

    def fit(self,X,y):
        X=np.array(X)
        y=np.array(y)
        data=np.concatenate((X,y.reshape(-1,1)),axis=1).tolist()

        #for feature boostrapping
        if self.n_features is None:
            self.n_features=len(data[0])-1
        elif self.n_features=='sqrt':
            self.n_features=int(sqrt(len(data[0])-1))
        elif self.n_features=='log2':
            self.n_features=int(log2(len(data[0])-1))

        self.trees = list()
        for i in range(self.n_estimators):
            sample = self.subsample(data)
            tree = self.build_tree(sample)
            self.trees.append(tree)

    def predict(self,test):
        test=np.array(test)
        predictions = [self.bagging_predict(row) for row in test]
        return (predictions)

class DecisionTreeRegressor :
    def __init__(self,max_depth=5,min_size=10):
        self.tree=None
        self.max_depth = max_depth
        self.min_size = min_size

    def test_split(self,index, value, dataset):
            left, right = list(), list()
            for row in dataset:
                if row[index] <= value:
                    left.append(row)
                if row[index] > value:
                    right.append(row)
            return left, right

    def variance(self,y):
        if len(y)==0:
            return 0
        mean = np.mean(y)
        return np.mean((y - mean) ** 2)

    def best_split(self,dataset):
        b_index, b_value, b_score, b_groups = 999, 999, -float('inf'), None
        for index in range(len(dataset[0])-1):
            #print('index',index)
            for row in dataset:
                #print('row',row)
                groups = self.test_split(index, row[index], dataset)
                #print('groups',len(groups),len(groups[0]),len(groups[1]))
                mse = self.mse_cal(dataset,groups)
                #print('X%d < %.3f Gini=%.3f' % ((index+1), row[index], gini))
                if mse > b_score:
                    b_index, b_value, b_score, b_groups = index, row[index], mse, groups
        return {'index':b_index, 'value':b_value, 'groups':b_groups}


    def mse_cal(self,dataset,groups):
        n_instances = float(sum([len(group) for group in groups]))
        se=0.0
        for group in groups:
            size=float(len(group))
            if size==0:
                continue
            yh=[row[-1]for row in group]
            se+= np.var(yh) * (size/len(dataset))
            #print(se)
        return np.var([row[-1] for row in dataset]) - se


    def to_terminal(self,group):
        outcomes = [row[-1] for row in group]
        return np.mean(outcomes)


    #recursive function for child split
    def split(self,node, depth):
        left, right = node['groups']
        del(node['groups'])
        # check for a no split
        if not left or not right:
            node['left'] = node['right'] = self.to_terminal(left + right)
            return

        # check for max depth
        if depth >= self.max_depth:
            node['left'], node['right'] = self.to_terminal(left), self.to_terminal(right)
            return
        # process left child
        if len(left) <= self.min_size:
            node['left'] = self.to_terminal(left)
        else:
            node['left'] = self.best_split(left)
            self.split(node['left'], depth+1)
        # process right child
        if len(right) <= self.min_size:
            node['right'] = self.to_terminal(right)
        else:
            node['right'] = self.best_split(right)
            self.split(node['right'], depth+1)

    def pred_util(self,node, row):
        if row[node['index']] < node['value']:
            if isinstance(node['left'], dict):
                return self.pred_util(node['left'], row)
            else:
                return node['left']
        else:
            if isinstance(node['right'], dict):
                return self.pred_util(node['right'], row)
            else:
                return node['right']



    def build_tree(self,data):
        root = self.best_split(data)
        self.split(root,1)
        return root

    def fit(self,X,y):
        X=np.array(X)
        y=np.array(y)
        data=np.concatenate((X,y.reshape(-1,1)),axis=1).tolist()
        self.tree = self.build_tree(data)

    def predict(self,test):
        test=np.array(test)
        predictions = list()
        for row in test:
                prediction = self.pred_util(self.tree, row)
                predictions.append(prediction)
        return (predictions)

class BaggingRegressor(DecisionTreeRegressor) :
    def __init__(self,n_estimators=10,max_depth=5,min_size=10,sample_size = 0.50):
        self.trees=None
        self.max_depth = max_depth
        self.min_size = min_size
        self.sample_size = sample_size
        self.n_estimators=n_estimators
        DecisionTreeRegressor.__init__(self,max_depth,min_size)


    def subsample(self,dataset):
        sample = list()
        n_sample = round(len(dataset) * self.sample_size)
        while len(sample) < n_sample:
            index = randrange(len(dataset))
            sample.append(dataset[index])
        return sample

    def bagging_predict(self,row):
        predictions = [self.pred_util(tree, row) for tree in self.trees]
        return np.mean(predictions)

    def fit(self,X,y):
        X=np.array(X)
        y=np.array(y)
        data=np.concatenate((X,y.reshape(-1,1)),axis=1).tolist()
        self.trees = list()
        for i in range(self.n_estimators):
            sample = self.subsample(data)
            tree = self.build_tree(sample)
            self.trees.append(tree)

    def predict(self,test):
        test=np.array(test)
        predictions = [self.bagging_predict(row) for row in test]
        return (predictions)

class RandomForestRegressor(BaggingRegressor) :
    def __init__(self,n_estimators=10,max_depth=5,min_size=2,sample_size = 0.50,n_features='sqrt'):
        self.trees=None
        self.max_depth = max_depth
        self.min_size = min_size
        self.sample_size = sample_size
        self.n_estimators=n_estimators
        BaggingRegressor.__init__(self,n_estimators,max_depth,min_size,sample_size)
        self.n_features=n_features
    #for features bootstrapping
    def get_split(self,dataset):
        class_values = list(set(row[-1] for row in dataset))
        b_index, b_value, b_score, b_groups = 999, 999, 999, None
        features = list()
        while len(features) < self.n_features:
            index = randrange(len(dataset[0])-1)
            if index not in features:
                features.append(index)
            for index in features:
                for row in dataset:
                    groups = self.test_split(index, row[index], dataset)
                    gini = self.gini_index(groups, class_values)
                    if gini < b_score:
                        b_index, b_value, b_score, b_groups = index, row[index], gini, groups
        return {'index':b_index, 'value':b_value, 'groups':b_groups}

    def best_split(self,dataset):
        b_index, b_value, b_score, b_groups = 999, 999, -float('inf'), None
        features = list()
        while len(features) < self.n_features:
            index = randrange(len(dataset[0])-1)
            if index not in features:
                features.append(index)
            for index in features:
                for row in dataset:
                    groups = self.test_split(index, row[index], dataset)
                    mse = self.mse_cal(dataset,groups)
                    if mse > b_score:
                        b_index, b_value, b_score, b_groups = index, row[index], mse, groups
        return {'index':b_index, 'value':b_value, 'groups':b_groups}

    def fit(self,X,y):
        X=np.array(X)
        y=np.array(y)
        data=np.concatenate((X,y.reshape(-1,1)),axis=1).tolist()

        #for feature boostrapping
        if self.n_features is None:
            self.n_features=len(data[0])-1
        elif self.n_features=='sqrt':
            self.n_features=int(sqrt(len(data[0])-1))
        elif self.n_features=='log2':
            self.n_features=int(log2(len(data[0])-1))

        self.trees = list()
        for i in range(self.n_estimators):
            sample = self.subsample(data)
            tree = self.build_tree(sample)
            self.trees.append(tree)

    def predict(self,test):
        test=np.array(test)
        predictions = [self.bagging_predict(row) for row in test]
        return (predictions)
