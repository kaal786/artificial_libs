import numpy as np


class DecisionTree :
    def __init__(self,max_depth=5,min_size=10):
        self.tree=None
        self.max_depth = max_depth
        self.min_size = min_size
        print('im on')

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
        print('termination condition leaf node' ,max(set(outcomes), key=outcomes.count))
        return max(set(outcomes), key=outcomes.count)

    #recursive function for child split
    def split(self,node, max_depth, min_size, depth):
        left, right = node['groups']
        del(node['groups'])
        # check for a no split
        if not left or not right:
            node['left'] = node['right'] = self.to_terminal(left + right)
            return

        # check for max depth
        if depth >= max_depth:
            node['left'], node['right'] = self.to_terminal(left), self.to_terminal(right)
            return
        # process left child
        if len(left) <= min_size:
            node['left'] = self.to_terminal(left)
        else:
            node['left'] = self.get_split(left)
            self.split(node['left'], max_depth, min_size, depth+1)
        # process right child
        if len(right) <= min_size:
            node['right'] = self.to_terminal(right)
        else:
            node['right'] = self.get_split(right)
            self.split(node['right'], max_depth, min_size, depth+1)

    #build tree
    def build_tree(self,train, max_depth, min_size):
        root = self.get_split(train)
        self.split(root, max_depth, min_size, 1)
        return root

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

    def fit(self,X,y):
        print('fit calling')
        self.X=X
        self.y=y
        self.dt=np.concatenate((X,y.reshape(-1,1)),axis=1).tolist()
        self.tree = self.build_tree(self.dt,self.max_depth, self.min_size)

    def predict(self,test):
        predictions = list()
        for row in test:
                prediction = self.pred_util(self.tree, row)
                predictions.append(prediction)
        return (predictions)





class RandomForest :
    def __init__(self):
        pass