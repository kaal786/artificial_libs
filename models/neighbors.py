from utils.stats import euclidean_distance

class KNeighborsClassifier :
  def __init__(self,n_neighbors=5):
    """
    model=KNeighborsClassifier()
    model.fit(X,y)
    yhat=model.predict(X)
    accuracy_score(y,yhat)
    
    """
    self.n_neighbors=n_neighbors

  def get_neighbors(self,test_row):
    distances = list()
    for idx in range(len(self.X)):
      dist = euclidean_distance(test_row, self.X[idx])
      distances.append((idx, dist))
    distances.sort(key=lambda tup: tup[1])
    neighbors = list()
    for i in range(self.n_neighbors):
      neighbors.append(distances[i][0])
    return neighbors

  def fit(self,X,y) :
    """
    X: List[List] , features
    y: List , label
    """
    self.X=X
    self.y=y
  def predict(self,test):
    """
    testX : List[List]
    """
    predictions = list()
    for row in test:
      neighbors = self.get_neighbors(row)         # return top K neighbors
      output_values = [self.y[idx] for idx in neighbors]
      prediction = max(set(output_values), key=output_values.count)
      predictions.append(prediction)
    return predictions