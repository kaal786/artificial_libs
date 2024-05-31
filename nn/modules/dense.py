import numpy as np
class Dense:
    def __init__(self,inputs,outputs=32):


        self.weight=np.random.rand(outputs,inputs ).astype(np.float32)
        self.bias=np.random.rand(outputs).astype(np.float32)

    def __call__(self,x):
        self.z=np.matmul(x,self.weight.T)+self.bias
        return self.z
