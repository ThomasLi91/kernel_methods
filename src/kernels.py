import numpy as np

class RBF:
    def __init__(self, gamma=1.):
        self.gamma = gamma  ## the variance of the kernel
    def kernel(self,X,Y):
        ## Input vectors X and Y of shape Nxd and Mxd
        distance_matrix = np.sum(X**2, axis = 1, keepdims = True) + np.sum(Y**2, axis = 1, keepdims = False) - 2 * np.dot(X, Y.T) # N,M matrix
        kernel_matrix = np.exp(- self.gamma * distance_matrix)
        return  kernel_matrix  ## Matrix of shape NxM


class Linear:
    def kernel(self,X,Y):
        ## Input vectors X and Y of shape Nxd and Mxd
        kernel_matrix = np.dot(X, Y.T)
        return kernel_matrix  ## Matrix of shape NxM
    

class Polynomial:
    def __init__(self, gamma=1., coef0 = 0):
        self.gamma = gamma 
        self.coef0 = coef0
    def kernel(self,X,Y):
        ## Input vectors X and Y of shape Nxd and Mxd
        kernel_matrix = np.dot(X, Y.T)
        kernel_matrix = self.gamma * kernel_matrix + self.coef0
        kernel_matrix = np.power(kernel_matrix, self.gamma)
        return kernel_matrix  ## Matrix of shape NxM