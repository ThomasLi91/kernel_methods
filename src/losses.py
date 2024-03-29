import numpy as np

def logistic_loss(u):
    loss = np.log(1 + np.exp(-u))
    return loss.mean()

def SVM_hinge_loss(u):
    loss = np.max(0, 1 - u)
    return loss.mean()

def SVM2_hinge_loss_squared(u):
    loss = np.max(0, 1 - u)**2
    return loss.mean()

def boosting_loss(u):
    loss = np.exp(-u)
    return loss