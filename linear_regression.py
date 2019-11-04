import numpy as np
import random


"""
  X - input data (numpy array)
  y - output labels (numpy array)
  epoch - No of training epochs
  lr - learning rate 
  
  Example:
      X = np.asarray([1,2,3,4,5])
      y = np.asarray([3,6,9,12,15])
"""
def train(X,y,epoch=100,lr=0.01):

    train_size = len(X)

    #randomly initialized weights
    w = random.randint(0,10)

    y_pred = {}
    L = {}
    dl = {}

    for i in range(epoch):
        y_pred[i] = w * X
        L[i] = (0.5/train_size)*np.square(np.subtract(y,y_pred[i]))
        dl[i] = (1/train_size)*np.dot(np.subtract(y_pred[i], y),X)
        w = w - (lr * dl[i])

        print('Prediction in epoch {}: {} '.format(i, y_pred[i]))
        print('Loss in epoch {}: {} '.format(i, L[i]))
        print('W updated in epoch {}: {} '.format( i, w),'')
