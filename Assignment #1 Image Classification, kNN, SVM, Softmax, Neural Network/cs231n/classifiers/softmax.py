import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)



  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = np.shape(X)[0]
  num_classes = W.shape[1]

  for i in range(num_train):
    fi = X[i].dot(W)
    loss_change = -(fi[y[i]])+np.log(np.sum(np.exp(fi)))
    loss += loss_change
    for j in range(num_classes):
      a = np.exp(fi[j])/np.sum(np.exp(fi))
      if j == y[i]:
        dW[:,j] -=X[i]
      dW[:,j] += a*X[i]

  loss /= num_train
  loss += reg * np.sum(W*W)
  dW = dW/num_train + 2*reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = np.shape(X)[0]
  num_classes = W.shape[1]

  f = X.dot(W)

  softmax_output = np.exp(f)/np.sum(np.exp(f),axis=1,keepdims=True) 
  loss += np.sum(-np.log(softmax_output[range(num_train), y]))
  loss /= num_train
  loss += reg * np.sum(W*W)

  softmax_output[[range(num_train), y]] -=1
  dW = (X.T).dot(softmax_output)
  dW = dW/num_train + 2*reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

