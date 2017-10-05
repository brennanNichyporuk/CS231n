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
  num_classes = W.shape[1]
  num_train = X.shape[0]
  for i in xrange(num_train):
    scores = X[i].dot(W) # (1xD)x(DxC)=(1xC)
    correct_class_exp_score = np.e ** scores[y[i]]
    
    exp_sum = 0
    for j in xrange(num_classes):
      exp_sum += np.e ** scores[j]

    for j in xrange(num_classes):
      dW[:,j] += (1 / exp_sum) * (np.e ** scores[j]) * X[i,:]
    
    loss += -np.log(correct_class_exp_score / exp_sum)
    dW[:,y[i]] -= X[i,:]
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW += reg * 2 * W
    
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
  num_classes = W.shape[1]
  num_train = X.shape[0]
  exp_scores = np.e ** X.dot(W) # (NxD)x(DxC)=(NxC)
  exp_sums = np.sum(exp_scores, axis=1) # (Nx1)
  
  correct_class_exp_scores = exp_scores[np.arange(y.shape[0]),y]
  loss = sum(-np.log(correct_class_exp_scores / exp_sums))
  
  y_matrix = np.zeros((num_train, num_classes)) # (NxC)
  y_matrix[np.arange(y.shape[0]),y] = 1

  exps = (1 / exp_sums[np.newaxis].T) * exp_scores # (Nx1)x(NxC)=(NxC) via broadcasting
  dW = np.dot(X.T,exps) # (DxN)x(NxC)=(DxC)
  dW -= np.dot(X.T, y_matrix) # (DxN)x(NxC)=(DxC)

  loss /= num_train
  loss += reg * np.sum(W * W)

  dW /= num_train
  dW += reg * 2 * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

