import numpy as np
from random import shuffle

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


  N=X.shape[0]
  C=W.shape[1]
  D=W.shape[0]


  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################

  # computing the scores

  scores=np.dot(X,W)    # N x C
  #computing the loss function
  for i in range(N):
    correct_class_score=scores[i,y[i]]

    loss=loss+(-correct_class_score+np.log(sum([np.exp(elem) for elem in scores[i,:]])))

  loss=loss/N
  #converting y in OHE
  y_ohe=np.zeros((N,C))
  for i in range(N):
    y_ohe[i,y[i]]=1
  #computing the gradient
  for i in range(N):
    score_i=scores[i,:]
    
    y_hat=np.exp(score_i)/np.sum(np.exp(score_i))

    #computing the gradient

    grad=(y_hat-y_ohe[i,:])

    #backpropogating the grad

    grad=grad.reshape(1,C)

    X_i=X[i,]

    X_i=X_i.reshape(1,D)

    temp_W=np.dot(X_i.T,grad)

    # print('temp_W'),temp_W

    dW=dW+temp_W

  dW=dW/N








  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

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
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores=np.dot(X,W)

  correct_class_score=scores(np.arange(N),y)

  loss=np.mean(np.log(np.exp(correct_class_score)/np.sum(np.exp(scores),axis=1)))

  

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

