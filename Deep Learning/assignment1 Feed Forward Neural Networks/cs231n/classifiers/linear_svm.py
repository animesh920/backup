import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops)
  Inputs:
  - W: C x D array of weights
  - X: D x N array of data. Data are D-dimensional columns
  - y: 1-dimensional array of length N with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  

  
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[0]
  num_train = X.shape[1]
  loss = 0.0
  for i in xrange(num_train):
    train=X[i,:]
    # train=train.reshape(1,-1)   #  1 x D
    print train.shape
    print W.shape
    
    scores = np.dot(train,W)    # 1 x C

    print scores.shape

    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        temp_score=scores
        temp_score=np.delete(temp_score,y[i])
        predicted_class_weight=W[y[i],:]
        predicted_class_weight=predicted_class_weight.reshape(-1,1)
        temp_score=temp_score-(np.dot(predicted_class_weight.T,train))+1
        num_violators=np.sum(temp_score>0)
        
        dW[j,:]=dW[j,:]-num_violators*train.T
        
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        # dW[j,:]=dW[j,:]+train.T
        loss += margin

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  
  dW/=num_train

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  
  
  
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  num_train=X.shape[0]
  num_classes=W.shape[1]
    
     
    
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  scores=np.dot(X,W)   #  C x N
  
  # print scores.shape  
  # print y.shape
  # print y
  
  #indexes=np.array([[i,elem] for i,elem in zip(range(num_train,y))])
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  
  correct_scores=scores[np.arange(num_train),y]
  
  #for j in range(num_train):
  #    correct_scores[j]=scores[y[j],j]
  
  # correct_scores=correct_scores.reshape(1,-1)
          
  # margin=(scores-(correct_scores))+1
  margin=((scores.T-correct_scores).T)+1
  
  margin[np.arange(num_train),y]=0
  
  #for j in range(num_train):
  #    margin[y[j],j]=0
  #
  #print margin[margin>1]
  
  loss=np.sum(margin[margin>0])
  loss=loss/num_train
  
  regularization_term=reg*np.sum(np.power(W,2))
  loss=loss+regularization_term


  #caculating the number of classes that have violated the margin across the entire training
  #data
  thres=np.maximum(np.zeros((num_train,num_classes)),margin)
  
  binary=thres
  binary[thres>0]=1
  
  col_sum=np.sum(binary,axis=1)

  # print 'col_sum',col_sum.shape

  # print 'binary',binary.shape
  
  binary[np.arange(num_train),y]=-col_sum[range(num_train)]
  
  # print 'binary',binary.shape

  dW=np.dot(X.T,binary)
  dW=dW/num_train
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
