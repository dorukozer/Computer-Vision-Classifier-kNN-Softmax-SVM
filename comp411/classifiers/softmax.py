import numpy as np
from random import shuffle
import builtins

def softmax_loss_naive(W, X, y, reg_l2, reg_l1 = 0):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg_l2: (float) regularization strength for L2 regularization
    - reg_l1: (float) default: 0. regularization strength for L1 regularization 
                to be used in Elastic Net Reg. if supplied, this function uses Elastic
                Net Regularization.

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
##############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.      #
    # Store the loss in loss and the gradient in dW. If you are not careful      #
    # here, it is easy to run into numeric instability. Don't forget the         #
    # regularization! If regtype is set as 'L2' just implement L2 Regularization #
    # else implement both L2 and L1.                                             #
    ##############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_classes = W.shape[1]
    num_train = X.shape[0]
    dW = np.zeros_like(W)
    scores = X @ W

    for i in range(num_train):
        normalized_scores = scores[i] - np.max(scores[i])
        numerator = np.exp(normalized_scores)
        denominator = (np.sum(np.exp(normalized_scores)))
        softmax = (numerator/denominator)
        loss += -np.log(softmax[y[i]])
        for j in range(num_classes):
            if j != y[i]:
                dW[:,j] += softmax[j] * X[i]
            else:
                dW[:,j] += ((softmax[j]) * X[i] )-X[i]

    loss /= num_train
    dW /= num_train

    regtype = None
    if reg_l1 == 0:
        regtype = 'L2'
        loss += reg_l2 * np.sum(W * W)
        dW += reg_l2 * 2 * W
    else:
        regtype = 'ElasticNet'
        loss += reg_l2 * np.sum(W * W)
        loss += reg_l1 * np.sum(W)
        dW += reg_l2 * 2 * W
        dW += reg_l1 *(W/W)
    pass


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg_l2, reg_l1 = 0):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.

    dW = np.zeros_like(W)
    regtype = None
    num_train = X.shape[0]


##############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.   #
    # Store the loss in loss and the gradient in dW. If you are not careful      #
    # here, it is easy to run into numeric instability. Don't forget the         #
    # regularization! If regtype is set as 'L2' just implement L2 Regularization #
    # else implement both L2 and L1.                                             #
    ##############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    scores = X @ W
    normalized_scores = scores - np.max(scores,axis=1)[:,np.newaxis]
    numerator = np.exp(normalized_scores)
    denominator = (np.sum(np.exp(normalized_scores),1))[:,np.newaxis]
    softmax = numerator/denominator
    loss= np.sum( -np.log (softmax[np.arange(num_train),y]))


    softmax[np.arange(num_train),y]-=1
    dW = ( X.T @ softmax)


    loss =loss /  num_train
    dW = dW/ num_train

    regtype = None
    if reg_l1 == 0:
        regtype = 'L2'
        loss += reg_l2 * np.sum(W * W)
        dW += reg_l2 * 2 * W
    else:
        regtype = 'ElasticNet'
        loss += reg_l2 * np.sum(W * W)
        loss += reg_l1 * np.sum(W)
        dW += reg_l2 * 2 * W
        dW += reg_l1 *(W/W)


    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
