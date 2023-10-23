import numpy as np
from random import shuffle
import builtins


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) L2 regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    dW = np.zeros(W.shape) # initialize the gradient as zero
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0


    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1
            if margin >=0:
                dW[:, y[i]] = dW[:, y[i]] - X[i]
                dW[:,j] = dW[:,j] + X[i]
                loss += margin


    loss /= num_train
    dW /= num_train
    dW = dW + reg * 2 * W
    loss += reg * np.sum(W * W)
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW

def huber_loss_naive(W, X, y, reg):
    """
    Modified Huber loss function, naive implementation (with loops).
    Delta in the original loss function definition is set as 1.
    Modified Huber loss is almost exactly the same with the "Hinge loss" that you have
    implemented under the name svm_loss_naive. You can refer to the Wikipedia page:
    https://en.wikipedia.org/wiki/Huber_loss for a mathematical discription.
    Please see "Variant for classification" content.

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.z

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) L2 regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """

    ###############################################################################
    # TODO:                                                                       #
    # Complete the naive implementation of the Huber Loss, calculate the gradient #
    # of the loss function and store it dW. This should be really similar to      #
    # the svm loss naive implementation with subtle differences.                  #
    ###############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    dW = np.zeros(W.shape) # initialize the gradient as zero
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = (scores[j] - correct_class_score + 1)
            if margin >=0:
                dW[:, y[i]] = dW[:, y[i]] - 2*(margin)*X[i]
                dW[:,j] = dW[:,j] + 2*(margin)*X[i]

                loss += margin**2
            else:
                loss += -4*(margin)
                dW[:, y[i]] = dW[:, y[i]] + 4*X[i]
                dW[:,j] = dW[:,j] - 4*X[i]



    loss = loss/ num_train
    dW = dW/num_train
    dW = dW + reg * 2 * W
    loss += reg * np.sum(W * W)

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    scores = X.dot(W)
    correct_class_score = scores[np.arange(X.shape[0]), y][:,np.newaxis]
    margin = np.maximum(0, scores - correct_class_score + 1)
    margin[np.arange(X.shape[0]), y] = 0
    loss += np.sum(margin)/X.shape[0]
    loss += reg * np.sum(W * W)


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    indicies_no_zero=margin != 0
    margin[indicies_no_zero] =1
    margin[np.arange(X.shape[0]), y] = -1* np.sum(margin, 1)

    dW = ((X.T) @ (margin)/ X.shape[0])
    dW = dW +  reg*2*W

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss,dW

def huber_loss_vectorized(W, X, y, reg):
    """
    Structured Huber loss function, vectorized implementation.

    Inputs and outputs are the same as huber_loss_naive.
    """

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured Huber loss, storing the  #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero
    scores = X.dot(W)
    correct_class_score = scores[np.arange(X.shape[0]), y][:,np.newaxis]
    margin =   scores - correct_class_score + 1
    margin[np.arange(X.shape[0]), y] = 0
    loss = loss+ np.sum(margin[margin>= 0]**2)
    loss = loss+ np.sum(margin[margin<0]*(-4))
    loss = (loss/X.shape[0])  + reg* np.sum(W * W)

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    indicies_no_zero=  np.array(margin > 0)
    indicies_zero=np.array(margin < 0)

    margin2=np.array(margin)

    margin[indicies_zero] = -4
    margin[indicies_no_zero]= 0
    margin[np.arange(X.shape[0]), y] = 0
    margin[np.arange(X.shape[0]), y] = -1*np.sum(margin, 1)


    margin2[indicies_no_zero]= 2*margin2[indicies_no_zero]
    margin2[indicies_zero] = 0
    margin2[np.arange(X.shape[0]), y] = 0
    margin2[np.arange(X.shape[0]), y] = -1*np.sum(margin2, 1)

    dW = ( (X.T) @ (margin2+margin)  )/ X.shape[0]


    dW = dW + reg * 2 * W
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW

