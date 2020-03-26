import numpy as np
import autograd.numpy
import autograd
from autograd.misc import flatten

# Function to compute classification accuracy
def mean_zero_one_loss(weights, x, y_integers, unflatten):
	(W, b, V, c) = unflatten(weights)
	out = feedForward(W, b, V, c, x)
	pred = np.argmax(out, axis=1)
	return(np.mean(pred != y_integers))

# Feed forward output i.e. L = -O[y] + log(sum(exp(O[j])))
def feedForward(W, b, V, c, train_x):
        hid = np.tanh(np.dot(train_x, W) + b)
        out = np.dot(hid, V) + c
        return out

# Logistic Loss function
def logistic_loss_batch(weights, x, y, unflatten):
	# regularization penalty
        lambda_pen = 10

        # unflatten weights into W, b, V and c respectively 
        (W, b, V, c) = unflatten(weights)

        # Predict output for the entire train data
        out  = feedForward(W, b, V, c, x)
        pred = np.argmax(out, axis=1)

	# True labels
        true = np.argmax(y, axis=1)
        # Mean accuracy
        class_err = np.mean(pred != true)

        # Computing logistic loss with l2 penalization
        logistic_loss = np.sum(-np.sum(out * y, axis=1) + np.log(np.sum(np.exp(out),axis=1))) + lambda_pen * np.sum(weights**2)
        
        # returning loss. Note that outputs can only be returned in the below format
        return (logistic_loss, [autograd.tracer.getval(logistic_loss), autograd.tracer.getval(class_err)])

# Loading the dataset
print('Reading image data ...')
temp = np.load('../../Data/data_train.npy')
train_x = temp
temp = np.load('../../Data/train_labels.npy')
train_y_integers = temp
temp = np.load('../../Data/data_test.npy')
test_x = temp
print("Data read")

# Make inputs approximately zero mean (improves optimization backprob algorithm in NN)
train_x = train_x.astype(float)
test_x = test_x.astype(float)

train_x -= .5
test_x  -= .5

# Number of output dimensions
dims_out = 4
# Number of hidden units
dims_hid = 5
# Learning rate
epsilon = 0.0001
# Momentum of gradients update
momentum = 0.1
# Number of epochs
nEpochs = 1000
# Number of train examples
nTrainSamples = train_x.shape[0]
# Number of input dimensions
dims_in = train_x.shape[1]

# Convert integer labels to one-hot vectors
# i.e. convert label 2 to 0, 0, 1, 0
train_y = np.zeros((nTrainSamples, dims_out))
train_y[np.arange(nTrainSamples), train_y_integers] = 1

assert momentum <= 1
assert epsilon <= 1

# Batch compute the gradients (partial derivatives of the loss function w.r.t to all NN parameters)
grad_fun = autograd.grad_and_aux(logistic_loss_batch)

# Initializing weights
W = np.random.randn(dims_in, dims_hid)
b = np.random.randn(dims_hid)
V = np.random.randn(dims_hid, dims_out)
c = np.random.randn(dims_out)
smooth_grad = 0

# Compress all weights into one weight vector using autograd's flatten
all_weights = (W, b, V, c)
weights, unflatten = flatten(all_weights)

# Compute gradients (partial derivatives) using autograd toolbox
weight_gradients, returned_values = grad_fun(weights, train_x, train_y, unflatten)
print('logistic loss: ', returned_values[0], 'Train error =', returned_values[1])

# Update weight vector
smooth_grad = (1 - momentum) * smooth_grad + momentum * weight_gradients
weights = weights - epsilon * smooth_grad

print('Train accuracy =', 1-mean_zero_one_loss(weights, train_x, train_y_integers, unflatten))