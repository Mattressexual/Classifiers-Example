import numpy
import torch
import autograd
import autograd.numpy as np

# sigmoid function
sig  = np.tanh
sigp = lambda x : 1 - np.tanh(x)**2

# Loss function
def L(x, y, W, V, b, c):
	l = c + V @ sig(b + W @ x)

	return -l[y] + np.log(np.sum(np.exp(l)))

# Partial derivatives
def partial_derivatives(x, y, W, V, b, c):
	# Filling in some dummy values
	# THIS IS WHERE YOU WILL WRITE YOUR PARTIAL DERIVATIVES
	s = b + W @ x
	h = np.tanh(s)
	f = c + np.matmul(V, h)

	# dLdf: -e + g(f(x))
	dLdf = -f * y + np.exp(f) / np.sum(np.exp(f))

	# dLdc = dL/df * df/dc
	dLdc = dLdf * np.exp(f) / np.sum(np.exp(c))

	# dLdV = dL/df * htranspose
	h_transpose = np.transpose(h)
	dLdV = dLdf * h_transpose

	# dLdb = sig'(b + Wx) @ (Vtranspose * dLdf)
	sigp = lambda x : 1 - np.tanh(x)**2
	V_transpose = np.transpose(V)
	dLh = np.matmul(V_transpose, dLdf)
	dLdb = sigp(s) * dLh

	# dLdW = dL/df * df/dW
	x_transpose = np.transpose(x)
	sigp_of_s = sigp(s)
	dLdW = sigp_of_s * np.multiply(dLh, x_transpose)

	return dLdW, dLdV, dLdb, dLdc

# DO NOT REMOVE OR UNCOMMENT THIS LINE OF CODE
# setting random seed for reproducibility
seed = 356
np.random.seed(seed)

# Loading the input
x = np.load('nn_gradient_sample.npy')
# Number of input dimensions
dims_in  = x.shape[0]
# Setting label
y = np.array([2])
# Number of output dimensions
dims_out = 4

# Number of hidden units
dims_hid = 5

# Initializing weights
W = np.random.randn(dims_hid, dims_in)
b = np.random.randn(dims_hid, 1)
V = np.random.randn(dims_out, dims_hid)
c = np.random.randn(dims_out, 1)

# Computing partial derivatives
dLdW_pd, dLdV_pd, dLdb_pd, dLdc_pd = partial_derivatives(x, y, W, V, b, c)

np.set_printoptions(precision=6)

# print loss
loss = L(x, y, W, V, b, c)
print('Loss = %0.4f' % loss)

# Computing partial derivatives using autograd. L is the loss function and 5 is the position of the c
dLdc_autograd = autograd.grad(L, 5)
print('dLdc, Autograd\n', dLdc_autograd(x, y, W, V, b, c).T)
print("dLdc =\n", dLdc_pd)
print('dLdc, partial derivative\n', dLdc_pd.T)

# Computing partial derivatives using autograd. L is the loss function and 3 is the position of the V
dLdV_autograd = autograd.grad(L, 3)
print('dLdV, Autograd\n', dLdV_autograd(x, y, W, V, b, c))
print("dLdV =\n", dLdV_pd)
print('dLdV, partial derivative\n', dLdV_pd)

# Computing partial derivatives using autograd. L is the loss function and 4 is the position of the b
dLdb_autograd = autograd.grad(L, 4)
print('dLdb, Autograd\n', dLdb_autograd(x, y, W, V, b, c).T)
print("dLdb =\n", dLdb_pd)
print('dLdb, partial derivative\n', dLdb_pd.T)

# Computing partial derivatives using autograd. L is the loss function and 2 is the position of the W
dLdW_autograd = autograd.grad(L, 2)
# Due to space limitations we are only printing few values of W
to_print_rows = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
to_print_cols = np.array([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
print('dLdW, Autograd\n', dLdW_autograd(x, y, W, V, b, c)[to_print_rows, to_print_cols])
print("dLdW =\n", dLdW_pd)
print('dLdW, partial derivative\n', dLdW_pd[to_print_rows, to_print_cols])
