# Import python modules
import numpy
import kaggle
import torch
import autograd
import autograd.numpy as np
from autograd.misc import flatten
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import time
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import validation_curve

# Read in train and test data
def read_image_data():
	print('Reading image data ...')
	train_x = np.load('../../Data/data_train.npy')
	train_y = np.load('../../Data/train_labels.npy')
	test_x = np.load('../../Data/data_test.npy')

	return (train_x, train_y, test_x)

############################################################################

def read_image_data_knn():
	print('Reading image data ...')
	train_x = np.load('../../Data/data_train_knn.npy')
	train_y = np.load('../../Data/train_labels_knn.npy')
	test_x = np.load('../../Data/data_test.npy')

	return (train_x, train_y, test_x)

############################################################################

train_x, train_y, test_x = read_image_data()
print('Train=', train_x.shape)
print('Test=', test_x.shape)

# Create dummy test output values to compute accuracy
test_y = np.ones(test_x.shape[0])
predicted_y = np.random.randint(0, 4, test_x.shape[0])
print('DUMMY Accuracy=%0.4f' % accuracy_score(test_y, predicted_y, normalize=True))

# Select the section to run by changing the associated boolean to True
# Code for 2b
run2 = False
# Code for 3b
run3 = False
# Code for 4a
run4 = False
# Code for 5a
run5a = True
# Code for 5b
run5b = True


# 2b) Decision Tree Classifier
if run2:
	depths_2b = [3, 6, 9, 12, 14]
	means_2b = []
	classifiers_2b = []

	for d in depths_2b:
		clf = DecisionTreeClassifier(random_state = 0, max_depth = d)
		scores = cross_val_score(clf, train_x, train_y, cv = 5, verbose = 10, n_jobs = -1)
		mean = sum(scores) / len(scores)
		means_2b.append(mean)

	best_2b = means_2b.index(max(means_2b))
	best_depth = depths_2b[best_2b]
	print("Best depth:", best_depth)
	clf = DecisionTreeClassifier(random_state = 0, max_depth = best_depth)
	clf.fit(train_x, train_y)
	predicted_y_2b = clf.predict(test_x)

	file_name_2b = '../Predictions/2_best.csv'
	print('Writing output to ', file_name_2b)
	kaggle.kaggleize(predicted_y_2b, file_name_2b)

	fig = go.Figure(data = [go.Table(
		header = dict(values = ['Depth 3', 'Depth 6', 'Depth 9', 'Depth 12', 'Depth 14']),
		cells = dict(values = means_2b)
	)])
	fig.show()

# 3b) K Nearest Neighbors
if run3:
	ktrain_x, ktrain_y, ktest_x = read_image_data_knn()
	neighbors_3b = [3, 5, 7, 9, 11]
	means_3b = []
	for n in neighbors_3b:
		clf = KNeighborsClassifier(n_neighbors = n)
		scores = cross_val_score(clf, train_x, train_y, cv = 5, verbose = 10, n_jobs = 5)
		mean = sum(scores) / len(scores)
		means_3b.append(mean)
		print("Neighbors", n, "Scores:", scores)

	best_3b = means_3b.index(max(means_3b))
	best_n_neighbors = neighbors_3b[best_3b]
	print("Best score:", max(means_3b))
	print("Best n_neighbors:", best_n_neighbors)

	clf = KNeighborsClassifier(n_neighbors = best_n_neighbors)
	clf.fit(train_x, train_y)
	predicted_y_3b = clf.predict(test_x)

	file_name_3b = '../Predictions/3_best.csv'
	kaggle.kaggleize(predicted_y_3b, file_name_3b)

	fig = go.Figure(data = [go.Table(
		header = dict(values = ['k = 3', 'k = 5', 'k = 7', 'k = 9', 'k = 11']),
		cells = dict(values = means_3b)
	)])
	fig.show()

# 4a) SGD Classifier
if run4:
	alpha = [10**-6, 10**-4, 10**-2, 1, 10]
	means_4a_hinge = []
	means_4a_log = []
	fit_times = []
	for a in alpha:
		start = time.time()
		clf1 = SGDClassifier(alpha = a, loss = 'hinge')
		scores1 = cross_val_score(clf1, train_x, train_y, cv = 5, verbose = 10, n_jobs = -1)
		means_4a_hinge.append(sum(scores1) / len(scores1))

		clf2 = SGDClassifier(alpha = a, loss = 'log')
		scores2 = cross_val_score(clf2, train_x, train_y, cv = 5, verbose = 10, n_jobs = -1)
		means_4a_log.append(sum(scores2) / len(scores2))

		fit_times.append(time.time() - start)

	best_4a_hinge = means_4a_hinge.index(max(means_4a_hinge))
	best_4a_log = means_4a_log.index(max(means_4a_log))
	best_alpha_hinge = alpha[best_4a_hinge]
	best_alpha_log = alpha[best_4a_log]

	clf1 = SGDClassifier(loss = 'hinge', alpha = best_alpha_hinge)
	clf2 = SGDClassifier(loss = 'log', alpha = best_alpha_log)
	clf1.fit(train_x, train_y)
	clf2.fit(train_x, train_y)
	predicted_y_4a_hinge = clf1.predict(test_x)
	predicted_y_4a_log = clf2.predict(test_x)

	print("Best hinge mean score:", max(means_4a_hinge))
	print("Best log mean score:", max(means_4a_log))
	print("Best hinge alpha:", best_alpha_hinge)
	print("Best log alpha:", best_alpha_log)

	file_name_4a_hinge = '../Predictions/4_hinge_best.csv'
	file_name_4a_log = '../Predictions/4_log_best.csv'
	kaggle.kaggleize(predicted_y_4a_hinge, file_name_4a_hinge)
	kaggle.kaggleize(predicted_y_4a_log, file_name_4a_log)

	fig = go.Figure(data = [go.Table(
		header = dict(values = ['alpha = 10^-6', 'alpha = 10^-4', 'alpha = 10^-2', 'alpha = 1', 'alpha = 10']),
		cells = dict(values = means_4a_hinge)
	)])
	fig.show()

	fig = go.Figure(data = [go.Table(
		header = dict(values = ['alpha = 10^-6', 'alpha = 10^-4', 'alpha = 10^-2', 'alpha = 1', 'alpha = 10']),
		cells = dict(values = means_4a_log)
	)])
	fig.show()

# 5a) Partial Derivatives

# Functions for 5a

# Loss function
def L(x, y, W, V, b, c):
	sig = np.tanh
	l = c + V @ sig(b + W @ x)

	return -l[y] + np.log(np.sum(np.exp(l)))

# Partial derivatives
def partial_derivatives(x, y, W, V, b, c):
	# Filling in some dummy values
	# THIS IS WHERE YOU WILL WRITE YOUR PARTIAL DERIVATIVES
	s = b + W @ x
	h = np.tanh(s)
	f = c + V @ h
	eHat = np.zeros(c.shape)
	eHat[y] = 1

	# dLdf: -e + g(f(x))
	dLdf = -eHat + np.exp(f) / np.sum(np.exp(f))

	# dLdc = dL/df * df/dc
	dLdc = -eHat + (np.exp(f) / np.sum(np.exp(f)))

	# dLdV = dL/df * htranspose
	h_transpose = np.transpose(h)
	dLdV = dLdf * h_transpose

	# dLdb = sig'(b + Wx) elementwise mult (Vtranspose * dLdf)
	sigp = lambda x : 1 - np.tanh(x)**2
	V_transpose = np.transpose(V)
	dLh = V_transpose @ dLdf
	dLdb = sigp(s) * dLh

	# dLdW = dL/df * df/dW
	x_transpose = np.transpose(x)
	dLdW = dLdb * x_transpose

	return dLdW, dLdV, dLdb, dLdc

if run5a:
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
	print('dLdc, partial derivative\n', dLdc_pd.T)

	# Computing partial derivatives using autograd. L is the loss function and 3 is the position of the V
	dLdV_autograd = autograd.grad(L, 3)
	print('dLdV, Autograd\n', dLdV_autograd(x, y, W, V, b, c))
	print('dLdV, partial derivative\n', dLdV_pd)

	# Computing partial derivatives using autograd. L is the loss function and 4 is the position of the b
	dLdb_autograd = autograd.grad(L, 4)
	print('dLdb, Autograd\n', dLdb_autograd(x, y, W, V, b, c).T)
	print('dLdb, partial derivative\n', dLdb_pd.T)

	# Computing partial derivatives using autograd. L is the loss function and 2 is the position of the W
	dLdW_autograd = autograd.grad(L, 2)
	# Due to space limitations we are only printing few values of W
	to_print_rows = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
	to_print_cols = np.array([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
	print('dLdW, Autograd\n', dLdW_autograd(x, y, W, V, b, c)[to_print_rows, to_print_cols])
	print('dLdW, partial derivative\n', dLdW_pd[to_print_rows, to_print_cols])

# 5b) Neural Network Classifiers
if run5b:
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

	# Number of train examples
	nTrainSamples = train_x.shape[0]
	# Number of input dimensions
	dims_in = train_x.shape[1]

	# Number of output dimensions
	dims_out = 4
	# Number of hidden units
	M = [5, 40, 70]
	# Learning rate
	epsilon = 0.0001
	# Momentum of gradients update
	momentum = 0.1
	# Number of epochs
	nEpochs = 1000

	# Convert integer labels to one-hot vectors
	# i.e. convert label 2 to 0, 0, 1, 0
	train_y = np.zeros((nTrainSamples, dims_out))
	train_y[np.arange(nTrainSamples), train_y_integers] = 1

	#Create values and labels for bar chart
	values = np.random.rand(3)
	inds   = np.arange(3)
	labels = ["M = 5","M = 40","M = 70"]

	losses = []
	fit_times = []
	scores = []
	for m in M:
		start = time.time()
		dims_hid = m
		print("Hidden layer size:", dims_hid)

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
		loss = []
		errors = []
		for i in range(0, nEpochs):
			# Compute gradients (partial derivatives) using autograd toolbox
			weight_gradients, returned_values = grad_fun(weights, train_x, train_y, unflatten)
			#print('logistic loss: ', returned_values[0], 'Train error =', returned_values[1])
			loss.append(returned_values[0])
			errors.append(returned_values[1])
			# Update weight vector
			smooth_grad = (1 - momentum) * smooth_grad + momentum * weight_gradients
			weights = weights - epsilon * smooth_grad

		losses.append(loss)
		score = 1-mean_zero_one_loss(weights, train_x, train_y_integers, unflatten)
		print('Train accuracy =', score)
		scores.append(score)
		fit_t = time.time() - start
		print("Fit time for hidden layer size", m, ":", fit_t)
		fit_times.append(fit_t)

		(W, b, V, c) = unflatten(weights)
		out = feedForward(W, b, V, c, test_x)
		pred = np.argmax(out, axis = 1)
		file_name = '../Predictions/bestNN_' + str(m) + '.csv'
		print('Writing output to ', file_name)
		kaggle.kaggleize(pred, file_name)


	#Create values and labels for line graphs
	values =np.random.rand(2,5)
	inds   =np.arange(5)
	labels =["M = 5", "M = 40", "M = 70"]

	#Plot a line graph
	plt.figure(2, figsize=(6,4))  #6x4 is the aspect ratio for the plot
	plt.plot(inds,values[0,:],'or-', linewidth=3) #Plot the first series in red with circle marker
	plt.plot(inds,values[1,:],'sb-', linewidth=3) #Plot the first series in blue with square marker

	#This plots the data
	plt.grid(True) #Turn the grid on
	plt.ylabel("Error") #Y-axis label
	plt.xlabel("Value") #X-axis label
	plt.title("Error vs Value") #Plot title
	plt.xlim(-0.1, 4.1) #set x axis range
	plt.ylim(0, 1) #Set yaxis range
	plt.legend(labels, loc = "best")

	x_axis = np.linspace(1, 1000, num = 1000)
	for i in range(0, len(losses)):
		plt.plot(x_axis, losses[i], label = 'M = ' + str(M[i]))

	#Save the line plot
	plt.savefig("../Figures/example_line_plot.pdf")

	# Table
	fig = go.Figure(data = [go.Table(
		header = dict(values = ['M = 5', 'M = 40', 'M = 70']),
		cells = dict(values = scores)
	)])
	fig.show()

	"""
	I used this to predict for my Kaggle submission.
	
	M = [5, 40, 70]
	means = []

	epochs = 1000
	fixed_learning_rate = 0.0001
	fixed_momentum = 0.1
	fixed_penalty = 10

	# For each hidden layer size
	for m in M:
		# Create classifier with required parameters
		clf = MLPClassifier(
			random_state = 0,
			hidden_layer_sizes = (m,),
			activation = 'tanh',
			alpha = fixed_penalty,
			learning_rate_init = fixed_learning_rate,
			momentum = fixed_momentum,
			max_iter = epochs,
			verbose = False)
	
		# Plot curve
		plt.plot(train_s, label='Training scores of M =' + str(m))
		plt.plot(valid_s, label='Validation scores of M =' + str(m))

		scores = cross_val_score(
			clf,
			train_x,
			train_y,
			n_jobs = -1)

		mean = sum(scores) / len(scores)
		means.append(mean)
		print("Hidden layer size:", m)
		print("Scores:", scores)
		print("Mean:", mean)

	best_5b = means.index(max(means))
	best_m = M[best_5b]

	print("Best M:", best_m)
	print("Best mean score:", means[best_5b])
	print("max of means:", max(means))
	
	clf = MLPClassifier(
			random_state = 0,
			hidden_layer_sizes = (best_m,),
			activation = 'tanh',
			alpha = fixed_penalty,
			learning_rate_init = fixed_learning_rate,
			momentum = fixed_momentum,
			max_iter = epochs,
			verbose = False)
	clf.fit(train_x, train_y)
	predicted_y_5b = clf.predict(test_x)
	file_name_5b = '../Predictions/5_best.csv'
	kaggle.kaggleize(predicted_y_5b, file_name_5b)
	"""
