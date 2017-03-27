import numpy as np 
import math
import random

def sigmoid(x): 
	return 1.0/(1.0+np.exp(-x))

def sigmoid_prime(x):
     return x*(1.0-x)

def k_fold_cv(train_data, labels, k):
	paired = list(zip(train_data,labels))
	random.shuffle(paired)
	train_data, labels = zip(*paired)

	sum_acc = 0
	size = len(labels)/k

	for i in xrange(k):

		print "Cross Validate iteration: "+str(i+1)

		train_a = train_data[0:i*size] 
		train_b = train_data[i*size:(i+1)*size]
		train_c = train_data[(i+1)*size:] 

		labels_a = labels[0:i*size]
		labels_b = labels[i*size:(i+1)*size]
		labels_c = labels[(i+1)*size:]

		if i > 0:
			tr = np.vstack((train_a,train_c)) if i < k-1 else train_a
			lab = np.vstack((labels_a,labels_c)) if i < k-1 else labels_a
		else:
			tr = train_c
			lab = labels_c

		res = train(tr, lab, train_b, labels_b)
		sum_acc += res[2]
	return sum_acc/k

def accuracy(valid_data, valid_labels, weights, biases): 
	# FEEDFORWARD
	predictions = []
	for k,img in enumerate(valid_data):
		img = img.reshape(FEATURES,1)
		activations = [img]
		for i in xrange(HIDDEN_LAYERS+1):
			activations.append(sigmoid(np.dot(weights[i],activations[-1])+biases[i]))
		predictions.append(np.argmax(activations[-1]))

	#print predictions[:10]
	
	# ACCURACY
	correct = 0.0
	for i in xrange(len(valid_data)):
		if predictions[i] == np.argmax(valid_labels[i]):
			correct += 1
	return correct/len(predictions)

def train(train_data,train_labels,valid_data,valid_labels):
	# INITIALIZAION
	weights = [np.random.randn(HIDDEN_DIM,FEATURES)]
	for i in xrange(HIDDEN_LAYERS-1):
		weights.append(np.random.randn(HIDDEN_DIM,HIDDEN_DIM))
	weights.append(np.random.randn(OUTPUT_DIM,HIDDEN_DIM))

	biases = []
	for i in xrange(HIDDEN_LAYERS):
		biases.append(np.random.randn(HIDDEN_DIM,1))
	biases.append(np.random.randn(OUTPUT_DIM,1))

	# TRAINING
	for epoch in xrange(ITERATIONS):
		paired = list(zip(train_data,train_labels))
		random.shuffle(paired)
		train_data, train_labels = zip(*paired)

		for k,img in enumerate(train_data):

			y = train_labels[k].reshape(OUTPUT_DIM,1)		
			img = img.reshape(FEATURES,1)

			# FEEDFORWARD
			activations = [img]
			for i in xrange(HIDDEN_LAYERS+1):
				activations.append(sigmoid(np.dot(weights[i], activations[-1])+biases[i]))

			# BACKPROPAGATION
			delta = (activations[-1] - y) * sigmoid_prime(activations[-1])
			biases[-1] -= learning_rate * delta
			weights[-1] *= 1 - (learning_rate * lmbda/len(train_data)) #L2 Regularization
			weights[-1] -= learning_rate * np.dot(delta, activations[-2].T)
			for i in xrange(2,HIDDEN_LAYERS+2): 
				delta = np.dot(weights[-i+1].T, delta) * sigmoid_prime(activations[-i])
				biases[-i] -= learning_rate * delta
				weights[-i] *= 1 - (learning_rate * lmbda/len(train_data)) #L2 Regularization
				weights[-i] -= learning_rate * np.dot(delta, activations[-i-1].T)

		print "epoch "+str(epoch)+" complete"
		acc = accuracy(valid_data, valid_labels, weights, biases)
		print "accuracy: "+str(acc)

	return [weights,biases,acc]

#############################################################
np.random.seed(0)

# PARAMETERS
ITERATIONS = 100 		# Number of epochs the neural net trains for
HIDDEN_LAYERS = 1 		# Number of hidden layers
HIDDEN_DIM = 50 		# Dimension of each hidden layer
learning_rate = 0.05 	# Alpha #0.3
lmbda = 0.01			# Regularization term #0.1

FEATURES = 3600
OUTPUT_DIM = 19

train_x = np.fromfile('train_x.bin', dtype='uint8', count=100000*3600)
train_x = train_x.reshape((100000,3600))
train_y = np.genfromtxt('train_y.csv', delimiter=',', skip_header=1)

TRAINING = 90000
VALIDATION = 10000

X = train_x[:TRAINING,:]
V = train_x[TRAINING:TRAINING+VALIDATION,:]
ty = np.zeros((TRAINING+VALIDATION,OUTPUT_DIM))
for i in xrange(TRAINING+VALIDATION):
	ty[i][int(train_y[i][1])] = 1.0

Y = ty[:TRAINING,:]
VY = ty[TRAINING:TRAINING+VALIDATION,:]

#Thresholding
X[X <= 220.0] = 0
X[X > 220.0] = 1
V[V <= 220.0] = 0
V[V > 220.0] = 1

###########################################################################
#print "KFold average accuracy: "+str(k_fold_cv(X,Y,5))
train(X,Y,V,VY)
###########################################################################
