import numpy
import theano

import theano.tensor as T

from collections import OrderedDict
from sklearn.datasets import fetch_mldata
rng = numpy.random

# Load data
mnist = fetch_mldata("MNIST original")
train_X, train_y = mnist.data[:60000, :], mnist.target[:60000]
test_X, test_y = mnist.data[60000:, :], mnist.target[60000:]

n_features = 784

y = T.ivector('y')
x = T.dmatrix('x')

n_epochs = 10

# Create a three layer network

# Layer 1
l1_dim = (n_features, n_features)
w1 = theano.shared(rng.normal(size=l1_dim), name='w1')
b1 = theano.shared(0., name='b1')
lf1 = T.nnet.relu(T.dot(x.T, w1) + b1)

# Layer 2
l2_dim = (n_features, n_features)
w2 = theano.shared(rng.normal(size=l2_dim), name='w2')
b2 = theano.shared(0., name='b2')
lf2 = T.nnet.relu(T.dot(lf1, w2) + b2)

# Layer 3
l3_dim = (n_features, n_features)
w3 = theano.shared(rng.normal(size=l3_dim), name='w3')
b3 = theano.shared(0., name='b3')
lf3 = T.nnet.relu(T.dot(lf2, w3) + b3)

# Output layer
o_dim = (n_features, 10)
w4 = theano.shared(rng.normal(size=o_dim), name='w4')
b4 = theano.shared(0., name='b4')
output = T.nnet.softmax(T.dot(lf3, w4) + b4)
prediction = T.argmax(output, axis=1)
loss = T.mean(T.nnet.categorical_crossentropy(output, y))

# Update rules
# Since we have a number
alpha = T.iscalar('alpha')
param_updates = OrderedDict()
for a in [w1, b1, w2, b2, w3, b3, w4, b4]:
    param_updates[a] = a + alpha * T.grad(loss, a)



obj_fn = theano.function(inputs=[x, y, alpha],
                         outputs=loss,
                         updates=param_updates,
                         allow_input_downcast=True)

pred_fn = theano.function(inputs=[x],
                          outputs=prediction,
                          allow_input_downcast=True)

log_interval = 1000
for i in range(n_epochs):
    count = 0
    for (data_x, data_label) in zip(train_X, train_y):
        cost = obj_fn(data_x.reshape((784,1)), data_label.reshape((1)), 0.01/numpy.sqrt(count+1+i*len(train_y)))
        if count % log_interval == 0:
            print("Cost at iteration {} epoch {} is {}".format(
                count, i, cost))
        count += 1
