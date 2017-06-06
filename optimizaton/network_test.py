import numpy
import theano
import matplotlib.pyplot as plt
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

# Number of hidden nodes
n_hidden = 3

def sgd(cost, params, lr):
    gradients = T.grad(cost=cost, wrt=params)
    return [(a, a - lr * g) for (a, g) in zip(params, gradients)]

# Layer 1
l1_dim = (n_features, n_hidden)
w1 = theano.shared(rng.normal(size=l1_dim), name='w1')
b1 = theano.shared(0., name='b1')
lf1 = T.nnet.relu(T.dot(x, w1) + b1)

# Layer 2
l2_dim = (n_hidden, n_hidden)
w2 = theano.shared(rng.normal(size=l2_dim), name='w2')
b2 = theano.shared(0., name='b2')
lf2 = T.nnet.relu(T.dot(lf1, w2) + b2)

# Layer 3
l3_dim = (n_hidden, n_hidden)
w3 = theano.shared(rng.normal(size=l3_dim), name='w3')
b3 = theano.shared(0., name='b3')
lf3 = T.nnet.relu(T.dot(lf2, w3) + b3)

# Output layer
o_dim = (n_hidden, 10)
w4 = theano.shared(rng.normal(size=o_dim), name='w4')
b4 = theano.shared(0., name='b4')
output = T.nnet.softmax(T.dot(lf3, w4) + b4)
prediction = T.argmax(output, axis=1)
loss = T.mean(T.nnet.categorical_crossentropy(output, y))

# Update rules
# Since we have a number
alpha = T.iscalar('alpha')
params = [w1, b1, w2, b2, w3, b3, w4, b4]
param_updates = [(a, a - alpha * T.grad(cost=loss, wrt=a)) for a in [w1, b1, w2, b2, w3,
    b3, w4, b4]]

obj_fn = theano.function(inputs=[x, y, alpha],
                         outputs=loss,
                         updates=sgd(loss, params, alpha),
                         allow_input_downcast=True)

pred_fn = theano.function(inputs=[x],
                          outputs=prediction,
                          allow_input_downcast=True)

def get_test_error():
    total_loss = 0
    for (td_x, td_y) in zip(test_X, test_y):
        pred = pred_fn(td_x.reshape((784,1)).T)
        if pred != td_y:
            total_loss += 1
    print("{} out of {} correct".format(total_loss, len(test_y)))
    return total_loss/len(test_y)

batch_size = 30
log_interval = len(train_y) / batch_size / 10
# Train network
test_err = []
iter_num = []
i_num = 0
for i in range(n_epochs):
    count = 0
    for (start, end) in zip(range(0, len(train_y) - batch_size, batch_size), range(batch_size,
        len(train_y), batch_size)):
        learning_rate = 0.1 / numpy.sqrt(count + 1 + i*len(train_y))
        cost = obj_fn(train_X[start:end,:],
                train_y[start:end],learning_rate)
        if count % log_interval == 0:
            print("Cost at iteration {} epoch {} is {} lr {:.6}".format(
                count, i, cost, learning_rate))
            test_err.append(get_test_error())
            iter_num.append(i_num)
            i_num += 1
        count += 1

plt.plot(iter_num, test_err)
plt.savefig("training_error.png", dpi=700)
