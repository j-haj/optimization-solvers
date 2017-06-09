import numpy as np
import theano
import matplotlib.pyplot as plt
import theano.tensor as T

from sklearn.datasets import fetch_mldata
rng = np.random

# Load data
mnist = fetch_mldata("MNIST original")
indices = list(range(70000))
np.random.shuffle(indices)
train_indices = indices[:60000]
test_indices = indices[60000:]
train_X, train_y = mnist.data[train_indices, :], mnist.target[train_indices]
test_X, test_y = mnist.data[test_indices, :], mnist.target[test_indices]
n_features = 784

y = T.ivector('y')
x = T.dmatrix('x')

n_epochs = 5 

# Number of hidden nodes
n_hidden = 50

def sgd(cost, params, lr):
    gradients = T.grad(cost=cost, wrt=params)
    return [(a, a - lr * g) for (a, g) in zip(params, gradients)]

def softmax(X):
    e_x = T.exp(X - X.max(axis=1).dimshuffle(0, 'x'))
    return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')

# Layer 1
l1_dim = (n_features, n_hidden)
w1 = theano.shared(rng.normal(loc=1.0, size=l1_dim), name='w1')
b1 = theano.shared(rng.normal(size=(n_hidden)), name='b1')
lf1 = T.nnet.relu(T.dot(x, w1) + b1)

# Layer 2
l2_dim = (n_hidden, n_hidden)
w2 = theano.shared(rng.normal(loc=1.0, size=l2_dim), name='w2')
b2 = theano.shared(rng.normal(size=(n_hidden)), name='b2')
lf2 = T.nnet.relu(T.dot(lf1, w2) + b2)

# Layer 3
l3_dim = (n_hidden, n_hidden)
w3 = theano.shared(rng.normal(loc=1.0, size=l3_dim), name='w3')
b3 = theano.shared(rng.normal(size=(n_hidden)), name='b3')
lf3 = T.nnet.relu(T.dot(lf2, w3) + b3)

# Output layer
o_dim = (n_hidden, 10)
w4 = theano.shared(rng.normal(loc=1.0, size=o_dim), name='w4')
b4 = theano.shared(rng.normal(size=(10)), name='b4')
output = softmax(T.dot(lf3, w4) + b4)
prediction = T.argmax(output, axis=1)
loss = T.mean(T.nnet.categorical_crossentropy(output, y))

# Update rules
# Since we have a number
alpha = theano.shared(0.01, name='alpha')
alpha_orig = theano.shared(0.01, name='alpha_orig')
t = theano.shared(1, name='t')
params = [w1, b1, w2, b2, w3, b3, w4, b4]

# Gradient descent
param_updates = [(a, a - alpha * T.grad(cost=loss, wrt=a)) for a in [w1, b1, w2, b2, w3,
    b3, w4, b4]]

# Update learning rate
param_updates.append((alpha, alpha_orig / np.sqrt(t)))
param_updates.append((t, t + 1))
param_updates.append((alpha_orig, alpha_orig))

obj_fn = theano.function(inputs=[x, y],
                         outputs=loss,
                         updates=param_updates,
                         allow_input_downcast=True)

pred_fn = theano.function(inputs=[x],
                          outputs=prediction,
                          allow_input_downcast=True)


batch_size = 100
log_interval = len(train_y) / batch_size / 10
# Train network
test_err = []
iter_num = []
i_num = 0
get_test_error()
for i in range(n_epochs):
    count = 0
    for (start, end) in zip(range(0, len(train_y) - batch_size, batch_size),
            range(batch_size, len(train_y), batch_size)):
        cost = obj_fn(train_X[start:end,:],
                train_y[start:end])
        if count % log_interval == 0:
            print("alpha: {}".format(alpha.get_value()))
            print("Cost at iteration {} epoch {} is {}".format(
                t.get_value(), i, cost))
            test_err.append(cost)
            iter_num.append(i_num)
            i_num += 1
        count += 1
    print("Test loss: {}".format(np.mean(test_y ==
        np.argmax(pred_fn(test_X), axis=0))))

plt.plot(iter_num, test_err)
plt.savefig("training_error.png", dpi=700)
