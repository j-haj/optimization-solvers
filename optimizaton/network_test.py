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

class Layer(object):
    """
    Creates network layers with specified shape and activation function.

    A layer consists of a specified number of nodes, a speified input size, and
    a specified output size. Each node has a set of weights and constant offset,
    along with an activation function.
    """

    def __init__(self, input_dim, output_dim, n_nodes, activation=T.nnet.relu,
                 weight_init_scale=0.01):
        """
        Constructor

        Params:
            input_dim (int): dimension of layer input
            output_dim (int): dimension of layer output
            n_nodes (int): number of nodes in the layer
            activation: activation function used for the nodes
            weight_init_scale: scale factor for random weight initialization.
                               Default value is 0.01
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight_init_scale_ = weight_init_scale
        self.activation = activation

        self.W = theano.shared(value=rng.normal(size=(input_dim, output_dim),
                                                scale=self.weight_init_scale_),
                              borrow=True)
        self.b = theano.shared(value=self.weight_init_scale_, borrow=True)

    def eval(self, x):
        """
        Evaluates the layer.

        Layer evaluation is based on the formulat

            activation(<w.T, x> + b)

        Params:
            x: input tensor to be run through the layer

        Returns:
            Result of activation function
        """
        return self.activation(T.dot(w.T, x) + b)

    def params(self):
        """
        Returns the Theano parameters as a list

        Returns:
            List containing W and b
        """
        return [self.W, self.b]

n_epochs = 10

# Number of hidden nodes
n_hidden = 450

def sgd(cost, params, lr, step):
    gradients = T.grad(cost=cost, wrt=params)
    returned_updates = [(a, a - lr * g) for (a, g) in zip(params, gradients)]
    returned_updates.append((lr, alpha_orig / np.sqrt(step)))
    returned_updates.append((step, step + 1))
    return returned_updates


history_size = 5
B = None
prior_gradients = None
def lbfgs(x, y, params, cost, B):
    """
    See https://en.wikipedia.org/wiki/Limited-memory_BFGS
    for details on this algorithm. Algorithm taken from this resource
    """
    # Perform warm start if t < history_size
    
    gradients = theano.gradient.jacobian(cost, wrt=params)
    if B is None:
        B = theano.shared(value=[np.identity(n=i.shape.eval()[0],
            m=i.shape.eval()[1], dtype=float) for i in gradients],
                          name='B', borrow=True)
        prior_gradients = gradients
    # Get line search direction
    p = -T.dot(T.nlinalg.MatrixInverse(B), gradients)

    # Perform line search
    step_size = 0.01
    num_steps = 100
    step_vals = [step_size * i for i in range(num_steps)]
    obj_vals = []
    for step in step_vals:
        new_params = params.copy() + step * p
        mod_obj = obj_fn.copy(swap={params: new_params})
        obj_vals.append(mod_obj(x, y))
    best_step = step_vals[obj_vals.index(min(obj_vals))]
    sk = best_step * p
    params = params + sk
    yk = T.grad(cost=cost, wrt=params) - prior_gradient
    B += T.dot(yk, yk.T) / T.dot(yk.T, sk) -\
            T.dot(T.dot(B, sk), T.dot(sk.T, B))/\
            T.dot(T.dot(sk.T, B), sk)

    # Update y and B
    return params

def softmax(X):
    e_x = T.exp(X - X.max(axis=1).dimshuffle(0, 'x'))
    return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')

weight_scale = 0.01

# Layer 1
l1_dim = (n_features, n_hidden)
w1 = theano.shared(value=rng.normal(size=l1_dim, scale=weight_scale),
                   name='w1', borrow=True)
b1 = theano.shared(value=weight_scale, name='b1', borrow=True)
lf1 = T.nnet.relu(T.dot(x, w1) + b1)

# Layer 2
l2_dim = (n_hidden, n_hidden)
w2 = theano.shared(value=rng.normal(size=l2_dim, scale=weight_scale),
                   name='w2', borrow=True)
b2 = theano.shared(value=weight_scale, name='b2', borrow=True)
lf2 = T.nnet.relu(T.dot(lf1, w2) + b2)

# Layer 3
l3_dim = (n_hidden, n_hidden)
w3 = theano.shared(value=rng.normal(size=l3_dim, scale=weight_scale),
                   name='w3', borrow=True)
b3 = theano.shared(value=weight_scale, name='b3', borrow=True)
lf3 = T.nnet.relu(T.dot(lf2, w3) + b3)

# Output layer
o_dim = (n_hidden, 10)
w4 = theano.shared(rng.normal(size=o_dim, scale=weight_scale),
                   name='w4', borrow=True)
b4 = theano.shared(value=weight_scale, name='b4', borrow=True)
output = softmax(T.dot(lf3, w4) + b4)
prediction = T.argmax(output, axis=1)
loss = T.mean(T.nnet.categorical_crossentropy(output, y))

# Update rules
# Since we have a number
alpha_val = 0.05
alpha = theano.shared(value=alpha_val, name='alpha')
alpha_orig = theano.shared(value=alpha_val, name='alpha_orig')
t = theano.shared(value=1, name='t')
params = [w1, b1, w2, b2, w3, b3, w4, b4]

# Gradient descent
param_updates = [(a, a - alpha * T.grad(cost=loss, wrt=a)) for a in params]

# Update learning rate
param_updates.append((alpha, alpha_orig / np.sqrt(t)))
param_updates.append((t, t + 1))
param_updates.append((alpha_orig, alpha_orig))


obj_fn = theano.function(inputs=[x, y],
                         outputs=loss,
                         #updates=sgd(loss, params, alpha, t),
			             updates=lbfgs(x, y, params, loss, B),
                         allow_input_downcast=True)

pred_fn = theano.function(inputs=[x],
                          outputs=prediction,
                          allow_input_downcast=True)


batch_size = 10
log_interval = len(train_y) / batch_size / 10

# Train network
test_err = []
iter_num = []
i_num = 0
for i in range(n_epochs):
    count = 0
    for (start, end) in zip(range(0, len(train_y) - batch_size, batch_size),
            range(batch_size, len(train_y), batch_size)):
        cost = obj_fn(train_X[start:end,:],
                train_y[start:end])
    
    print("Epoc {} test accuracy: {}".format(i, np.mean(test_y == pred_fn(test_X))))

#plt.plot(iter_num, test_err)
#plt.savefig("training_error.png", dpi=700)
