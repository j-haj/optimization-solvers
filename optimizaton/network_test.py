import numpy as np
import theano
import matplotlib.pyplot as plt
import theano.tensor as T
from enum import Enum
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

    def __init__(self, dim, activation=T.nnet.relu,
                 weight_init_scale=0.01):
        """
        Constructor

        Params:
            dim : tuple representing the input and output dimensions
                  (respectively) of the layer
            n_nodes (int): number of nodes in the layer
            activation: activation function used for the nodes
            weight_init_scale: scale factor for random weight initialization.
                               Default value is 0.01
        """
        self.dim = dim
        self.weight_init_scale_ = weight_init_scale
        self.activation = activation

        self.W = theano.shared(value=rng.normal(size=dim),
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


class NeuralNetwork(object):
    """
    Represents a neural network model through the use of layer composition.

    Neural Networks are composed via composition of layers and specifying a
    solver.
    """

    class Optimizer(Enum):
        SGD = "SGD"
        LBFGS = "L-BFGS"

    def __init__(self, loss=T.nnet.categorical_crossentropy, optimizer=None):
        """
        Constructor

        Params:
            loss: loss function (Default is categorical crossentropy)
            solver: solver used to train the network
        """
        self.loss_ = loss
        self.optimizer_ = optimizer
        self.layers = []
        self.params = []


    def add_layer(self, dim, activation=T.nnet.relu,
                  weight_init_scale=.01):
        """
        Creates a `Layer` object and adds it to the network

        Params:
            dim: tuple representing the input and output dimensions of the layer 
            activation: activation function. Default value is ReLU
            weight_init_scale: scale used in initializing layer weights. Default
                                is 0.01

        Returns:
            This method returns `self` to allow for the fluent style of method
            chaining.
        """
        # Check layer compaitability
        if len(self.layers) != 0:
            if self.layers[-1].output_dim != input_dim:
                raise ValueError(("Dimension mismatch - input dimension of"
                "current layer must match output dimension of prior layer"))

        # Create layer object
        self.layers.append(Layer(input_dim, output_dim, activation,
            weight_init_scale))
        self.params += self.layers[-1].params

        return self

    def optimizer_(self):
        """
        Returns the updates list for the specified optimizer

        Returns:
            List of tuples specifying update rules for model
        """
        if self.optimizer_ == Optimizer.SGD:
            # SGD
            
            # TODO: need a better way of allowing the user to pick the learning
            # rate
            alpha = theano.shared(value=0.01, borrow=True)
            t = theano.shared(value=1, borrow=True)
            updates = [(p, p - alpha * g) for (p, g) in zip(self.params,
                gradients)]
            updates.append((alpha, alpha * np.sqrt(t/(t + 1))))
            updates.append((t, t + 1))
            return updates
        
        elif self.optimizer_ == Optimizer.LBFGS:
            # L-BFGS
            pass
        return None



    def compile(self):
        """
        Creates a Theano function object to store the model definition
        """
        if len(self.layers) == 0:
            raise RuntimeError("Cannot compile model with 0 layers")
        y = T.ivector()
        x = T.dmatrix()

        num_layers = len(self.layers)
        first_layer = self.layers[0]
        cur_layer = first_layer.activation(T.dot(first_layer.W.T, x) +
                first_layer.b)
        layer_output = x
        for l in self.layers:
            layer_output = l.activation(T.dot(l.W.T, layer_output) + l.b)
        model_out = T.mean(self.loss_(layer_output, y))

        return function(inputs=[x, y], outputs=model_out,
                        updates=self.optimizer_(),
                        allow_input_downcast=True)

    def deep_copy(self):
        """
        Creates a deep copy of the model.

        Returns:
            A new Neural Network object, identical to the model being copied,
            but with separate parameters and independently compiled
        """
        copied_model = NeuralNetwork()
        copied_model.loss_ = self.loss_
        copied_model.params = [theano.shared(value=l.get_value(), borrow=True) for l
                in self.params]
        copied_model.params = [Layer(l.dim, l.activation, l.weight_init_scale) for l
                in self.layers]
        copied_model.compile()

        return copied_model

    def train(self, x, y):
        """
        Trains the network on batch of data `x` and `y`

        This method will most likely be called multiple times over several
        mini-batches of data for each epoch. It is not meant to be called on the
        entire dataset, although it can if that is how the user sets up their
        mini-batches.

        Params:
            x: feature data
            y: label data

        Returns:
            Training loss
        """
        pass

    def predict(self, x):
        """
        Runs the model and returns the output

        Params:
            x: feature data used in the prediction

        Returns:
            If f is a function representing the neural network, this method
            returns f(x), where `x` is the input
        """
        pass

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


prediction = T.argmax(output, axis=1)
loss = T.mean(T.nnet.categorical_crossentropy(output, y))

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
