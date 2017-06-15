import numpy as np
import theano
import matplotlib.pyplot as plt
import theano.tensor as T
from enum import Enum
from sklearn.datasets import fetch_mldata

rng = np.random

class Optimizer(Enum):
    SGD = "SGD"
    LBFGS = "L-BFGS"

def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate

def load_mnist(shuffle=False):
    """
    Loads MNIST from the mldata repository

    Key information:
        * Number of features: 784
        * Test data: 10,000 points (adjustable)

    Params:
        shuffle: flag used to determine whether the data should be shuffled or
                 not prior to creating the test and training sets

    Returns:
        A tuple consisting of (train_X, train_y, test_X, test_y)
    """
    mnist = fetch_mldata("MNIST original")
    indices = list(range(70000))
    if shuffle:
        np.random.shuffle(indices)
    else:
        tmp_x = indices[:60000]
        tmp_y = indices[60000:]
        np.random.shuffle(tmp_x)
        indices = tmp_x + tmp_y
    train_indices = indices[:60000]
    test_indices = indices[60000:]
    train_X, train_y = mnist.data[train_indices, :], mnist.target[train_indices]
    test_X, test_y = mnist.data[test_indices, :], mnist.target[test_indices]
    
    return (train_X, train_y, test_X, test_y)

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

        self.W = theano.shared(value=rng.normal(size=dim,
                                                scale=self.weight_init_scale_),
                              borrow=True)
        self.b = theano.shared(value=self.weight_init_scale_, borrow=True)

    # TODO: Possibly remove this function
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

    def __init__(self, loss=T.nnet.categorical_crossentropy, optimizer=None):
        """
        Constructor

        Params:
            loss: loss function (Default is categorical crossentropy)
            optimizer: optimizer used to train the network
        """
        # Labels
        self.y_ = T.ivector()

        # Feature data
        self.x_ = T.dmatrix()

        # Specified loss function
        self.loss_ = loss

        # Optimizer enum holding the chosen optimizer
        self.optimizer_ = optimizer

        # List of Layer objects
        self.layers = []

        # Network parameters
        self.params = []

        # Holds references to parameters used by the optimizer to keep them in
        # memory
        self.optimizer_params_ = None

        # theano.function representing the neural network
        self.compiled_model_ = None

        # theano.function for model ouput
        self.model_out_ = None

        # theano.function for the cost function
        self.cost_ = None

        # theano.function that is a deep copy of the cost function
        self.copied_cost_ = None
    
        # theano.function used for predictions
        self.compiled_predictor_ = None

        # learning rate used by the optimizers
        self.learning_rate_ = theano.shared(value=0.05, borrow=True)

        # Tracks the number of iterations completed by the optimizer
        self.t_ = theano.shared(value=1, borrow=True)

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
            if self.layers[-1].dim[1] != dim[0]:
                raise ValueError(("Dimension mismatch - input dimension of"
                "current layer must match output dimension of prior layer"))

        # Create layer object
        self.layers.append(Layer(dim, activation,
            weight_init_scale))
        self.params = self.params + self.layers[-1].params()
        return self

    def get_updates_func(self):
        """
        Returns the updates list for the specified optimizer

        Returns:
            List of tuples specifying update rules for model
        """
        if self.optimizer_ == Optimizer.SGD:
            # SGD
            
            # TODO: need a better way of allowing the user to pick the learning
            # rate
            def sgd():
                updates = [(p, p - self.learning_rate_ * g) for (p, g) in
                        zip(self.params, T.grad(self.cost_, wrt=self.params))]
                updates.append((self.learning_rate_, self.learning_rate_ *\
                                np.sqrt(self.t_/(self.t_ + 1))))
                updates.append((self.t_, self.t_+1))
                print("sgd")
                return updates
            return sgd
        
        elif self.optimizer_ == Optimizer.LBFGS:
            # L-BFGS
            self.copied_cost_ = self.cost_.deep_copy()
            @static_vars(B=None, prior_gradient=None)
            def lbfgs():
                gradients = theano.gradient.jacobian(self.cost_, wrt=params)
                self.optimizer_params_.["B"] = theano.shared(value=[np.identity(n=i.shape.eval()[0],
                    m=i.shape.eval()[1], dtype=float) for i in gradients],
                                  borrow=True)
                self.optimizer_params_["prior_gradients"] = gradients
                # Get line search direction
                p = -T.dot(T.nlinalg.MatrixInverse(self.optimizer_params_["B"]), gradients)

                # Perform line search
                num_steps = 100
                step_vals = [self.learning_rate_ * i for i in range(num_steps)]
                best_step = None
                best_cost = None
                for step in step_vals:
                    self.copied_cost_.params +=  step * p
                    if best_step == None:
                        best_step = step
                        best_cost = self.copied_cost_(self.x_, self.y_)
                    else:
                        tmp_cost = self.copied_cost_(self.x_, self.y_)
                        if tmp_cost < best_cost:
                            best_step = step
                            best_cost = tmp_cost
                    self.copied_cost_.params -= step * p
                sk = best_step * p
                self.copied_cost_.params -= sk
                updates = [(p, np) for (p, np) in zip(self.params, new_params)]
                yk = T.grad(cost=self.cost_, wrt=self.params) - lbfgs.prior_gradient
                lbfgs.B += T.dot(yk, yk.T) / T.dot(yk.T, sk) -\
                        T.dot(T.dot(lbfgs.B, sk), T.dot(sk.T, B))/\
                        T.dot(T.dot(sk.T, lbfgs.B), sk)

                # Update y and B
                return updates
            return lbfgs
    '''
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
    '''
    def compile(self):
        """
        Creates a Theano function object to store the model definition
        """
        if len(self.layers) == 0:
            raise RuntimeError("Cannot compile model with 0 layers")

        num_layers = len(self.layers)
        self.model_out_ = self.x_
        for l in self.layers:
            self.model_out_ = l.activation(T.dot(self.model_out_, l.W) + l.b)
        self.cost_ = T.mean(self.loss_(self.model_out_, self.y_))

        updater = self.get_updates_func()
        self.compiled_model_ = theano.function(inputs=[self.x_, self.y_], 
                                               outputs=self.cost_,
                                               updates=updater(),
                                               allow_input_downcast=True)

        self.compiled_predictor_ = theano.function(inputs=[self.x_],
                                                   outputs=T.argmax(self.model_out_,
                                                                    axis=1),
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
        return self.compiled_model_(x, y)

    def predict(self, x):
        """
        Runs the model and returns the output

        Params:
            x: feature data used in the prediction

        Returns:
            If f is a function representing the neural network, this method
            returns f(x), where `x` is the input
        """
        return self.compiled_predictor_(x)

    def test_error(self, x, y):
        """
        Calculated the test error for the current model state

        Params:
            x: test feature data
            y: test labels

        Returns:
            Percent incorrect on the test set
        """
        return np.mean(y != self.predict(x))

    def test_accuracy(self, x, y):
        """
        Calculates the test accuracy for the current model state

        Params:
            x: test feature data
            y: test label data

        Returns:
            Percent correct on the test set
        """
        return 1 - self.test_error(x, y)

# TODO: Remove this as more updates are made
'''
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

'''

if __name__ == "__main__":

    n_hidden = 450

    # Create neural network with SGD optimizer
    nn = NeuralNetwork(optimizer=Optimizer.SGD)

    # Build a three layer network
    nn.add_layer(dim=(784, n_hidden)).\
       add_layer(dim=(n_hidden, n_hidden)).\
       add_layer(dim=(n_hidden, n_hidden)).\
       add_layer(dim=(n_hidden, 10),activation=T.nnet.softmax)

    # Compile network
    obj_fn = nn.compile()

    # Train network
    test_err = []
    iter_num = []
    i_num = 0
    n_epochs = 10
    batch_size = 50
    train_X, train_y, test_X, test_y = load_mnist()
    for i in range(n_epochs):
        count = 0
        for (start, end) in zip(range(0, len(train_y) - batch_size, batch_size),
                range(batch_size, len(train_y), batch_size)):
            cost = nn.train(train_X[start:end,:],
                    train_y[start:end])
        print("Epoc {} test accuracy: {}".format(i, nn.test_accuracy(test_X,
            test_y)))

    #plt.plot(iter_num, test_err)
    #plt.savefig("training_error.png", dpi=700)
