import numpy as np # switch to cupy at some point for speed
import time
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# add loss plots for diff functions eventually
import matplotlib.pyplot as plt


# forgotten long ago
# np.random.seed(5) # remember to disable

# add chain rule 
class LayerDense():

    # weights by neurons, biases
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)

        self.biases = 0.10 * np.random.randn(1, n_neurons)
        # self.biases = np.zeros((1, n_neurons)) # can change if lots of dead neurons


    def forward(self, inputs):
        self.inputs = inputs # (batch_size, n_inputs)
        self.outputs = np.dot(inputs, self.weights) + self.biases 


    # what in the calculus
    def backward(self, forward_dvals):
        
        # self.dweights = np.dot(forward_dvals, self.inputs)
        self.dweights = np.dot(self.inputs.T, forward_dvals)

        self.dbiases = np.sum(forward_dvals, axis=0) # sums batch-wise

        # self.back_dvals = np.dot(self.weights, forward_dvals)
        self.back_dvals = np.dot(forward_dvals, self.weights.T)


    def learn(self, optimiser, learning_rate):

        optimiser.forward(self.weights, self.dweights, self.biases, self.dbiases)
        self.weights, self.biases = optimiser.new_weights, optimiser.new_biases


class ActivationReLU():

    def forward(self, inputs):
        self.inputs = inputs # (batch_size, n_inputs)
        self.outputs = np.clip(np.maximum(0, inputs), a_min=None, a_max=1e6)


    def backward(self, forward_dvals):

        # to propogate to previous layer
        # should be 1 (if dvals > 0)
        self.relu_dvals = self.inputs / self.inputs

        self.dvals = forward_dvals * self.relu_dvals
        # check the maths on this one - hasnt broken yet so probably right
        # is probably wrong



class ActivationSoftmax():

    def forward(self, inputs):
        self.inputs = inputs # (batch_size, n_inputs)

        max_vals = np.max(self.inputs, axis=1, keepdims=True)
        # must be done for numerical stability (otherwise returns inf)
        exp_values = np.exp(self.inputs - max_vals) 
        exp_values = np.clip(exp_values, 1e-6, 1e6) # clip to avoid log(0) (which is infinite)
        self.outputs = exp_values / np.sum(exp_values, axis=1, keepdims=True) # axis 1 sums batch-wise


    # done in a way to allow random order of target_classes (not diagly aligned)
    def backward(self, target_classes, forward_dvals):

        # if the inputs were jacobean matrices (ie aligned diagly)
        # this could be significantly easier
        # but i assume this would cause the neurons to just learn the sequence 
        # instead of the handwriting features

        '''
        output at the same index as the "true" output has derivative: 
        (output * (1 - output)), 

        and at every other index it is:
        (output * (output at the "true" output's index) * -1))


        # returns matrix of only "relevant" outputs (can sum to remove empty spaces)
        # ie ([0.7, 0, 0, 0]...)
        partial_d = outputs ([0.7, 0.05, 0.15, 0.1]...) * target_classes ([1, 0, 0, 0]...)
        
        # maybe dot product? 
        # ie ([0.21, 0, 0, 0]...)
        diag_d = partial_d * (1 - partial_d)
        
        # not sure if np.dot or regular multiplication
        # ie ([0, -0.035, -0.105, -0.07]...)
        off_diag_d = -1 * sum(partial_d, axis = 1, keepdims) ([0.7]...) * (outputs - partial_d) ([0, 0.05, 0.15, 0.1]...)

        # interpolate diag_d & off_diag_d somehow
        # ie ([0.21, -0.035, -0.105,- 0.07])
        final_dvals = diag_d + off_diag_d

        '''
        # returns matrix of only "relevant" outputs (can sum to remove empty spaces)
        # ie ([0.7, 0, 0, 0]...)
        rel_outputs = self.outputs * target_classes

        # maybe dot product? 
        # ie ([0.21, 0, 0, 0]...)
        diag_dvals = rel_outputs * (1 - rel_outputs)

        # not sure if np.dot or regular multiplication
        # ie ([0, -0.035, -0.105, -0.07]...)
        off_diag_dvals = -1 * np.sum(rel_outputs, axis=1, keepdims=True) * (self.outputs - rel_outputs)

        softmax_dvals = diag_dvals + off_diag_dvals
        self.dvals = (softmax_dvals.T * forward_dvals).T
    


class LossCategoricalCrossEntropy():


    def forward(self, outputs, target_classes):

        prob_diff = 1 - np.sum(outputs * target_classes, axis=1) # raw diff 
        self.loss = -1 * np.log(prob_diff) # log scaling encourages drastic changes when loss is large


    def backward(self):
        self.dloss = -1 / self.loss


class OptimiserSGD():

    def forward(self, layer, learning_rate):

        layer.weights += layer.dweights * learning_rate
        layer.biases += layer.dbiases * learning_rate


def dashes(length=20):
    out_str = ""

    for dash in range(length):
        out_str += "-"

    return out_str


# -----------------------------------------------------------------------
# ---------------------------- initialisation ----------------------------- 
# -----------------------------------------------------------------------

def initialise_network(neuron_nums, input_dimensions):

    layer1_neurons, layer2_neurons, layer3_neurons = neuron_nums

    layer1 = LayerDense(input_dimensions, layer1_neurons)
    activation1 = ActivationReLU()

    layer2 = LayerDense(layer1_neurons, layer2_neurons)
    activation2 = ActivationReLU()

    layer3 = LayerDense(layer2_neurons, layer3_neurons)
    activation3 = ActivationSoftmax() # single-class problem

    loss_function = LossCategoricalCrossEntropy()

    optimiser = OptimiserSGD()

    return [layer1, activation1, layer2, activation2, layer3, activation3, loss_function], optimiser


# -----------------------------------------------------------------------
# ---------------------------- forward pass ----------------------------- 
# -----------------------------------------------------------------------

def forward_pass(network, X_arr, y_arr):

    layer1, activation1, layer2, activation2, layer3, activation3, loss_function = network

    # add for loop here 

    layer1.forward(X_arr)
    activation1.forward(layer1.outputs)

    layer2.forward(layer1.outputs)
    activation2.forward(layer2.outputs)

    layer3.forward(layer2.outputs)
    activation3.forward(layer3.outputs)

    loss_function.forward(activation3.outputs, y_arr)

    # change below so doesnt print 60,000 times

    print(f"{dashes()} Layer [1] {layer1.outputs.shape} {dashes()}\n{activation1.outputs[0][:20]}\n")
    print(f"weights: {layer1.weights.shape}")
    print(f"biases: {layer1.biases.shape}")

    print(f"{dashes()} Layer [2] {layer2.outputs.shape} {dashes()}\n{activation2.outputs[0]}\n")
    print(f"weights: {layer2.weights.shape}")
    print(f"biases: {layer2.biases.shape}")

    print(f"{dashes()} Layer [3] {layer3.outputs.shape} {dashes()}\n{activation3.outputs[0]}\n")

    print(f"sum: {np.sum(activation3.outputs, axis=1)}")

    print(f"weights: {layer3.weights[0]}")
    print(f"weights: {layer3.weights.shape}")

    print(f"biases: {layer3.biases[0]}")
    print(f"biases: {layer3.biases.shape}\n")

    print(f"{dashes()} CCE Loss {loss_function.loss.shape} {dashes()}\n{loss_function.loss}")


# -----------------------------------------------------------------------
# ---------------------------- backward pass ---------------------------- 
# ------------------------------------------------------------------------

def backward_pass(network, output_arr, target_classes):

    layer1, activation1, layer2, activation2, layer3, activation3, loss_function = network

    # loss function
    loss_function.backward()
    # activation3: [batch_size, output_dim], same for y_train/batch => must be transposed

    print(f"{dashes()} Loss Derivative {loss_function.dloss.shape} {dashes()}\n{loss_function.dloss}")

    activation3.backward(target_classes, loss_function.dloss)
    layer3.backward(activation3.dvals)

    # learn calculus, what is going on here
    # calculus learnt
    activation2.backward(layer3.back_dvals)
    layer2.backward(activation2.dvals)

    activation1.backward(layer2.back_dvals)
    layer1.backward(activation1.dvals)

    print(f"{dashes()} Layer [3] [backward pass] {dashes()}")
    print(f"dweights: {layer3.dweights[0]}")
    print(f"dbiases: {layer3.dbiases[0]}")

    print(f"{dashes()} Layer [2] [backward pass] {dashes()}")
    print(f"relu_dvals: {activation2.relu_dvals[:10]}")
    print(f"dweights: {layer2.dweights[0][:10]}")
    print(f"dbiases: {layer2.dbiases[:10]}")

    print(f"{dashes()} Layer [1] [backward pass] {dashes()}")
    print(f"[2]back_dvals: {layer2.back_dvals[:10]}")
    print(f"relu_dvals: {activation1.relu_dvals[:10]}")
    print(f"activation.dvals: {activation1.dvals[:10]}")
    print(f"dweights: {layer1.dweights[0][:10]}")
    print(f"dbiases: {layer1.dbiases[:10]}")

def update_params(network_layers, optimiser, learning_rate):
    
    layer1, layer2, layer3 = network_layers

    optimiser.forward(layer1, learning_rate)
    optimiser.forward(layer2, learning_rate)
    optimiser.forward(layer3, learning_rate)


# -----------------------------------------------------------------------
# ---------------------------- data handling ----------------------------- 
# -----------------------------------------------------------------------

# MNIST data:

# X_train: [60,000 samples, 784 pixel activations]
# y_train: [60,000 samples, 10 classes (1-10)]

# X_batch: [32 samples, 784 pixel activations]
# y_batch: [32 samples, 10 classes (1-10)]

def initialise_mnist_dataset(batch_size):

    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(-1, 28 * 28).astype("float32") / 255.0
    X_test = X_test.reshape(-1, 28 * 28).astype("float32") / 255.0

    y_train = to_categorical(y_train, 10)  # 10 classes for MNIST
    y_test = to_categorical(y_test, 10)

    batch_nums = X_train.shape[0] // batch_size 

    # may need to transpose 
    X_train_batch = np.array_split(X_train, batch_nums, axis=0)
    X_test_batch = np.array_split(X_test, batch_nums, axis=0)

    y_train_batch = np.array_split(y_train, batch_nums, axis=0)
    y_test_batch = np.array_split(y_test, batch_nums, axis=0)

    return X_train_batch, y_train_batch, X_test_batch, y_test_batch


# -----------------------------------------------------------------------
# ---------------------------- network run ---------------------------- 
# ------------------------------------------------------------------------

def main():

    input_dims = 28 * 28
    network_neurons = [784, 16, 10]
    batch_size = 32 # will play around with other values depending on performance
    learning_rate = 1  

    network, optimiser = initialise_network(network_neurons, input_dims)
    network_layers = network[:-1][::2]
    X_train, y_train, X_test, y_test = initialise_mnist_dataset(batch_size) # all in batches

    for batch_num, X_batch in enumerate(X_train):

        train_output = forward_pass(network, X_batch, y_train[batch_num])
        backward_pass(network, train_output, y_train[batch_num])

        update_params(network_layers, optimiser, learning_rate)

        input('Continue...')

        # layer 1 dweights are all 0 for some reason

        # order passes into functions
        # only print 1st & last pass
        # print & plot accuracy, loss, etc

        # pass in multiple times

main()









