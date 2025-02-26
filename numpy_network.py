import numpy as np # switch to cupy at some point for speed
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

# forgotten long ago
np.random.seed(5) # remember to disable

# add chain rule 
class LayerDense():

    # weights by neurons, biases
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.inputs = inputs # (batch_size, n_inputs)
        self.outputs = np.dot(inputs, self.weights) + self.biases 

    # what in the calculus
    def backward(self, forward_dvals):
        
        # self.dweights = np.dot(forward_dvals, self.inputs)
        self.dweights = np.dot(self.inputs.T, forward_dvals)
        print(f"inputs: {self.inputs.shape}")
        print(f"dweights: {self.dweights.shape}")

        self.dbiases = np.sum(forward_dvals, axis=0) # sums batch-wise
        print(f"dbiases: {self.dbiases.shape}")

        # self.back_dvals = np.dot(self.weights, forward_dvals)
        self.back_dvals = np.dot(forward_dvals, self.weights.T)
        print(f"back_dvals: {self.back_dvals.shape}")
        # self.dbiases = np.dot(partial_dbiases, forward_dvals) # should they use the same forward_dvals?


class ActivationReLU():

    def forward(self, inputs):
        self.inputs = inputs # (batch_size, n_inputs)
        self.outputs = np.maximum(0, inputs)

    def backward(self, forward_dvals):

        print(f"forward_dvals: {forward_dvals.shape}")

        # to propogate to previous layer
        relu_dvals = self.inputs / self.inputs
        print(f"relu_dvals: {relu_dvals.shape}")


        self.dvals = forward_dvals * relu_dvals
        print(f"dvals: {self.dvals.shape}")
        # check the maths on this one - hasnt broken yet so probably right


class ActivationSoftmax():

    def forward(self, inputs):
        self.inputs = inputs # (batch_size, n_inputs)

        exp_values = np.exp(self.inputs)
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
        print(f"diag_dvals: {diag_dvals.shape}")

        # not sure if np.dot or regular multiplication
        # ie ([0, -0.035, -0.105, -0.07]...)
        off_diag_dvals = -1 * np.sum(rel_outputs, axis=1, keepdims=True) * (self.outputs - rel_outputs)
        print(f"off_diag_dvals: {off_diag_dvals.shape}")

        softmax_dvals = diag_dvals + off_diag_dvals
        self.dvals = (softmax_dvals.T * forward_dvals).T
        print(f"self.dvals: {self.dvals.shape}")
    

class LossCategoricalCrossEntropy():

    def calculate(self, outputs, target_classes):

        prob_diff = 1 - np.sum(outputs * target_classes, axis=1) # raw diff 
        self.loss = -1 * np.log(prob_diff) # log scaling encourages drastic changes when loss is large
        # -1 / pred

        self.dloss = -1 / self.loss


def dashes(length=20):
    out_str = ""

    for dash in range(length):
        out_str += "-"

    return out_str


# -----------------------------------------------------------------------
# ---------------------------- data & __init__ ----------------------------- 
# -----------------------------------------------------------------------

# creating batch dataset to prepare for batch inputs
# mock_data = 0.10 * np.random.randn(batch_size, data_size)
# print(f"{dashes(26)} mock data {mock_data.shape} {dashes(26)} \n{mock_data[0]}\n")

# MNIST data:

# X_train: [60,000 samples, 784 pixel activations]
# y_train: [60,000 samples, 10 classes (1-10)]

# X_batch: [32 samples, 784 pixel activations]
# y_batch: [32 samples, 10 classes (1-10)]

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(-1, 28 * 28).astype("float32") / 255.0
X_test = X_test.reshape(-1, 28 * 28).astype("float32") / 255.0

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train, 10)  # 10 classes for MNIST
y_test = to_categorical(y_test, 10)

# currently size 32, will play around with other values depending on performance
batch_size = 32 
batch_nums = X_train.shape[0] // batch_size 

X_batch = np.array_split(X_train, batch_nums, axis=0)
X_batch = X_batch # may need to transpose 

y_batch = np.array_split(y_train, batch_nums, axis=0)
y_batch = y_batch # may need to transpose 

input_dim = X_train.shape[1]

layer1_neurons = 784
layer2_neurons = 16
layer3_neurons = 10

layer1 = LayerDense(input_dim, layer1_neurons)


# -----------------------------------------------------------------------
# ---------------------------- forward pass ----------------------------- 
# -----------------------------------------------------------------------

layer1.forward(X_batch[0])
activation1 = ActivationReLU()
activation1.forward(layer1.outputs)

layer2 = LayerDense(layer1_neurons, layer2_neurons)
layer2.forward(layer1.outputs)
activation2 = ActivationReLU()
activation2.forward(layer2.outputs)

layer3 = LayerDense(layer2_neurons, layer3_neurons)
layer3.forward(layer2.outputs)
activation3 = ActivationSoftmax()
activation3.forward(layer3.outputs)

print(f"{dashes()} Layer [1] {layer1.outputs.shape} {dashes()}\n{activation1.outputs[0][:20]}\n")
print(f"weights: {layer1.weights.shape}")
print(f"biases: {layer1.biases.shape}")

print(f"{dashes()} Layer [2] {layer2.outputs.shape} {dashes()}\n{activation2.outputs[0]}\n")
print(f"weights: {layer2.weights.shape}")
print(f"biases: {layer2.biases.shape}")

print(f"{dashes()} Layer [3] {layer3.outputs.shape} {dashes()}\n{activation3.outputs[0]}\n")
print(f"weights: {layer3.weights.shape}")
print(f"biases: {layer3.biases.shape}\n")


# -----------------------------------------------------------------------
# ---------------------------- backward pass ---------------------------- 
# ------------------------------------------------------------------------

# loss function

loss_function = LossCategoricalCrossEntropy()
loss_function.calculate(activation3.outputs, y_batch[0])
# activation3: [batch_size, output_dim], same for y_train/batch => must be transposed

# do backwards

print(f"{dashes()} CCE Loss {loss_function.loss.shape} {dashes()}\n{loss_function.loss}")
print(f"{dashes()} Loss Derivative {loss_function.dloss.shape} {dashes()}\n{loss_function.dloss}")

print(f"{dashes()} Layer [3] [backward pass] {dashes()}")
print(f"target_classes: {y_batch[0].shape}")
print(f"loss_function: {loss_function.loss.shape}")
activation3.backward(y_batch[0], loss_function.dloss)
layer3.backward(activation3.dvals)

# learn calculus, what is going on here
# calculus learnt
print(f"{dashes()} Layer [2] [backward pass] {dashes()}")
activation2.backward(layer3.back_dvals)
layer2.backward(activation2.dvals)

print(f"{dashes()} Layer [1] [backward pass] {dashes()}")
activation1.backward(layer2.back_dvals)
layer1.backward(activation1.dvals)

# order passes into functions
# only print 1st & last pass
# print & plot accuracy, loss, etc
# SGD Optimiser

