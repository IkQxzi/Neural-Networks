import numpy as np 
import random

np.random.seed(5)

# add chain rule 
class LayerDense():

    # weights by neurons, biases
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.outputs = np.dot(inputs, self.weights) + self.biases # np.dot avoids for loop
        self.inputs = inputs # (batch_size, n_inputs)

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs, dvalues)
        self.dbiases = np.dot(np.full((self.biases.shape), 1), dvalues)


class ActivationReLU():

    def forward(self, inputs):
        self.inputs = inputs # (batch_size, n_inputs)

        self.outputs = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dvalues = np.minimum(1, self.outputs)
        # check the maths on this one


class ActivationSoftmax():

    def forward(self, inputs):
        self.inputs = inputs # (batch_size, n_inputs)

        exp_values = np.exp(self.inputs)
        self.outputs = exp_values / np.sum(exp_values, axis=1, keepdims=True) # axis 1 only sums each batch

    def backward(self):
        self.dvalues = self.outputs * (1-self.outputs)
    

def dashes(length=20):
    out_str = ""

    for dash in range(length):
        out_str += "-"

    return out_str

data_size = 16
batch_size = 32

layer1_neurons = 16
layer2_neurons = 8
layer3_neurons = 4

# creating batch dataset to prepare for batch inputs
mock_data = 0.10 * np.random.randn(batch_size, data_size)
print(f"{dashes(26)} mock data {mock_data.shape} {dashes(26)} \n{mock_data[0]}\n")


# -----------------------------------------------------------------------
# ---------------------------- forward pass ----------------------------- 
# -----------------------------------------------------------------------

layer1 = LayerDense(data_size, layer1_neurons)
layer1.forward(mock_data)
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

print(f"{dashes()} Layer [1] {layer1.outputs.shape} {dashes()}\n{activation1.outputs[0]}\n")
print(f"{dashes()} Layer [2] {layer2.outputs.shape} {dashes()}\n{activation2.outputs[0]}\n")
print(f"{dashes()} Layer [3] {layer3.outputs.shape} {dashes()}\n{activation3.outputs[0]}\n")


# -----------------------------------------------------------------------
# ---------------------------- backward pass ---------------------------- 
# -----------------------------------------------------------------------

# do backwards

activation3.backward()
layer3.backward()

activation2.backward()
layer2.backward()

activation1.backward()
layer1.backward()

print(f"dSoftmax: {activation3.dvalues[0]}")



