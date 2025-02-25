import numpy as np 
import random

random.seed(1)
np.random.seed(1)

class LayerDense():

    # weights by neurons, biases
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randint(0, 10, (n_inputs, n_neurons))

        # mixing negatives into weights
        for inp_ind, inp in enumerate(self.weights):
            for neuron_ind, neuron in enumerate(inp):
                sign = np.random.choice([1, -1])
                self.weights[inp_ind, neuron_ind] *= sign

        # can change if many dead neurons
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.outputs = np.dot(inputs, self.weights) + self.biases # np.dot avoids for loop
        self.inputs = inputs

    def ReLU(self):
        self.outputs = np.maximum(0, self.outputs)

    def Sigmoid(self):
        self.outputs = np.max

        # add Sigmoid logic 

layer1_neurons = 4
layer2_neurons = 8

# creating mock dataset
mock_data = 0.10 * np.random.randint(1, 10, (1, 16))

# mixing negatives into dataset
for row_ind, row in enumerate(mock_data): # better practice than len()-1
    for data_ind, data in enumerate(row):
        sign = np.random.choice([1, -1])
        mock_data[row_ind, data_ind] *= sign

print(mock_data)

# forward pass
layer1 = LayerDense(16, 4)
layer1.forward(mock_data)
layer1.ReLU()
output1 = layer1.outputs
layer2 = LayerDense(layer1_neurons, layer2_neurons)

# outputs 
print(f"------------------- layer 1 -------------------\n{output1}")
print(output1)



