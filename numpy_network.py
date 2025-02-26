import numpy as np 
from tensorflow.keras.datasets import mnist

np.random.seed(5) # remember to disable

# add chain rule 
class LayerDense():

    # weights by neurons, biases
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.inputs = inputs # (batch_size, n_inputs)
        self.outputs = np.dot(inputs, self.weights) + self.biases # np.dot avoids for loop

    def backward(self, chain_dvalues):
        partial_dweights = self.inputs # for clarity, may remove for memory efficiency
        self.dweights = np.dot(partial_dweights, chain_dvalues)

        partial_dbiases = np.full((self.biases.shape), 1)
        self.dbiases = np.dot(partial_dbiases, chain_dvalues) # should they use the same chain_dvalues?


class ActivationReLU():

    def forward(self, inputs):
        self.inputs = inputs # (batch_size, n_inputs)
        self.outputs = np.maximum(0, inputs)

    def backward(self, chain_dvalues):
        partial_dvalues = np.minimum(1, self.outputs)
        self.dvalues = np.dot(partial_dvalues, chain_dvalues)
        # check the maths on this one


class ActivationSoftmax():

    def forward(self, inputs):
        self.inputs = inputs # (batch_size, n_inputs)

        exp_values = np.exp(self.inputs)
        self.outputs = exp_values / np.sum(exp_values, axis=1, keepdims=True) # axis 1 only sums each batch

    def backward(self, chain_dvalues):
        partial_dvalues = self.outputs * (1-self.outputs)
        self.dvalues = np.dot(partial_dvalues, chain_dvalues)
    

class LossCategoricalCrossEntropy():

    def calculate(outputs, target_classes):
        # multiply 1 hot vector by outputs, then -log(ans)
        # might need to transpose one of the matrices
        self.loss = -1 * np.log(np.dot(outputs, target_classes))
        # log scaling encourages drastic changes when loss is large

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


# -----------------------------------------------------------------------
# ---------------------------- data handling ----------------------------- 
# -----------------------------------------------------------------------

# creating batch dataset to prepare for batch inputs
mock_data = 0.10 * np.random.randn(batch_size, data_size)
print(f"{dashes(26)} mock data {mock_data.shape} {dashes(26)} \n{mock_data[0]}\n")

# MNIST data:

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(-1, 28 * 28).astype("float32") / 255.0
X_test = X_test.reshape(-1, 28 * 28).astype("float32") / 255.0

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train, 10)  # 10 classes for MNIST
y_test = to_categorical(y_test, 10)

print(X_train)

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

# loss function



# do backwards

output_vector = ([0.7, 0.1, 0.1, 0.1])

activation3.backward()
layer3.backward()

activation2.backward()
layer2.backward()

activation1.backward()
layer1.backward()

print(f"dSoftmax: {activation3.dvalues[0]}")



