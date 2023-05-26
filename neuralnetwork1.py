import sys
import numpy as np

def activation_function(x):
    return 1 / (1 + np.exp(-x)) # sigmoid

# Lo que le cuesta a una red para acercarse a la realidad (costeo)
# cantidad y calidad
def cost_function(weights, biases):
    expected = np.random.rand()
    expected[2] = 1

# Diferencia de minimos cuadrados (evitar valores negativos)
def correct(utter_trash, expected):
    pass

# k: neurons count, n: inputs count per neuron, inputs: n x 1
def forward_propagation(k, n, inputs): 
    weights = np.random.rand(k, n) # k x n
    biases = np.random.rand(k) # k x 1
    wi = np.dot(weights, inputs) # (k x n) . (n x 1) = k x 1
    z = np.add(wi, biases) # k x 1
    return np.vectorize(activation_function)(z)
    
# Aprender: Llegar a los valores precisos de pesos y sesgos para la totalidad de la red
def learn(image, expected):
    inputs = get_image(image) # n = 784
    outputs1 = forward_propagation(len(inputs), 1, inputs)
    outputs2 = forward_propagation(16, len(outputs1), outputs1)
    outputs3 = forward_propagation(16, len(outputs2), outputs2)
    outputs4 = forward_propagation(10, len(outputs3), outputs3)
    correct(outputs4, expected)

def main():
    expected = []
    for i in range(10):
        new_array = np.zeros(10) 
        new_array[i] = 1
        expected += [new_array]

    digit = 3
    learn(expected[digit])

if __name__ == '__main__':
    main()
