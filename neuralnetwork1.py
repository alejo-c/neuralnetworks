# ~/AppData/Local/Programs/Python/Python39/python
import sys
import numpy as np
import matplotlib.pyplot as plt


class Model:
    def __init__(self, layers):
        self.layers = layers
        self.weights_history = []
        self.biases_history = []
        self.outputs_history = []

    # Aprender: Llegar a los valores precisos de pesos y sesgos para la totalidad de la red
    def fit(self, inputs, expected_outputs, epochs):
        output = self.forward_propagation(len(inputs), 1, inputs)
        for neurons in self.layers[1:]:
            output = self.forward_propagation(neurons, len(output), output)
        self.correct(output, expected_outputs)

    def activation_function(z):
        return 1 / (1 + np.exp(-z))  # sigmoid

    # k: neurons count, n: inputs count per neuron, inputs: n x 1
    def forward_propagation(self, k, n, inputs):
        weights = np.random.rand(k, n)  # k x n
        biases = np.random.rand(k)  # k x 1

        wi = np.dot(weights, inputs)  # (k x n) . (n x 1) = k x 1
        z = np.add(wi, biases)  # k x 1
        output = np.vectorize(Model.activation_function)(z)

        self.weights_history += [weights]
        self.biases_history += [biases]
        self.outputs_history += [output]
        return output

    # Diferencia de minimos cuadrados (evitar valores negativos)
    def correct(utter_trash, expected):
        pass

    # Lo que le cuesta a una red para acercarse a la realidad (costeo)
    def cost_function(weights, biases):
        expected = np.random.rand()
        expected[2] = 1


def main():
    values = np.loadtxt(sys.argv[1])
    inputs = np.array(values[:, 0])
    expected = np.array(values[:, 1])

    model = Model(layers=[10, 10, 10, 1])
    training = model.fit(inputs, expected, epochs=1000)

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.plot(training.loss)


if __name__ == "__main__":
    main()
