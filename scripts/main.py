"""
Backpropagation from scratch
"""

from math import exp
from random import random, seed


def initialize_network(n_inputs, n_hidden, n_outputs):
    network = list()
    hidden_layer = [
        {"weights": [random() for i in range(n_inputs + 1)]}
        for i in range(n_hidden)
    ]
    network.append(hidden_layer)
    output_layer = [
        {"weights": [random() for i in range(n_hidden + 1)]}
        for i in range(n_outputs)
    ]
    network.append(output_layer)
    return network


def activate(weights, inputs):
    return sum(map(lambda w, i: w * i, weights[:-1], inputs)) + weights[-1]


def transfer(activation):
    return 1.0 / (1.0 + exp(-activation))


def forward_propagate(network, row):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            neuron["output"] = transfer(activate(neuron["weights"], inputs))
            new_inputs.append(neuron["output"])
        inputs = new_inputs
    return inputs


def transfer_derivative(output):
    return output * (1.0 - output)


def backward_propagate_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network) - 1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += neuron["weights"][j] * neuron["delta"]
                errors.append(error)
        else:
            for j, neuron in enumerate(layer):
                errors.append(expected[j] - neuron["output"])
        for j, neuron in enumerate(layer):
            neuron["delta"] = errors[j] * transfer_derivative(neuron["output"])


def main():
    seed(1)
    network = initialize_network(2, 1, 2)
    row = [1, 0, None]
    forward_propagate(network, row)
    expected = [0, 1]
    backward_propagate_error(network, expected)
    print(network)


if __name__ == "__main__":
    main()
