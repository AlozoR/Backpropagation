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


def main():
    seed(1)
    network = initialize_network(2, 1, 2)
    print(network)
    row = [1, 0, None]
    output = forward_propagate(network, row)
    print(output)


if __name__ == "__main__":
    main()
