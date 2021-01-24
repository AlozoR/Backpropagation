"""
Backpropagation from scratch
"""

from csv import reader
from math import exp
from random import random, seed, shuffle
from typing import Any, List


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


def update_weights(network, row, l_rate):
    for i, layer in enumerate(network):
        inputs = row[:-1]
        if i != 0:
            inputs = [neuron["output"] for neuron in network[i - 1]]
        for neuron in layer:
            for j, inp in enumerate(inputs):
                neuron["weights"][j] += l_rate * neuron["delta"] * inp
            neuron["weights"][-1] = l_rate * neuron["delta"]


def train_network(network, train, l_rate, n_epoch, n_outputs):
    for epoch in range(n_epoch):
        sum_error = 0
        for row in train:
            outputs = forward_propagate(network, row)
            expected = [0 for i in range(n_outputs)]
            expected[row[-1]] = 1
            sum_error += sum(
                [(expected[i] - outputs[i]) ** 2 for i in range(len(expected))]
            )
            backward_propagate_error(network, expected)
            update_weights(network, row, l_rate)
        print(f">epoch={epoch}, lrate={l_rate:.3f}, error={sum_error:.3f}")


def predict(network, row):
    outputs = forward_propagate(network, row)
    return outputs.index(max(outputs))


def load_csv(filename):
    dataset: List[List[Any]] = list()
    with open(filename, "r") as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)

    for col in range(len(dataset[0]) - 1):
        for row in dataset:
            row[col] = float(row[col].strip())

    col = len(dataset[0]) - 1
    for row in dataset:
        row[col] = int(row[col].strip()) - 1
    return dataset


def normalize(dataset):
    minmax = [[min(column), max(column)] for column in zip(*dataset)]
    length = len(dataset[0]) - 1
    for i in range(length):
        for row in dataset:
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])


def main():
    seed(1)
    network = initialize_network(2, 1, 2)
    row = [1, 0, None]
    forward_propagate(network, row)
    expected = [0, 1]
    backward_propagate_error(network, expected)
    # print(network)

    seed(1)
    dataset = [
        [2.7810836, 2.550537003, 0],
        [1.465489372, 2.362125076, 0],
        [3.396561688, 4.400293529, 0],
        [1.38807019, 1.850220317, 0],
        [3.06407232, 3.005305973, 0],
        [7.627531214, 2.759262235, 1],
        [5.332441248, 2.088626775, 1],
        [6.922596716, 1.77106367, 1],
        [8.675418651, -0.242068655, 1],
        [7.673756466, 3.508563011, 1],
    ]

    n_inputs = len(dataset[0]) - 1
    n_outputs = 2
    network = initialize_network(n_inputs, 2, n_outputs)
    # train_network(network, dataset, 0.5, 20, n_outputs)

    # for row in dataset:
    #     prediction = predict(network, row)
    #     print(f"Expected={row[-1]}, Got={prediction}")
    # for layer in network:
    #     print(layer)

    seed(1)
    dataset = load_csv("data/wheat-seeds.csv")
    shuffle(dataset)
    normalize(dataset)
    split_limit = int(0.8 * len(dataset))
    training_set = dataset[:split_limit]
    test_set = dataset[split_limit:]
    # print('Training set:')
    # for row in training_set:
    #     print(row)
    # print('Test set:')
    # for row in test_set:
    #     print(row)

    n_inputs = len(training_set[0]) - 1
    n_hidden = 5
    n_outputs = 3
    network = initialize_network(n_inputs, n_hidden, n_outputs)
    train_network(network, training_set, 0.5, 200, n_outputs)

    n_correct = 0
    for row in test_set:
        prediction = predict(network, row)
        if prediction == row[-1]:
            n_correct += 1
        print(f"Expected={row[-1]}, Got={prediction}")

    print(f"Accuracy: {n_correct / len(test_set)}")
    # for layer in network:
    #     print(layer)


if __name__ == "__main__":
    main()
