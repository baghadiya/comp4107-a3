#!/usr/bin/env python
from sklearn.datasets import fetch_mldata
import numpy as np
import random
import matplotlib.pyplot as plt


def show(vector, title=''):
    plt.imshow(np.array(vector).reshape(28, 28))
    plt.title(title)
    plt.show()


def unpack(a):
    return zip(*a)


def test(index, data, network):
    v = network.activate(data[index])
    show(data[index], "Input")
    show(v, "Output")


class HopfieldNetwork(object):
    def __init__(self, train_dataset=[], tolerance=.335):
        self.tolerance = tolerance
        self.train_dataset = t = np.array(train_dataset)
        self.num_neurons = n = self.train_dataset[0].shape[0]

        self.W = np.zeros((n, n))
        for image_vector in t:
            p = np.array([image_vector]).T
            self.W += p * p.T
        self.W -= len(t) * np.identity(len(t[0]))

    def activate(self, vector):
        changed = True
        while changed:
            changed = False
            indices = range(0, len(vector))
            random.shuffle(indices)

            # Vector to contain updated neuron activations on next iteration
            new_vector = [0] * len(vector)

            for i in range(0, len(vector)):
                neuron_index = indices.pop()

                s = self.compute_sum(vector, neuron_index)
                new_vector[neuron_index] = 1 if s >= 0 else -1
                changed = vector[neuron_index] != new_vector[neuron_index]

            vector = new_vector

        return vector

    def compute_sum(self, vector, neuron_index):
        s = 0
        for pixel_index in range(len(vector)):
            pixel = vector[pixel_index]
            if pixel > 0:
                s += self.W[neuron_index][pixel_index]

        return s


if __name__ == '__main__':
    mnist = fetch_mldata('MNIST original', data_home='.cache')

    mnist_filtered = [[1 if pixel > 0 else -1 for pixel in vector] for vector in mnist.data[:10000]]

    hf = HopfieldNetwork(
        train_dataset=mnist_filtered[:2000]
    )

    test(11, mnist_filtered, hf)
    test(12, mnist_filtered, hf)
    test(13, mnist_filtered, hf)
