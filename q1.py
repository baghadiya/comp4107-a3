#!/usr/bin/env python
from sklearn.datasets import fetch_mldata
import numpy as np
import random
import matplotlib.pyplot as plt

def add_noise(vector, ratio=0.2):
    indices = range(len(vector))
    num = ratio * len(indices)
    for i in range(int(num)):
        c = random.choice(indices)
        vector[c] = 1 if vector[c] == -1 else -1

def show(vector, title='', suptitle=''):
    plt.imshow(np.array(vector).reshape(28, 28))
    plt.title(title)
    plt.suptitle(suptitle)
    plt.show()

def unpack(a):
    return zip(*a)


def test(index, data, network, suptitle=''):
    v = network.activate(data[index])
    show(data[index], "Input", suptitle)
    show(v, "Output", suptitle)


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
    targets = mnist.target.tolist()

    start, end = targets.index(1), targets.index(6)

    dataset = [[1 if pixel > 0 else -1 for pixel in vector] for vector in mnist.data[start:end]]

    hf = HopfieldNetwork(
        train_dataset=dataset[:1000]
    )

    test(11, dataset, hf, 'Without noise')
    add_noise(dataset[11], ratio=0.4)
    test(11, dataset, hf, 'With noise')
    test(12, dataset, hf)
    test(13, dataset, hf)
