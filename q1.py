#!/usr/bin/env python
from __future__ import division
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

    def hebbian(self):
        self.W = np.zeros([self.num_neurons, self.num_neurons])
        for image_vector, _ in self.train_dataset:
            self.W += np.outer(image_vector, image_vector) / self.num_neurons
        np.fill_diagonal(self.W, 0)

    def storkey(self):
        self.W = np.zeros([self.num_neurons, self.num_neurons])

        for image_vector, _ in self.train_dataset:
            self.W += np.outer(image_vector, image_vector) / self.num_neurons
            net = np.dot(self.W, image_vector)

            pre = np.outer(image_vector, net)
            post = np.outer(net, image_vector)

            self.W -= np.add(pre, post) / self.num_neurons
        np.fill_diagonal(self.W, 0)


    def __init__(self, train_dataset=[], mode='hebbian'):
        self.train_dataset = train_dataset
        self.num_training = len(self.train_dataset)
        self.num_neurons = len(self.train_dataset[0][0])

        self._modes = {
            "hebbian": self.hebbian,
            "storkey": self.storkey
        }

        self._modes[mode]()

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
