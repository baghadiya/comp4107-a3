#!/usr/bin/env python
from sklearn.datasets import fetch_mldata
import numpy as np
import random
import matplotlib.pyplot as plt

def show(vector):
    plt.imshow(np.array(vector).reshape(28, 28))
    plt.show()

class HopfieldNetwork(object):
    def __init__(self, train_dataset=[], test_dataset=[], threshold=60, time=100, theta=0.5):
        self.threshold = threshold
        self.time = time
        self.theta = theta
        self.train_dataset = t = np.array(train_dataset)
        self.num_neurons = n = self.train_dataset[0].shape[0]
        self.num_synapses = self.num_neurons ** 2 - self.num_neurons

        n_tup = (n, n)

        self.M = np.zeros(n_tup)

        for image_vector in t:
            m = np.zeros(n_tup)
            for i in range(n):
                for j in range(i + 1, n):
                    m[i][j] = m[j][i] = np.multiply(image_vector[i], image_vector[j])
            self.M += m
    def classify(self, vector):
        vector = vector[:]
        l = len(vector)
        for s in range(self.time):
            i = random.randint(0, l - 1)
            u = np.dot(self.M[i][:], vector) - self.theta

            if u > 0:
                vector[i] = 1
            elif u < 0:
                vector[i] = -1

        return map(lambda x: int(x >= self.threshold), vector)

def main():
    mnist = fetch_mldata('MNIST original', data_home='.cache')
    mnist_filtered = unpack(filter(lambda item: 1 <= item[1] and item[1] <= 5, zip(mnist.data, mnist.target)))[0]
    hf = HopfieldNetwork(
        train_dataset=mnist_filtered[:100]
    )
    v = hf.classify(mnist_filtered[101])
    show(mnist_filtered[101])
    show(v)

def unpack(a):
    return zip(*a)

if __name__ == '__main__':
    main()
