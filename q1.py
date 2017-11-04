#!/usr/bin/env python
from sklearn.datasets import fetch_mldata
import numpy as np

class HopfieldNetwork(object):
    def __init__(self, train_dataset=[], test_dataset=[], threshold=1.):
        self.train_dataset = t = np.array(train_dataset)
        self.num_neurons = n = self.train_dataset[0].shape[0]
        self.num_synapses = self.num_neurons ** 2 - self.num_neurons

        n_tup = (n, n)

        self.M = np.zeros(n_tup)
        m = np.zeros(n_tup)
        for image_vector in t:
            for i in range(n):
                for j in range(i + 1, n):
                    m[i][j] = m[j][i] = np.multiply(image_vector[i], image_vector[j])
            self.M += m
        print self.M


def main():
    mnist = fetch_mldata('MNIST original', data_home='.cache')
    mnist_filtered = unpack(filter(lambda item: 1 <= item[1] and item[1] <= 5, zip(mnist.data, mnist.target)))[0]
    hf = HopfieldNetwork(
        train_dataset=mnist_filtered[:5000]
        test_dataset=mnist_filtered[5000:5001]

    )

def unpack(a):
    return zip(*a)

if __name__ == '__main__':
    main()
