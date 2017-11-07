#!/usr/bin/env python
from sklearn.datasets import fetch_mldata
import numpy as np
import random
import matplotlib.pyplot as plt

def show(vector):
    plt.imshow(np.array(vector).reshape(28, 28))
    plt.show()

def unpack(a):
    return zip(*a)

class HopfieldNetwork(object):
    def __init__(self, train_dataset=[], test_dataset=[], threshold=60, theta=0.5, tolerance=.05):
        self.tolerance = .335;
        self.threshold = threshold
        self.theta = theta
        self.train_dataset = t = np.array(train_dataset)
        self.num_neurons = n = self.train_dataset[0].shape[0]

        self.W = np.zeros((n,n))
        for image_vector in t:
            p = np.matrix(image_vector)
            self.W += p.T * p
        self.W -= len(t) * np.identity(len(t[0]))

    def activate(self, vector):
        old_energy = energy = 0
        while True:
            s = 0
            neuron_index = random.randint(0, len(vector) - 1)
            for pixel_index in range(len(vector)):
                pixel = vector[pixel_index]
                if pixel > 0:
                    s += self.W[neuron_index][pixel_index]
            if s >= 0:
                vector[neuron_index] = 1
            elif s < 0:
                vector[neuron_index] = 0
            ww = 0
            for j in range(784):
                for i in range(784):
                    ww += self.W[i,j] * vector[j] * vector[i]
            energy = -(1./2) * ww

            if energy - old_energy <= self.tolerance:
                break;
        return vector
    #  def classify(self, vector):
    #     vector = vector[:]
    #     l = len(vector)
    #     for s in range(self.time):
    #         unit = random.randint(0, l - 1)
    #         u = np.dot(self.W[unit][:], vector)
     #
    #         if u > 0:
    #             vector[unit] = 1
    #         elif u < 0:
    #             vector[unit] = 0
     #
    #     return map(lambda x: int(x >= self.threshold), vector)

mnist = fetch_mldata('MNIST original', data_home='.cache')

mnist_filtered = [[float(pixel != 0) for pixel in vector] for vector in mnist.data[:10000]]

hf = HopfieldNetwork(
    train_dataset=mnist_filtered[2000:2050]
)
# v = hf.classify(mnist_filtered[11])
# show(mnist_filtered[11])
# show(v)
