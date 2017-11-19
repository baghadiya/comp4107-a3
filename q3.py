"""
Feed forward Neural Network for classifying faces using the the dataset
provided by scikit learn at http://scikit-learn.org/0.18/auto_examples/applications/face_recognition.html.

Note that the data used contains faces that only have had at least 70 pictures taken of them. In total
there are 7 different people to classify.

The following code has been writen with Python 3.5
"""
import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold

EPOCHS = 100
PCA_COMPONENTS = 75


def load_data(pca=False):
    """
    Loads the cached scikit learn data.
    """
    data = np.load('.cache/mldata/lfw_people_data.npy')
    target = np.load('.cache/mldata/lfw_people_target.npy')

    # One-hot the label data
    labels = np.zeros(shape=(target.shape[0], target.max() + 1))
    for i, x in enumerate(target):
        labels[i][x] = 1

    if pca:
        # Compute principle components of input data
        data = PCA(n_components=PCA_COMPONENTS, svd_solver='randomized', whiten=True).fit_transform(data)
    else:
        # Regularize the input data
        data = data / data.max()

    return data, labels


class FacialRecognitionNetwork(object):
    def __init__(self, hidden_layer1, hidden_layer2, output_layer, input_size=1850, learning_rate=0.002):
        """
        Initializes the neural network.
        :param hidden_layer1: Size of the first hidden layer used in the network.
        :param output_layer: Size of the output layer used in the network.
        :param input_size: Size of the input vectors fed into the network.
        """
        self.size_h1 = tf.constant(hidden_layer1, dtype=tf.int32)
        self.size_h2 = tf.constant(hidden_layer2, dtype=tf.int32)
        self.X = tf.placeholder("float", [None, input_size])
        self.Y = tf.placeholder("float", [None, output_layer])

        # Initialize weight matrices
        self.w_h1 = self.__init_weights(input_size, hidden_layer1)
        self.w_h2 = self.__init_weights(hidden_layer1, hidden_layer2)
        self.w_o = self.__init_weights(hidden_layer2, output_layer)

        # Initialize the Neural Network model
        self.py_x = self.__model()
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.py_x, labels=self.Y))
        self.train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.cost)
        self.predict_op = tf.argmax(self.py_x, 1)

        # Cross-Validation attributes
        self.accuracies = []
        self.mean_accuracy = None

    @staticmethod
    def __init_weights(s1, s2):
        sigma = math.sqrt(2) * math.sqrt(2 / (s1 + s2))
        return tf.Variable(tf.random_normal([s1, s2], stddev=sigma))

    def __model(self):
        h1 = tf.nn.leaky_relu(tf.matmul(self.X, self.w_h1), alpha=0.2)
        h2 = tf.nn.leaky_relu(tf.matmul(h1, self.w_h2), alpha=0.2)
        return tf.matmul(h2, self.w_o)

    def cross_validate(self, data, labels):
        """
        Perform K-Fold cross validation to test the network architecture.

        :param data: Input image vectors
        :param labels: Labels for the given input image vectors
        """
        print("Performing K-Fold cross validation...")
        self.accuracies = []
        fold = 1
        for train_index, test_index in KFold(n_splits=10).split(data):
            trX, teX = data[train_index], data[test_index]
            trY, teY = labels[train_index], labels[test_index]

            acc = 0
            with tf.Session() as sess:
                tf.global_variables_initializer().run()
                for i in range(EPOCHS):
                    for start, end in zip(range(0, len(trX), 128), range(128, len(trX) + 1, 128)):
                        sess.run(self.train_op, feed_dict={self.X: trX[start:end], self.Y: trY[start:end]})
                    acc = np.mean(np.argmax(teY, axis=1) == sess.run(self.predict_op, feed_dict={self.X: teX}))

            self.accuracies.append(acc)
            fold += 1

        self.mean_accuracy = np.mean(self.accuracies)

    def plot_accuracies(self, title):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(self.accuracies)

        plt.xlabel('Fold Iteration')
        plt.ylabel('Accuracy (%)')
        plt.title(title)
        plt.show()
