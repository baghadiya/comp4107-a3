from numpy.linalg import norm

class SelfOrganizingMap(object):
    def __init__(self, trainset=[], p=1., q=1., learning_rate=0.1, rad=5):
        self.n = len(trainset)
        self.trainset = trainset

        self.learning_rate = learning_rate
        self.radius = rad
        self.p = p
        self.q = q
        self.W = np.random.uniform(-.5, .5, (p, q))

    def train(self):
        p_range = range(self.p)
        q_range = range(self.q)
        for x in self.trainset:
            best_candidate = (0, 0)
            min_norm = float('inf')

            for i in p_range:
                for j in q_range:
                    d_ij = norm(x - self.W[i,j])
                    if d_ij < min_norm:
                        min_norm = d_ij
                        best_candidate = self.W[i,j]
            for i in p_range:
                for j in q_range:
                    self.W[i,j] += self.learning_rate * self.h(best_candidate, self.W[i,j], self.radius) * min_norm

    def update_unit(self):
        """
        """
    def h(self, w_ij, w_mn):
        return -1
