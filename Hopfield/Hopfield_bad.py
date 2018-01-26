import numpy as np
import matplotlib.pyplot as plt
import random


class Hopfield:
    def __init__(self, cities, max_iter=1000, tolerance=None, verbose=False):
        self._cities = cities
        self._n = cities.shape[0]
        self._distances = np.array(
            [[(np.linalg.norm(c1 - c2)) for c2 in cities] for c1 in cities])
        self._normalized_distances = self._distances / self._distances.max()

        self._max_iter = max_iter
        self._tolerance = tolerance
        self._verbose = verbose
        self.energy_list = []

        self._A = 500
        self._B = 500
        self._C = 200
        self._D = 500
        self._u0 = 0.02
        self._U = self.init_u()
        self._V = self._calc_V(self._U)
        self._time_step = 1e-6
        self._n_adjust = 1.5

        np.random.RandomState(1)

    def init_u(self):
        u = np.ones([self._n, self._n]) / self._n ** 2
        for x in range(self._n):
            for y in range(self._n):
                u[x][y] += ((np.random.random() - 0.5) / 10000)
        return u

    def _calc_V(self, U):
        return self._sigmoid(U)

    def _sigmoid(self, x):
        return 0.5 * (1 + np.tanh(x / self._u0))

    def _get_item_D(self, x, i):
        sum = 0.0
        for y in range(self._n):
            left = self._V[y, (i + 1) % self._n]
            right = self._V[y, (i - 1)]
            sum += self._distances[x, y] * (left + right)
        return sum

    def _get_item_C(self):
        sum = np.sum(self._V)
        sum -= self._n + self._n_adjust
        return sum * self._C

    def _get_item_B(self, x, i):
        sum = np.sum(self._V[:, i])
        sum -= self._V[x, i]
        return sum * self._B

    def _get_item_A(self, x, i):
        sum = np.sum(self._V[x, :])
        sum -= self._sigmoid(self._V[x, i])
        return sum * self._A

    def _calc_delta_u(self, x, i):
        delta_u = -self._U[x, i]
        delta_u -= self._get_item_A(x, i)
        delta_u -= self._get_item_B(x, i)
        delta_u -= self._get_item_C()
        delta_u -= self._get_item_D(x, i)
        return delta_u

    def _calc_energy(self):
        sum_A = 0
        for x in range(self._n):
            sum_A += sum(((vi * vj) for vi in self._V[x] for vj in self._V[x]))
            sum_A -= np.sum(self._V[x] * self._V[x])
        sum_A *= self._A / 2

        sum_B = 0
        for i in range(self._n):
            sum_B += sum(((vx * vy) for vx in self._V[:, i] for vy in self._V[:, i]))
            sum_B -= np.sum(self._V[:, i] * self._V[:, i])
        sum_B *= self._B / 2

        sum_C = (np.sum(self._V) - self._n)**2
        sum_C *= self._C / 2

        sum_D = 0
        for x in range(self._n):
            for i in range(self._n):
                for y in range(self._n):
                    left = self._V[y, (i + 1) % self._n]
                    right = self._V[y, (i - 1)]
                    sum_D += self._distances[x, y] * (left + right)
        sum_D *= self._D / 2

        return sum_A + sum_B + sum_C + sum_D

    def run(self):
        for it in range(1, self._max_iter + 1):
            delta_U = np.zeros((self._n, self._n))
            for x in range(self._n):
                for i in range(self._n):
                    delta_U[x, i] = self._time_step * self._calc_delta_u(x, i)
            self._U += delta_U
            self._V = self._calc_V(self._U)

            energy = self._calc_energy()
            self.energy_list.append(energy)

            if self._verbose:
                print('iter', it, 'energy', energy)

    @property
    def get_energy_list(self):
        return self.energy_list


def load_data(file):
    with open(file, 'r') as f:
        lines = f.readlines()
    cities = np.empty((len(lines), 2))
    for i, line in enumerate(lines):
        x, y = line.split(' ')
        cities[i, 0] = float(x)
        cities[i, 1] = float(y)
    return cities


if __name__ == '__main__':
    cities = load_data('data.txt')
    max_iter = 1000
    hopfield = Hopfield(cities, max_iter=max_iter, tolerance=None, verbose=True)
    hopfield.run()

    # h2 = HopfieldNet(hopfield._normalized_distances, 1000)
    # h2.run()

    energy_list = np.array(hopfield.energy_list)
    index = np.arange(1, max_iter + 1)
    plt.plot(index[100:], energy_list[100:])
    plt.show()
