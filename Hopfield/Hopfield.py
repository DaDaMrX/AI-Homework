import numpy as np
import matplotlib.pyplot as plt


class Hopfield:
    def __init__(self, cities):
        self.cities = cities
        self.n_cities = len(cities)
        shape = (self.n_cities, self.n_cities)

        self.distance = np.empty(shape)
        for i in range(self.n_cities):
            for j in range(i, self.n_cities):
                vector = (cities[i][0] - cities[j][0],
                          cities[i][1] - cities[j][1])
                self.distance[i, j] = np.linalg.norm(vector)
                self.distance[j, i] = self.distance[i, j]
        self.actual_distance = self.distance
        self.distance = self.distance / self.distance.max()

        self.a = 500
        self.b = 500
        self.c = 200
        self.d = 500

        self.u0 = 0.02
        self.alpha = 1e-6
        self.size_adjust = 1.5

        self.U = np.ones(shape)
        self.U /= self.n_cities ** 2
        self.U += np.random.uniform(-0.5, 0.5, shape) / 10000
        self.delta_U = np.zeros(shape)
        self.V = self._sigmoid(self.U)

    def _get_item_a(self, city, pos):
        value = np.sum(self.V[city, :])
        value -= self.V[city, pos]
        return value * self.a

    def _get_item_b(self, city, pos):
        value = np.sum(self.V[:, pos])
        value -= self.V[city, pos]
        return value * self.b

    def _get_item_c(self):
        value = np.sum(self.V)
        value -= self.n_cities + self.size_adjust
        return value * self.c

    def _get_item_d(self, city, pos):
        value = 0.0
        for x in range(self.n_cities):
            left = self.V[x, pos - 1]
            right = self.V[x, (pos + 1) % self.n_cities]
            value += self.distance[city, x] * (right + left)
        return value * self.d

    def _get_delta(self, city, pos):
        delta = -self.U[city, pos]
        delta -= self._get_item_a(city, pos)
        delta -= self._get_item_b(city, pos)
        delta -= self._get_item_c()
        delta -= self._get_item_d(city, pos)
        return delta

    def _sigmoid(self, u):
        return 0.5 * (1 + np.tanh(u / self.u0))

    def update_once(self, with_energy=False):
        self.delta_U = np.zeros((self.n_cities, self.n_cities))
        for city in range(self.n_cities):
            for pos in range(self.n_cities):
                self.delta_U[city, pos] = \
                    self.alpha * self._get_delta(city, pos)
        self.U += self.delta_U
        self.V = self._sigmoid(self.U)

    def update(self, max_iter, verbose=None, plot_iter=None):
        for i in range(1, max_iter + 1):
            hop.update_once()
            if verbose:
                self._log(i)
            if plot_iter and i % plot_iter == 0:
                self._plot_matrix(hop.V, i)

        path = np.array(hop.V).argmax(0)
        if plot_iter:
            self._plot_path(path)
        return path.tolist()

    def _log(self, iter_step):
        print(f'Iterate: {iter_step}')

    def _plot_matrix(self, matrix, iter_step):
        plt.imshow(matrix, cmap='hot',
                   vmin=0, vmax=1,
                   interpolation='nearest')
        plt.title(f'Iterate: {iter_step}')
        plt.show()

    def _plot_path(self, path):
        length = 0
        for i in range(-1, self.n_cities - 1):
            length += self.actual_distance[path[i], path[i + 1]]
            x = [self.cities[path[i]][0], self.cities[path[i + 1]][0]]
            y = [self.cities[path[i]][1], self.cities[path[i + 1]][1]]
            plt.plot(x, y)
        plt.title(f'Path length: {length:.4f}')
        plt.show()


def load_data(file):
    with open(file, 'r') as f:
        lines = f.readlines()
    lines = lines[6:]
    lines = lines[:-1]
    lines = [c.split(' ') for c in lines]
    cities = [(float(c[1]), float(c[2])) for c in lines]
    return cities


if __name__ == '__main__':
    cities = load_data('burma14.tsp')
    hop = Hopfield(cities)
    path = hop.update(1500, verbose=True, plot_iter=200)
    print(path)

