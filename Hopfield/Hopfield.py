import numpy as np
import matplotlib.pyplot as plt


class HopfieldNet:
    def __init__(self, cities):
        self.cities = cities
        self.n_cities = len(cities)
        shape = (self.n_cities, self.n_cities)
        self.distance = np.empty(shape)
        for i in range(self.n_cities):
            for j in range(self.n_cities):
                vector = (cities[i][0] - cities[j][0],
                          cities[i][1] - cities[j][1])
                self.distance[i, j] = np.linalg.norm(vector)
        self.distance = self.distance / self.distance.max()

        self.a = 500
        self.b = 500
        self.c = 200
        self.d = 500

        self.u0 = 0.02
        self.time_step = 1e-6
        self.size_adj = 1.5

        self.U = np.ones(shape)
        self.U /= self.n_cities ** 2
        self.U += np.random.uniform(-0.5, 0.5, shape) / 10000
        self.delat_U = np.zeros(shape)

    def activation(self, single_input):
        return 0.5 * (1 + np.tanh(single_input / self.u0))

    def get_a(self, city, position):
        sum = np.sum(self.activation(self.U[city, :]))
        sum -= self.activation(self.U[city, position])
        return sum * self.a

    def get_b(self, main_city, position):
        sum = np.sum(self.activation(self.U[:, position]))
        sum -= self.activation(self.U[main_city][position])
        return sum * self.b

    def get_c(self):
        sum = np.sum(self.activation(self.U[:, :]))
        sum -= self.n_cities + self.size_adj
        return sum * self.c

    def get_d(self, main_city, position):
        return self.get_neighbours_weights(main_city, position) * self.d

    def get_neighbours_weights(self, main_city, position):
        sum = 0.0
        for city in range(0, self.n_cities):
            preceding = self.activation(self.U[city, (position + 1) % self.n_cities])
            following = self.activation(self.U[city, (position - 1)])
            sum += self.distance[main_city][city] * (preceding + following)
        return sum

    def get_states_change(self, city, pos):
        new_state = -self.U[city][pos]
        new_state -= self.get_a(city, pos)
        new_state -= self.get_b(city, pos)
        new_state -= self.get_c()
        new_state -= self.get_d(city, pos)
        return new_state

    def update(self):
        self.delat_U = np.zeros([self.n_cities, self.n_cities], float)
        for city in range(0, self.n_cities):
            for pos in range(0, self.n_cities):
                self.delat_U[city, pos] = self.time_step * self.get_states_change(city, pos)
        self.U += self.delat_U

    def get_net_state(self):
        return self.activation(self.U)


def load_data(file):
    with open(file, 'r') as f:
        lines = f.readlines()
    lines = list(map(lambda c: c.split(' '), lines[6:]))
    cities = [(float(c[1]), float(c[2])) for c in lines]
    return cities


def plot_path(path, cities):
    n_cities = len(cities)
    for i in range(-1, n_cities - 1):
        x = [cities[path[i]][0], cities[path[i + 1]][0]]
        y = [cities[path[i]][1], cities[path[i + 1]][1]]
        plt.plot(x, y)
    # plt.title(f'Iterate: {iter_time},  Path length: {length:.4f}')
    plt.show()


if __name__ == '__main__':
    np.random.seed(1)
    cities = load_data('burma14.tsp')
    net = HopfieldNet(cities)
    for step in range(1001):
        print(f'Iterate: {step}')
        net.update()
        if step % 200 == 0:
            points = net.get_net_state()
            plt.imshow(points, cmap='hot', vmin=0, vmax=1, interpolation='nearest')
            plt.show()

    a = net.get_net_state()
    a = np.array(a)
    path = a.argmax(0)
    print(path)
    plot_path(path, cities)

