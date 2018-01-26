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

        self.net_state_list = []

        # self.seed = seed
        # random.seed(1)

        # values taken from paper
        # self.size = len(distances)

        self.inputs_change = np.zeros([self._n, self._n], float)
        self._A = 500
        self._B = 500
        self._C = 200
        self._D = 500

        self._u0 = 0.02
        self._time_step = 0.000001
        self.distances = self._normalized_distances

        # self.size_adj = size_adj
        self.size_adj = 1.5

        self.inputs = self.init_inputs()

        self._V = np.empty((self._n, self._n))
        self.energy_list = []

    def init_inputs(self):
        base = np.ones([self._n, self._n], float)
        base /= self._n ** 2
        for x in range(0, self._n):
            for y in range(0, self._n):
                base[x][y] += ((random.random() - 0.5) / 10000)
        return base

    def activation(self, single_input):
        sigm = 0.5 * (1 + np.tanh(single_input / self._u0))
        return sigm

    def get_a(self, city, position):
        sum = np.sum(self.activation(self.inputs[city, :]))
        sum -= self.activation(self.inputs[city, position])
        return sum * self._A

    def get_b(self, main_city, position):
        sum = np.sum(self.activation(self.inputs[:, position]))
        sum -= self.activation(self.inputs[main_city][position])
        return sum * self._B

    def get_c(self):
        sum = np.sum(self.activation(self.inputs[:, :]))
        sum -= self._n + self.size_adj
        return sum * self._C

    def get_d(self, main_city, position):
        return self.get_neighbours_weights(main_city, position) * self._D

    def get_neighbours_weights(self, main_city, position):
        sum = 0.0
        for city in range(0, self._n):
            preceding = self.activation(self.inputs[city, (position + 1) % self._n])
            following = self.activation(self.inputs[city, (position - 1)])
            sum += self.distances[main_city][city] * (preceding + following)
        return sum

    def get_states_change(self, city, pos):
        new_state = -self.inputs[city][pos]
        new_state -= self.get_a(city, pos)
        new_state -= self.get_b(city, pos)
        new_state -= self.get_c()
        new_state -= self.get_d(city, pos)
        return new_state

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

        sum_C = (np.sum(self._V) - self._n) ** 2
        sum_C *= self._C / 2

        sum_D = 0
        for x in range(self._n):
            for i in range(self._n):
                for y in range(self._n):
                    left = self._V[y, (i + 1) % self._n]
                    right = self._V[y, (i - 1)]
                    sum_D += self.distances[x, y] * (left + right)
        sum_D *= self._D / 2

        return sum_A + sum_B + sum_C + sum_D

    def run(self):
        for step in range(self._max_iter):
            self.inputs_change = np.zeros([self._n, self._n], float)
            for city in range(0, self._n):
                for pos in range(0, self._n):
                    self.inputs_change[city, pos] = self._time_step * self.get_states_change(city, pos)

            self.inputs += self.inputs_change

            self._V = self.activation(self.inputs)
            energy = self._calc_energy()
            self.energy_list.append(energy)

            if self._verbose:
                print('iter', step, end=' ')
                print(energy)

            if step % 20 == 0:
                self.net_state_list.append(self.get_net_state())
                net_state = self.get_net_state()
                title = 'title'
                path = '...'
                with Plotter(6, title, path) as ploter:
                    ploter.add_subplot(net_state["activations"], 'hot', 0, 1, f"Activations")
                    ploter.add_subplot(net_state["inputs"], 'coolwarm', -0.075, 0.075, f"Outputs of neurons")
                    ploter.add_subplot(net_state["inputsChange"], 'Blues_r', -0.001, 0, f"Negative change")
                    ploter.add_subplot(net_state["inputsChange"], 'Reds', 0, 0.001, f"Positive change")
                    # ploter.add_subplot(distances, 'plasma', 0, 1, f"Distance matrix")
                    ploter.add_graph(self.get_map(net_state["activations"], cities))

    def get_map(self, acts, cords):
        points = []
        for pos in range(0, len(acts)):
            for city in range(0, len(acts)):
                if acts[city][pos] > 0.6:
                    points.append(cords[city])

        return points

    def activations(self):
        return self.activation(self.inputs)

    def get_net_configuration(self):
        return {"a": self._A, "b": self._B, "c": self._C, "d": self._D, "u0": self._u0,
                "size_adj": self.size_adj, "timestep": self._time_step}

    def get_net_state(self):
        return {"activations": self.activations().tolist(),
                "inputs": self.inputs.tolist(),
                "inputsChange": self.inputs_change.tolist()}

class Plotter:
    def __init__(self, subplots, title, path):
        self.fig = plt.figure(figsize=(30, 10), dpi=50)
        self.subplot = 1
        self.expected_subplots = subplots
        self.title = title
        self.path = path

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.plot()

    def add_subplot(self, points, cmap, vmin, vmax, title):
        self.fig.add_subplot(1, self.expected_subplots, self.subplot)
        self.subplot += 1

        plt.imshow(points, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
        plt.title(title)
        # plt.colorbar()

    def add_graph(self, points):
        splot = self.fig.add_subplot(1, self.expected_subplots, self.subplot)
        self.subplot += 1
        if points and len(points) > 2:
            previous = points[0]
            for point in points[1:]:
                if point[0] != previous[0] and point[1] != previous[1]:
                    splot.arrow(previous[0],
                                previous[1],
                                point[0] - previous[0],
                                point[1] - previous[1],
                                head_width=0.0, head_length=0.0, width=0.0001, fc='k', ec='k')
                previous = point

            splot.set_xlim(min([p[0] for p in points]), max([p[0] for p in points]))
            splot.set_ylim(min([p[1] for p in points]), max([p[1] for p in points]))

    def plot(self):
        plt.suptitle(self.title)
        plt.show()
        # plt.savefig(self.path)
        # plt.close()


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
    max_iter = 1501
    hopfield = Hopfield(cities, max_iter=max_iter, tolerance=None, verbose=True)
    hopfield.run()

    # h2 = HopfieldNet(hopfield._normalized_distances, 1000)
    # h2.run()

    energy_list = np.array(hopfield.energy_list)
    index = np.arange(1, max_iter + 1)
    plt.plot(index[100:], energy_list[100:])
    plt.show()

    # title = 'title'
    # path = '...'

    # net_state_list = hopfield.net_state_list
    # for net_state in net_state_list:
    #     print('image')
    #     with Plotter(6, title, path) as ploter:
    #         ploter.add_subplot(net_state["activations"], 'hot', 0, 1, f"Activations")
    #         ploter.add_subplot(net_state["inputs"], 'coolwarm', -0.075, 0.075, f"Outputs of neurons")
    #         ploter.add_subplot(net_state["inputsChange"], 'Blues_r', -0.001, 0, f"Negative change")
    #         ploter.add_subplot(net_state["inputsChange"], 'Reds', 0, 0.001, f"Positive change")
    #         # ploter.add_subplot(distances, 'plasma', 0, 1, f"Distance matrix")
    #         # ploter.add_graph(self.get_map(net_state["activations"], cords))
