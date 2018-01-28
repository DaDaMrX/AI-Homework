import numpy as np
import matplotlib.pyplot as plt


class Gene:
    def __init__(self, distance, alpha, path=None):
        self.distance = distance
        self.n_cities = len(distance)
        self.alpha = alpha

        if path is None:
            self.path = list(range(self.n_cities))
            np.random.shuffle(self.path)
        else:
            self.path = path.copy()

        self.length = self._calc_length(self.path)
        self.score = self._calc_score(self.length)

    def _calc_length(self, path):
        length = 0
        for i in range(self.n_cities - 1):
            length += self.distance[path[i], path[i + 1]]
        length += self.distance[path[-1], path[0]]
        return length

    def _calc_score(self, length):
        return 1 / np.power(length, self.alpha)


class GA:
    def __init__(self, cities, n_genes=100, n_children=100,
                 alpha=50, recombination_prob=0.9, mutation_prob=1):
        self.cities = cities
        self.n_cities = len(cities)
        self.distance = np.empty((self.n_cities, self.n_cities))
        for i in range(self.n_cities):
            for j in range(self.n_cities):
                vector = (cities[i][0] - cities[j][0],
                          cities[i][1] - cities[j][1])
                self.distance[i, j] = np.linalg.norm(vector)

        self.n_genes = n_genes
        self.n_children = n_children

        self.alpha = alpha
        self.recombination_prob = recombination_prob
        self.mutation_prob = mutation_prob

        self.genes = [Gene(self.distance, self.alpha)
                      for i in range(self.n_genes)]

    def _adjust(self, path1, path2, l, r):
        while len(set(path1)) < self.n_cities:
            for i in list(range(0, l)) + list(range(r + 1, self.n_cities)):
                if path1.count(path1[i]) > 1:
                    p1 = i
                    break
            p2 = path1.index(path1[p1], l, r + 1)
            path1[p1] = path2[p2]

    def _recombine(self, path1, path2):
        path1 = path1.copy()
        path2 = path2.copy()
        if np.random.random() < self.recombination_prob:
            l, r = np.random.choice(range(1, self.n_cities - 1),
                                    size=2, replace=False)
            l, r = min(l, r), max(l, r)
            path1[l:r + 1], path2[l:r + 1] = path2[l:r + 1], path1[l:r + 1]
            self._adjust(path1, path2, l, r)
            self._adjust(path2, path1, l, r)
        return path1, path2

    def _mutate(self, path):
        if np.random.random() < self.mutation_prob:
            p1, p2 = np.random.choice(range(self.n_cities), 2, replace=False)
            path[p1], path[p2] = path[p2], path[p1]

    def _select(self, genes, number):
        weights = [x.score for x in genes]
        sum_weight = sum(weights)
        weights = [x / sum_weight for x in weights]
        samples = np.random.choice(genes, number, p=weights, replace=False)
        return list(samples)

    def evolve_once(self):
        children = []
        for i in range(self.n_children // 2):
            parents = self._select(self.genes, 10)
            parents.sort(key=lambda x: x.length)
            path1, path2 = self._recombine(parents[0].path, parents[1].path)
            self._mutate(path1)
            self._mutate(path2)
            children.append(Gene(self.distance, self.alpha, path1))
            children.append(Gene(self.distance, self.alpha, path2))
        self.genes += children
        self.genes = self._select(self.genes, self.n_genes)
        best_gene = max(self.genes, key=lambda x: x.length)
        return best_gene.length, best_gene.path

    def evolve(self, max_iter, verbose=None, plot_interval=None):
        self.genes = [Gene(self.distance, self.alpha)
                      for i in range(self.n_genes)]
        best_gene = max(self.genes, key=lambda x: x.length)
        best_length, best_path = best_gene.length, best_gene.path
        length_list = [best_length]
        if verbose is not None:
            self._log(0, best_length, best_path)
        if plot_interval:
            self.plot_path(0, best_length, best_path)

        for i in range(1, max_iter + 1):
            length, path = self.evolve_once()
            length_list.append(length)

            if length < best_length:
                best_length, best_path = length, path
                if verbose == 1:
                    self._log(i, best_length, best_path)

            if verbose == 2:
                self._log(i, best_length, best_path)
            if plot_interval is not None\
                    and plot_interval > 0\
                    and i % plot_interval == 0:
                plot_path(i, best_length, best_path, self.cities)

        if plot_interval is not None:
            plot_result(length_list, iter_start=0)
        return best_length, best_path

    def _log(self, iter_time, length, path):
        print(f'Iterate: {iter_time}, Length: {length:.4f}, Path: {path}')


def plot_path(iter_time, length, path, cities):
    n_cities = len(cities)
    for i in range(-1, n_cities - 1):
        x = [cities[path[i]][0], cities[path[i + 1]][0]]
        y = [cities[path[i]][1], cities[path[i + 1]][1]]
        plt.plot(x, y)
    plt.title(f'Iterate: {iter_time},  Path length: {length:.4f}')
    plt.show()


def plot_result(length_list, iter_start=1):
    n_points = len(length_list)
    x = list(range(iter_start, iter_start + n_points))
    plt.plot(x, length_list)
    plt.xlabel('Iterations')
    plt.ylabel('Path length')
    best_length = min(length_list)
    plt.title(f'Final length: {best_length:.4f}')
    plt.show()


def load_data(file):
    with open(file, 'r') as f:
        lines = f.readlines()
    lines = list(map(lambda c: c.split(' '), lines[6:]))
    cities = [(float(c[1]), float(c[2])) for c in lines]
    return cities


if __name__ == '__main__':
    cities = load_data('berlin52.tsp')
    ga = GA(cities, n_genes=100, n_children=100, alpha=50,
            recombination_prob=0.9, mutation_prob=1)
    ga.evolve(max_iter=500, verbose=1, plot_interval=0)
