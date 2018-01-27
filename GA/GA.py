import numpy as np
import matplotlib.pyplot as plt


class Gene:
    def __init__(self, distance, path=None):
        self.distance = distance
        self.n_cities = len(distance)

        if path is None:
            # self.path = list(range(1, self.n_cities + 1))
            self.path = list(range(self.n_cities))
            np.random.shuffle(self.path)
        else:
            self.path = path.copy()  # Must be the copy

        self.length = self._calc_length(self.path)

    def _calc_length(self, path):
        length = 0
        for i in range(self.n_cities - 1):
            length += self.distance[path[i] - 1, path[i + 1] - 1]
        length += self.distance[path[-1] - 1, path[0] - 1]
        return length


class GA:
    def __init__(self, cities, n_genes=100, n_parents=20, n_children=200,
                 recombination_prob=0.9, mutation_prob=1, mutation_times=10):
        self.cities = cities
        self.n_cities = len(cities)
        self.distance = np.empty((self.n_cities, self.n_cities))
        for i in range(self.n_cities):
            for j in range(self.n_cities):
                vector = (cities[i][0] - cities[j][0], cities[i][1] - cities[j][1])
                self.distance[i, j] = np.linalg.norm(vector)

        self.n_genes = n_genes
        self.n_parents = n_parents
        self.n_children = n_children
        self.recombination_prob = recombination_prob
        self.mutation_prob = mutation_prob
        self.mutation_times = mutation_times

        self.genes = [Gene(self.distance) for i in range(self.n_genes)]

    def adjust(self, path1, path2):
        # this function try to check and fix two solution and return the right ans
        # fix the sol1 and return sol1 !! fuck !! do not need to return
        p = path1.copy()
        n_cities = len(path1)
        first = -1

        while len(set(p)) < n_cities:
            # exist the error, need to be fixed !
            for i in range(n_cities):
                if p.count(p[i]) > 1:
                    first = i
                    break
            try:
                second = p.index(p[first], first + 1)
                # Find the anothor index of the [i]
                p[first] = path2[second]
            except:
                # There may be two error
                #   1. do not exist the number
                #   2. first + 1 is over the boundary
                # But they all mean that there is single number of [i]
                continue
        return p

    def generate(self, gene1, gene2):
        if np.random.random() < self.recombination_prob:
            path1 = list(gene1.path.copy())
            path2 = list(gene2.path.copy())
            pos1, pos2 = sorted(np.random.choice(range(gene1.n_cities - 1), size=2, replace=False))
            path1[pos1 + 1: pos2 + 1], path2[pos1 + 1: pos2 + 1] = path2[pos1 + 1: pos2 + 1], path1[pos1 + 1: pos2 + 1]
            path1 = self.adjust(path1, path2)
            path2 = self.adjust(path2, path1)
        else:
            path1 = list(range(gene1.n_cities))
            np.random.shuffle(path1)
            path2 = list(range(gene1.n_cities))
            np.random.shuffle(path2)

        if np.random.random() < self.mutation_prob:
            for i in range(1, self.mutation_times + 1):
                if np.random.random() < 1 / i:
                    x1, y1 = np.random.choice(range(gene1.n_cities), size=2, replace=False)
                    path1[x1], path1[y1] = path1[y1], path1[x1]
                    x2, y2 = np.random.choice(range(gene1.n_cities), size=2, replace=False)
                    path2[x2], path2[y2] = path2[y2], path2[x2]

        return path1, path2

    def generate_children(self, parents, n_children):
        children = []
        for i in range(n_children // 2):
            f1, f2 = np.random.choice(parents, size=2, replace=False)
            embryo1, embryo2 = self.generate(f1, f2)
            children.append(Gene(self.distance, embryo1))
            children.append(Gene(self.distance, embryo2))
        return children

    def select(self, genes, number):
        n_genes = len(genes)
        genes = sorted(genes, key=lambda x: x.length)
        sum_length = sum((agent.length for agent in genes))

        parents = []
        exist = []
        c_number = 0
        index = 0
        while True:
            god = np.random.uniform()
            if god > (genes[index].length / sum_length):
                if index in exist:
                    index = (index + 1) % n_genes
                    continue
                c_number += 1
                parents.append(genes[index])
                exist.append(index)
            if c_number > number:
                break
            index = (index + 1) % len(genes)

        return parents

    def evolve_once(self):
        parents = self.select(self.genes, self.n_parents)
        self.genes += self.generate_children(parents, self.n_children)
        self.genes = self.select(self.genes, self.n_genes)
        best_gene = max(self.genes, key=lambda x: x.length)
        return best_gene.length, best_gene.path

    def evolve(self, max_iter, verbose=None, plot_interval=None):
        self.genes = [Gene(self.distance) for i in range(self.n_genes)]
        best_gene = max(self.genes, key=lambda x: x.length)
        best_length, best_path = best_gene.length, best_gene.path
        length_list = [best_length]
        if verbose is not None:
            self.log(0, best_length, best_path)
        if plot_interval:
            self.plot_path(0, best_length, best_path)

        for i in range(1, max_iter + 1):
            length, path = self.evolve_once()
            length_list.append(length)

            if length < best_length:
                best_length, best_path = length, path
                if verbose == 1:
                    self.log(i, best_length, best_path)

            if verbose == 2:
                self.log(i, best_length, best_path)
            if plot_interval is not None and i % plot_interval == 0:
                self.plot_path(i, best_length, best_path)

        if plot_interval is not None:
            self.plot_result(max_iter, length_list)
        return best_length, best_path

    def log(self, iter_time, length, path):
        print(f'Iterate: {iter_time:2d}, Length: {length:.4f}, Path: {path}')

    def plot_path(self, iter_time, length, path):
        for i in range(-1, self.n_cities - 1):
            x = [self.cities[path[i] - 1][0], self.cities[path[i + 1] - 1][0]]
            y = [self.cities[path[i] - 1][1], self.cities[path[i + 1] - 1][1]]
            plt.plot(x, y)
        plt.title(f'Iterate: {iter_time},  Path length: {length:.4f}')
        plt.show()

    def plot_result(self, max_iter, length_list):
        x = list(range(max_iter + 1))
        plt.plot(x, length_list)
        plt.xlabel('Iterations')
        plt.ylabel('Path length')
        best_length = max(length_list)
        plt.title(f'Generation Algorithm\nFinal length: {best_length:.4f}')
        plt.show()


def load_data(file):
    with open(file, 'r') as f:
        lines = f.readlines()
    lines = list(map(lambda c: c.split(' '), lines[6:]))
    cities = [(float(c[1]), float(c[2])) for c in lines]
    return cities


if __name__ == '__main__':
    np.random.seed(1)
    cities = load_data('berlin52.tsp')
    ga = GA(cities, n_genes=100, n_parents=20, n_children=200, recombination_prob=0.9,
            mutation_prob=1, mutation_times=10)
    ga.evolve(max_iter=100, verbose=1, plot_interval=10)
