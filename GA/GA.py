import numpy as np
import matplotlib.pyplot as plt


class Agent:
    def __init__(self, distance, solution=None):
        self.distance = distance
        self.n_cities = len(distance)

        if solution is None:
            self.path = np.arange(1, self.n_cities + 1)
            np.random.shuffle(self.path)
        else:
            self.path = solution.copy()  # Must be the copy

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

        self.n_agents = n_genes
        self.n_parents = n_parents
        self.n_children = n_children
        self.recombination_prob = recombination_prob
        self.mutation_prob = mutation_prob
        self.mutation_times = mutation_times

        self.agents = [Agent(self.distance) for i in range(self.n_agents)]

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

    def generate(self, agent1, agent2):
        if np.random.random() < self.recombination_prob:
            path1 = list(agent1.path.copy())
            path2 = list(agent2.path.copy())
            pos1, pos2 = sorted(np.random.choice(range(agent1.n_cities - 1), size=2, replace=False))
            path1[pos1 + 1: pos2 + 1], path2[pos1 + 1: pos2 + 1] = path2[pos1 + 1: pos2 + 1], path1[pos1 + 1: pos2 + 1]
            path1 = self.adjust(path1, path2)
            path2 = self.adjust(path2, path1)
        else:
            path1 = list(range(agent1.n_cities))
            np.random.shuffle(path1)
            path2 = list(range(agent1.n_cities))
            np.random.shuffle(path2)

        if np.random.random() < self.mutation_prob:
            for i in range(1, self.mutation_times + 1):
                if np.random.random() < 1 / i:
                    x1, y1 = np.random.choice(range(agent1.n_cities), size=2, replace=False)
                    path1[x1], path1[y1] = path1[y1], path1[x1]
                    x2, y2 = np.random.choice(range(agent1.n_cities), size=2, replace=False)
                    path2[x2], path2[y2] = path2[y2], path2[x2]

        return path1, path2

    def generate_children(self, parents, n_children):
        children = []
        for i in range(n_children // 2):
            f1, f2 = np.random.choice(parents, size=2, replace=False)
            embryo1, embryo2 = self.generate(f1, f2)
            children.append(Agent(self.distance, embryo1))
            children.append(Agent(self.distance, embryo2))
        return children

    def select(self, agents, number):
        n_agents = len(agents)
        agents = sorted(agents, key=lambda x: x.length)
        sum_length = sum((agent.length for agent in agents))

        parents = []
        exist = []
        c_number = 0
        index = 0
        while True:
            god = np.random.uniform()
            if god > (agents[index].length / sum_length):
                if index in exist:
                    index = (index + 1) % n_agents
                    continue
                c_number += 1
                parents.append(agents[index])
                exist.append(index)
            if c_number > number:
                break
            index = (index + 1) % len(agents)

        return parents

    def evelove_once(self):
        parents = self.select(self.agents, self.n_parents)
        self.agents += self.generate_children(parents, self.n_children)
        self.agents = self.select(self.agents, self.n_agents)
        best_agant = max(self.agents, key=lambda x: x.length)
        return best_agant.length, best_agant.path

    def evelove(self, max_iter, verbose=None, plot_iter=None):
        self.agents = [Agent(self.distance) for i in range(self.n_agents)]
        best_length = float('inf')
        best_path = None
        length_list = []
        for step in range(max_iter):
            length, path = self.evelove_once()
            length_list.append(length)
            if length < best_length:
                best_length, best_path = length, path
                if verbose == 1:
                    print(f'[Update] Iterate: {step:2d}, '
                          f'Best length: {best_length:.4f}, '
                          f'Best path: {path}')

            if verbose == 2:
                print('Iterate: {step:2d}, '
                      'Best length: {best_length:.4f}, '
                      'Best path:', path)

            if plot_iter is not None and step % plot_iter == 0:
                for i in range(-1, self.n_cities - 1):
                    x = [cities[best_path[i] - 1][0], cities[best_path[i + 1] - 1][0]]
                    y = [cities[best_path[i] - 1][1], cities[best_path[i + 1] - 1][1]]
                    plt.plot(x, y)
                plt.title(f'Iterate: {step},  Path length: {best_length:.4f}')
                plt.show()

        if plot_iter is not None:
            X = list(range(1, max_iter + 1))
            plt.plot(X, length_list)
            plt.xlabel('Iterations')
            plt.ylabel('Path length')
            plt.title(f'Generation Algorithm\nFinal length: {best_length:.4f}')
            plt.show()

        return best_length, best_path


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
    ga.evelove(max_iter=1000, verbose=1, plot_iter=50)
