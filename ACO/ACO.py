import numpy as np
import matplotlib.pyplot as plt


class ACO:
    def __init__(self, cities, n_ants=50, alpha=1, beta=2, rho=0.5, Q=100):
        self.cities = cities
        self.n_cities = len(cities)
        self.distance = np.empty((self.n_cities, self.n_cities))
        for i in range(self.n_cities):
            for j in range(self.n_cities):
                vector = (cities[i][0] - cities[j][0], cities[i][1] - cities[j][1])
                self.distance[i, j] = np.linalg.norm(vector)
        self.pheromone = np.ones((self.n_cities, self.n_cities))

        self.n_ants = n_ants
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q

        self.best_path_length = float('inf')
        self.best_path = None

    def _calc_path_length(self, path):
        length = 0.0
        for i in range(-1, self.n_cities - 1):
            length += self.distance[path[i]][path[i + 1]]
        return length

    def _choose_city(self, current_city, available_cities):
        city_probs = []
        for i in available_cities:
            pheromone = pow(self.pheromone[current_city][i], self.alpha)
            distance = pow(self.distance[current_city][i], self.beta)
            city_probs.append(pheromone / distance)
        sum_weight = sum(city_probs)
        city_probs = list(map(lambda x: x / sum_weight, city_probs))
        next_city = np.random.choice(available_cities, p=city_probs)
        return next_city

    def _ant_go(self):
        available_cities = list(range(self.n_cities))
        path = []

        current_city = np.random.choice(available_cities)
        available_cities.remove(current_city)
        path.append(current_city)

        for i in range(self.n_cities - 1):
            next_city = self._choose_city(current_city, available_cities)
            available_cities.remove(next_city)
            path.append(next_city)
            current_city = next_city

        return path

    def search_once(self):
        path_list = []
        best_length = float('inf')
        best_path = None
        for i in range(self.n_ants):
            path = self._ant_go()
            length = self._calc_path_length(path)
            path_list.append((length, path))
            if length < best_length:
                best_length, best_path = length, path
        self._update_pheromone(path_list)
        return best_length, best_path

    def search(self, max_iter=200, verbose=None, plot_interval=None):
        best_length = float('inf')
        best_path = None
        length_list = []
        for i in range(1, max_iter + 1):
            length, path = self.search_once()
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
            plot_result(length_list, iter_start=1)
        return best_length, best_path

    def _update_pheromone(self, path_list):
        pheromones = np.zeros((self.n_cities, self.n_cities))
        for path_length, path in path_list:
            amount_pheromone = self.Q / path_length
            for i in range(-1, self.n_cities - 1):
                pheromones[path[i], path[i + 1]] += amount_pheromone
                pheromones[path[i + 1], path[i]] += amount_pheromone
        self.pheromone = self.pheromone * self.rho + pheromones

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
    aco = ACO(cities, n_ants=50, alpha=1, beta=2, rho=0.5, Q=100)
    aco.search(max_iter=200, verbose=1, plot_interval=20)
