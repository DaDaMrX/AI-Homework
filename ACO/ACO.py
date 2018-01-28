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

    def _update_pheromone(self, path_list):
        pheromones = np.zeros((self.n_cities, self.n_cities))
        for path_length, path in path_list:
            amount_pheromone = self.Q / path_length
            for i in range(-1, self.n_cities - 1):
                pheromones[path[i], path[i + 1]] += amount_pheromone
                pheromones[path[i + 1], path[i]] += amount_pheromone
        self.pheromone = self.pheromone * self.rho + pheromones

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

    def search_once(self):
        path_list = []
        cur_best_length = float('inf')
        for i in range(self.n_ants):
            path = self._ant_go()
            path_length = self._calc_path_length(path)
            path_list.append((path_length, path))

            if path_length < cur_best_length:
                cur_best_length = path_length

            if path_length < self.best_path_length:
                self.best_path_length = path_length
                self.best_path = path
        self._update_pheromone(path_list)
        return self.best_path_length, self.best_path, cur_best_length

    def search(self, max_iter=200, verbose=None, visualize=False):
        best_length = float('inf')
        best_path = None
        length_list = []
        for i in range(1, max_iter + 1):
            length, path, cur_best_length = self.search_once()
            length_list.append(cur_best_length)
            # length_list.append(length)

            if length < best_length:
                best_length, best_path = length, path
                if verbose == 1:
                    print(f'[Update] Iterate: {i:2d}, '
                          f'Best length: {best_length:.6f}, '
                          f'Best path: {path}')
                if visualize:
                    for i in range(-1, self.n_cities - 1):
                        x = [cities[best_path[i]][0], cities[best_path[i + 1]][0]]
                        y = [cities[best_path[i]][1], cities[best_path[i + 1]][1]]
                        plt.plot(x, y)
                    plt.title(f'Iterate: {i:2d},  Path length: {best_length:.6f}')
                    plt.show()

            if verbose == 2:
                print('Iterate: {i:2d}, '
                      'Best length: {best_length:.6f}, '
                      'Best path:', path)

        if visualize:
            X = np.arange(1, max_iter + 1)
            plt.plot(X, length_list)
            plt.xlabel('Iterations')
            plt.ylabel('Path length')
            plt.title(f'Change of path length\nShortest length: {best_length:.6f}' )
            plt.show()
        return best_length, best_path


def load_data(file):
    with open(file, 'r') as f:
        for i in range(6):
            f.readline()
        lines = f.readlines()
    cities = []
    for i, line in enumerate(lines):
        t = line[:-1].split(' ')
        cities.append((float(t[1]), float(t[2])))
    return cities


if __name__ == '__main__':
    cities = load_data('berlin52.tsp')
    aco = ACO(cities, n_ants=50, alpha=1, beta=2, rho=0.5, Q=100)
    aco.search(max_iter=200, verbose=1, visualize=True)
