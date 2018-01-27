"""Genetic Algorithm
"""
import random

import math


class GA:
    def __init__(self, cities, population_size, recombine_prob, mutate_prob):
        self.cities = cities
        self.n_cities = len(cities)

        self.population_size = population_size
        self.recombine_prob = recombine_prob
        self.mutate_prob = mutate_prob

        genes = self.population_init(self.population_size)
        scores = list(map(self.evaluate, genes.copy()))
        self.population = sorted(zip(genes, scores), key=lambda x: -x[1])
        self.best_gene, self.best_score = self.population[0]


    def population_init(self, population_size):
        permutation = list(range(self.n_cities))
        population = []
        for i in range(population_size):
            random.shuffle(permutation)
            population.append(permutation.copy())
        return population

    def evaluate(self, gene):
        return 1 / self.length(gene)

    def length(self, gene):
        s = 0
        for i in range(-1, len(gene) - 1):
            s += self._distance(self.cities[gene[i]], self.cities[gene[i + 1]])
        return s

    def _distance(self, city1, city2):
        dx = city1[0] - city2[0]
        dy = city1[1] - city2[1]
        return math.sqrt(dx * dx + dy * dy)

    def recombine(self, gene1, gene2, recombine_prob):
        if random.random() < recombine_prob:
            gene1, gene2 = gene1.copy(), gene2.copy()
            pos1, pos2 = sorted(random.sample(range(len(gene1)), 2))
            gene1[pos1:pos2 + 1], gene2[pos1:pos2 + 1] = gene2[pos1:pos2 + 1], gene1[pos1:pos2 + 1]
            self._adjust_genes(gene1, gene2, pos1, pos2)
        return gene1, gene2

    def _adjust_genes(self, gene1, gene2, pos1, pos2):
        while not self._is_unique(gene1):
            self._adjust(gene1, gene2, pos1, pos2)
        while not self._is_unique(gene2):
            self._adjust(gene2, gene1, pos1, pos2)

    def _is_unique(self, gene):
        for x in gene:
            if gene.count(x) > 1:
                return False
        return True

    def _adjust(self, gene1, gene2, pos1, pos2):
        for i in list(range(0, pos1)) + list(range(pos2 + 1, len(gene1))):
            if gene1[i] in gene1[pos1:pos2 + 1]:
                index = gene1[pos1:pos2 + 1].index(gene1[i]) + pos1
                gene1[i] = gene2[index]

    def mutate(self, gene, mutate_prob):
        gene = gene.copy()
        for i in range(1, 1001):
            if random.random() < 1 / i * mutate_prob:
                pos1, pos2 = random.sample(range(len(gene)), 2)
                gene[pos1], gene[pos2] = gene[pos2], gene[pos1]
        return gene

    def select(self, population):
        scores = [biont[1] for biont in population]
        sum_score = sum(scores)
        normalized_scores = list(map(lambda x: x / sum_score, scores))
        r = random.random()
        for i in range(self.n_cities):
            r -= normalized_scores[i]
            if r < 0.0:
                return population[i][0]
        return population[-1][0]

    def evolve(self):
        parent_size = int(0.2 * self.population_size)
        parent_genes = []
        for i in range(parent_size):
            gene = self.select(self.population)
            parent_genes.append(gene)

        child_size = int(1 * self.population_size)
        child_genes = []
        for i in range(child_size):
            gene1, gene2 = random.sample(parent_genes, 2)
            gene1, gene2 = self.recombine(gene1, gene2, self.recombine_prob)
            gene1 = self.mutate(gene1, self.mutate_prob)
            gene2 = self.mutate(gene2, self.mutate_prob)
            child_genes.append(gene1)
            child_genes.append(gene2)

        genes = child_genes + [x[0] for x in self.population]
        scores = list(map(self.evaluate, genes.copy()))
        self.population = sorted(zip(genes, scores), key=lambda x: -x[1])

        genes = []
        for i in range(self.population_size):
            gene = self.select(self.population)
            genes.append(gene)

        scores = list(map(self.evaluate, genes.copy()))
        self.population = sorted(zip(genes, scores), key=lambda x: -x[1])

        if self.population[0][1] > self.best_score:
            self.best_gene, self.best_score = self.population[0]


# def evolve(self):
    #     parents = self.select(self.population, 0.2 * self.population_size)
    #     children = self.generate(parents, self.population_size)
    #     self.population.append(children)
    #     self.population = self.select(self.population, self.population_size)

        # genes = []
        # while len(genes) < self.population_size:
        #     gene1 = self.select(self.population)
        #     gene2 = self.select(self.population)
        #     gene1, gene2 = self.recombine(gene1, gene2, self.recombine_prob)
        #     gene1 = self.mutate(gene1, self.mutate_prob)
        #     gene1 = self.mutate(gene1, self.mutate_prob)
        #     genes.append(gene1)
        #     genes.append(gene2)

        # scores = list(map(self.evaluate, genes.copy()))
        # self.population = sorted(zip(genes, scores), key=lambda x: -x[1])

        # if self.population[0][1] > self.best_score:
        #     self.best_gene, self.best_score = self.population[0]


def load_data(file):
    with open(file, 'r') as f:
        lines = f.readlines()
    cities = []
    for i, line in enumerate(lines):
        t = line[:-1].split(' ')
        index, x, y = t
        cities.append((float(x), float(y)))
    return cities


if __name__ == '__main__':
    cities = load_data('berlin52.tsp')
    ga = GA(cities,
            population_size=100,
            recombine_prob=0.8,
            mutate_prob=1)

    best_gene = ga.best_gene
    best_score = ga.best_score
    length = ga.length(best_gene)
    print('It:', -1, 'Length:', f'{length:.4f}', 'Path:', best_gene)

    for i in range(1000):
        ga.evolve()
        if ga.best_score > best_score:
            best_score = ga.best_score
            best_gene = ga.best_gene
            length = ga.length(best_gene)
            print('It:', i, 'Length:', f'{length:.4f}', 'Path:', best_gene)
        else:
            print('It:', i)

    # print('It:', 'final', 'Length:', length, 'Path:', best_gene)







