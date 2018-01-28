#!/usr/bin/python3
# Author : GMFTBY
# Time   : 2017.1.16

'''
The GA ALgorithm for the TSP Question, use the TSPLIB Dataset

Link : 
    1. https://www.cnblogs.com 
    2. https://wiki.org

Generate Algorithm:
    1. Init the swam for the Question, decide the size of the swarm (Y) 
    2. Decide the fittness function for the agent in the swarm (Y)
    3. Decide the number of the Max-Iteration and init the current time
       Or decide the terminations. (N)
    4. Select Operator : 
        1. Create the FG(Father Group) to generate the new agents
        2. Select the SG(Surviver Group) to create the next generations
        P.S : Use the possibility to execute the Select Operator
    5. Generat Operator :
        1. Combination : Near 0.9
            The core for the Algorithm and the result
        2. Mutation    : Near 0.1
            The core operator for search the solution in the State Space
            Escape the Local Optimization
            
Personal Define:
    1. The agent who live more than `n` loop will die
    2. We can add the function to hold the best answer of each loop and copy it 
       to next iterations (We can set its living is inf to make it) ???
    3. Use the point to point at the best result in the history

Personal Log:
    1. The result is better than the PSO Algorithm, but it still truck in the local optimization, the solution ?
        1. One solution is try to add the beta (Mutation Possibility) to extend the search area in this TSP problem ---- I set the mutation possibility as 1, and it work well.
        2. Add the mutation swap number at the solution ---- actually not so good
        3. Possibility to swap more gene, we can set more gene swap operations ---- works well
'''

import numpy as np
import random
import time
import matplotlib.pyplot as plt

import sys
sys.path.append('..')

import dataset
from operator import attrgetter, itemgetter

# The point to the best agent in the history
point_solution = None
point_fittness = None

def fittness(agent, cities_map):
    # this function calculate the fittness for the agent
    s = 0
    for i in range(agent.dimension - 1):
        s += cities_map[agent.solution[i] - 1, agent.solution[i + 1] - 1]
    s += cities_map[agent.solution[-1] - 1, agent.solution[0] - 1]
    return s

class agent:
    def __init__(self, dimension, cities_map, max_living, solution = None):
        self.dimension = dimension
        if solution is None:
            self.solution = np.arange(1, dimension + 1)
            # Init the agent for the swarm
            np.random.shuffle(self.solution)
        else:
            self.solution = solution.copy()    # Must be the copy
        self.fittness = fittness(self, cities_map)
        self.living   = 0
        self.max_living = max_living
    def live_forever(self):
        # the Elitism
        self.max_living = np.inf
    
def check(sol1, sol2, dimension):
    # this function try to check and fix two solution and return the right ans
    # fix the sol1 and return sol1 !! fuck !! do not need to return 
    p = sol1.copy()
    first = -1 

    while len(set(p)) < dimension :
        # exist the error, need to be fixed !
        for i in range(dimension):
            if p.count(p[i]) > 1 :
                first = i
                break
        try:
            second = p.index(p[first], first + 1)
            # Find the anothor index of the [i]
            p[first] = sol2[second]
        except:
            # There may be two error 
            #   1. do not exist the number
            #   2. first + 1 is over the boundary
            # But they all mean that there is single number of [i]
            continue
    return p
    
def init_swarm(size, dimension, cities_map, max_living):
    # Init the swarm for the GA
    swarm = []
    for i in range(size):
        swarm.append(agent(dimension, cities_map, max_living))
    return swarm

def generate(agent_1, agent_2, alpha, beta, mutation_number):
    # This function use two father to generate the new agent 
    
    # ---- Combination alpha % ---- #
    
    if random.random() < alpha : 
        # Possibility to combination
        pause_sol_1 = list(agent_1.solution.copy())
        pause_sol_2 = list(agent_2.solution.copy())
        # Step 1 : randomly choose **two** point to combination
        f, l = sorted(random.sample(range(agent_1.dimension - 1), 2))
        # switch 
        pause_sol_1[f + 1: l + 1], pause_sol_2[f + 1: l + 1] = pause_sol_2[f + 1: l + 1], pause_sol_1[f + 1: l + 1]
        
        # check and fix the pause_sol_1 
        pause_sol_1 = check(pause_sol_1, pause_sol_2, agent_1.dimension)
        # check and fix the pause_sol_2
        pause_sol_2 = check(pause_sol_2, pause_sol_1, agent_1.dimension)
    else:
        # If do not combina, create two random solution to exploration the result
        pause_sol_1 = list(range(agent_1.dimension))
        random.shuffle(pause_sol_1)
        pause_sol_2 = list(range(agent_1.dimension))
        random.shuffle(pause_sol_2)
    
    # ---- Mutation beta % ---- #
    
    if random.random() < beta : 
        # Possibility to mutation, one time to swap one tuple of times
        # mutation number control the number of the swap operators
        for i in range(mutation_number):
            if random.random() < 1 / (i + 1):
                x1, y1 = random.sample(range(0, agent_1.dimension), 2)
                x2, y2 = random.sample(range(0, agent_1.dimension), 2)
                
                pause_sol_1[x1], pause_sol_1[y1] = pause_sol_1[y1], pause_sol_1[x1]
                pause_sol_2[x2], pause_sol_2[y2] = pause_sol_2[y2], pause_sol_2[x2]
    
    # return two children 
    return np.array(pause_sol_1), np.array(pause_sol_2)

def select_parent(swarm, size, ii):
    # this function try to choose `size` parents based on the fittness
    # return `size` father to create the children
    if size % 2 == 1 :
        if size >= len(swarm) : size -= 1
        else : size += 1
    swarm = sorted(swarm, key = attrgetter('fittness'))
    # count the sum of the fittness
    s_fittness = 0
    for i in  swarm:
        s_fittness += i.fittness
    # Possibility
    result = []
    exist = []
    c_number = 0
    index= 0
    while True:
        god = np.random.uniform()
        if god > (swarm[index].fittness / s_fittness) :
            if index in exist : 
                index += 1
                if index == len(swarm):
                    index = index % len(swarm)
                continue
            c_number += 1
            result.append(swarm[index])
            exist.append(index)

        if c_number > size : 
            break
            
        index += 1
        if index == len(swarm):
            index = index % len(swarm)
    return result

def select_children(father, size, swarm, s_size, alpha, beta, cities_map, max_living, mutation_number):
    # choose 2 * `size` children and combine with the swarm, choose `s_size` survivor
    # Create children
    children = []
    for i in range(size):
        f1, f2 = random.sample(father, 2)
        embryo1, embryo2 = generate(f1, f2, alpha, beta, mutation_number)
        
        child1 = agent(f1.dimension, cities_map, max_living, embryo1)
        child2 = agent(f2.dimension, cities_map, max_living, embryo2)
        children.append(child1)
        children.append(child2)
    swarm.extend(children)
    return select_parent(swarm, s_size, 5)
    
def die(swarm):
    # make sure that some agent will die
    survivor = []
    for agent in swarm:
        agent.living += 1
        if agent.living <= agent.max_living:
            survivor.append(agent)
    # return the survivor
    print('%d agents died at last loop' % (len(swarm) - len(survivor)))
    return survivor

def main(size, dimension, cities_map, loop_time, parent_size, children_size, alpha, beta, max_living, mutation_number):
    swarm = init_swarm(size, dimension, cities_map, max_living)
    for i in range(loop_time):
        father = select_parent(swarm, parent_size, i)
        # print('%d is selecting parent ... ' % i)
        swarm  = select_children(father, children_size, swarm, size, alpha, beta, cities_map, max_living, mutation_number)
        # print('%d is selecting children ... ' % i)

        swarm  = sorted(swarm, key = attrgetter('fittness'))    # some agents must die
        point_solution = swarm[0].solution    # point save the best agent in the history
        point_fittness = swarm[0].fittness
        # print("%d is finding the best agent ... %f" % (i, point_fittness))
        yield i, point_fittness, point_solution

if __name__ == "__main__":
    cities_map, dimension, city = dataset.create_map('berlin52.tsp')
    # ---- test for fittness ---- #
    # print(fittness(agent(dimension, cities_map, 5), cities_map))
    # ---- test for check ---- #
    # print(check([1, 2, 1, 4, 5, 3, 6], [2, 7, 7, 1, 3 ,5, 4], 7))
    # ---- test for init_swarm ---- #
    # swarm = init_swarm(1000, dimension, cities_map, 5)
    # print(swarm)
    # ---- test for generator ---- #
    # agent1 = agent(dimension, cities_map, 5)
    # agent2 = agent(dimension ,cities_map, 5)
    # print(generate(agent1, agent2, 0.9, 0.1))
    # ---- test for select_parent ---- #
    # father = select_parent(swarm, 200)
    # ---- test for select_children ---- #
    # print(len(select_children(father, 200, swarm, 300, 0.9, 0.1, cities_map, 5)))
    
    # 145.8 for cha34 question 
    t = main(100, dimension, cities_map, 1000, 20, 100, 0.9, 1, 5, 10)
    c = 0
    y = []
    for i, j ,k in t:
        best_path = list(map(lambda x : x - 1, k))
        y.append(j)
        # if c % 50 == 0:
        #     for i in range(-1, len(best_path) - 1):
        #         x = [city[best_path[i]][1], city[best_path[i + 1]][1]]
        #         y = [city[best_path[i]][2], city[best_path[i + 1]][2]]
        #         plt.plot(x, y)
        #     plt.show()
        c += 1
        print(i, j, k)
    x = list(range(1, 1000 + 1))
    plt.plot(x, y)
    plt.show()
