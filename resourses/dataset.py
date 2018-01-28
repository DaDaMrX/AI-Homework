#!/usr/bin/python3

'''
    This function try to provide a interface function for the TSP Quesion
    1. ECU_2D     : The ECU distance calculate
    2. create_map : The function return the cities map
'''

import numpy as np

# the function for getting data and create the cities map
def ECU_2D(x1, y1, x2, y2):
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        
def create_map(filename):
    cities = []
    with open(filename, 'r') as f:
        dimension = 0
        for index, line in enumerate(f.readlines()):
            if 'EOF' in line : break
            if index == 3 : 
                dimension = int(line[11:-1].strip())
                print('Problem\'s dimension is %d' % dimension)
            if index >= 6 :
                content = tuple(map(float, line.split()))
                cities.append(content)
        print('%d cities / %d dimension' % (len(cities), dimension))
    cities_number = len(cities)
    cities_map = np.zeros([cities_number, cities_number])
    # create the distance map for the cities
    # this sentence may be very slow
    for i in range(cities_number):
        for j in range(cities_number):
            if j == i : continue
            else:
                cities_map[i, j] = ECU_2D(cities[i][1], cities[i][2], \
                        cities[j][1] ,cities[j][2])
    print('The distance map has been created !')
    # return the distance ndarray
    return cities_map, dimension, cities

if __name__ == "__main__":
    print(create_map('../DATA/berlin52.tsp'))
