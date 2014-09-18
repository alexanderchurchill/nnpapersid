#    This file is part of DEAP.
#
#    DEAP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    DEAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with DEAP. If not, see <http://www.gnu.org/licenses/>.

import array
import random
import copy
from deap import base
from deap import creator
from deap import benchmarks
from deap import tools
import numpy as np
IND_SIZE = 50

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMin)

def update(ind, mu, std):
    for i,m in enumerate(mu):
        ind[i] = ind[i] + random.gauss(0,std)
    return ind

 
toolbox = base.Toolbox()                  
toolbox.register("update", update)
toolbox.register("evaluate", benchmarks.sphere)

def main(experiment_name,iteration=0):
    """
    Implements a (1+1)-ES with the one fith rule
    """
    interval = (-5.12,5.12)
    mu = [0 for _ in range(IND_SIZE)]
    sigma = 0.1
    alpha = 1.5

    best = creator.Individual([random.uniform(interval[0],interval[1]) for i in range(IND_SIZE)])
    best.fitness.values = toolbox.evaluate(best)
    child = creator.Individual((0.0,)*IND_SIZE)

    NGEN = 300000
    best_fitnesses = []
    for g in range(NGEN):
        child = copy.deepcopy(best)
        child = update(child,best,sigma)
        child.fitness.values = toolbox.evaluate(child)
        random_p = random.random() < 0.00
        if child.fitness >= best.fitness or random_p:
            if random_p == False:
                sigma = sigma * alpha
            best, child = child, best
        else:
            sigma = sigma * alpha**(-0.25)
        best_fitnesses.append(best.fitness.values[0])
    np.savetxt("{0}_{1}.dat".format(experiment_name,iteration),best_fitnesses)
    print best_fitnesses[-1]
    return best
   
if __name__ == "__main__":
    experiment_name = "sphere"
    for i in range(0,10):
        print i
        main(experiment_name,i)
