import random,pickle,copy,bisect,os,sys
import numpy as np
import theano
from denoising_autoencoder import dA
from custom_dataset import SequenceDataset
from optimizers import sgd_optimizer
import distance
from hiff import HIFF
from scipy.spatial.distance import pdist,cdist

class AESolver(object):
    """
    The Denoising Autoencoder Genetic Algorithm
    """
    def __init__(self,fitness_f):
        super(AESolver, self).__init__()
        self.FITNESS_F = fitness_f
        if self.FITNESS_F == "hiff":
            self.HIFF = HIFF(NUMGENES=128,K=2,P=7)
            self.fitness = self.hiff_fitness
        elif self.FITNESS_F == "knapsack":
            self.fitness = self.knapsack_fitness
        elif self.FITNESS_F == "max_ones":
            self.fitness = self.max_ones_fitness
        elif self.FITNESS_F == "left_ones":
            self.fitness = self.left_ones
        elif self.FITNESS_F == "churchill":
            self.fitness = self.churchills_road
            self.optimum = 33

    def generate_random_string(self,l=20):
        return [random.choice([0,1]) for i in range(l)]

    def churchills_road(self,input,k=4,l=4):
        fitness = 0
        for partitions in range(0,l):
            first_part = sum(input[partitions*k*2:partitions*k*2+k])
            second_part = sum(input[(partitions*k*2)+k:(partitions*k*2)+k*2])
            if first_part == k and second_part == 0:
                fitness += 8
            if first_part == 0 and second_part == k:
                fitness += 8
        if sum(input[0:k]) == k and sum(input[len(input)-k:]) == k:
            fitness += 1
        if sum(input[0:k]) == 0 and sum(input[len(input)-k:]) == 0:
            fitness += 1
        return fitness

    def knapsack_fitness(self,string):
        knapsack = self.knapsack
        weights = []
        for i,c in enumerate(knapsack.capacities):
            weights.append(np.sum(np.array(knapsack.constraints[i])*string))
        over = 0
        for i,w in enumerate(weights):
            if w > knapsack.capacities[i]:
                over += (w - knapsack.capacities[i])
        if over > 0:
            return -over
        else:
            _fitness = np.sum(np.array(knapsack.values)*string)
            return _fitness

    def hiff_fitness(self,string):
        fitness = self.HIFF.H(string)
        return fitness

    def max_ones_fitness(self,string):
        fitness = np.sum(string^self.mask)
        if cache:
            self.cache_fitness(fitness)
        return fitness

    def left_ones_fitness(self,_string):
        string =_string^self.mask
        fitness = sum(string[0:len(string)/2]) - sum(string[len(string)/2:])
        if cache:
            self.cache_fitness(fitness)
        return fitness

    def tournament_selection_replacement(self,
                                         population,
                                         fitnesses=None,
                                         pop_size=None):
        if pop_size == None:
            pop_size = len(population)
        if fitnesses == None:
            fitnesses = self.fitness_many(population)
        new_population = []
        while len(new_population) < pop_size:
            child_1 = int(np.random.random() * pop_size)
            child_2 = int(np.random.random() * pop_size)
            if fitnesses[child_1] > fitnesses[child_2]:
                new_population.append(copy.deepcopy(population[child_1]))
            else:
                new_population.append(copy.deepcopy(population[child_2]))
        return new_population

    def get_good_strings(self,strings,lim=20,unique=False,fitnesses=None):
        if fitnesses == None:
            fitnesses = [self.fitness(s) for s in strings]
        sorted_fitnesses = sorted(range(len(fitnesses)),
                                  key=lambda k: fitnesses[k])
        sorted_fitnesses.reverse()
        if unique == False:
            return ([strings[i] for i in sorted_fitnesses[0:lim]],
                    [fitnesses[k] for k in sorted_fitnesses[0:lim]])
        else:
            uniques = {}
            good_pop = []
            good_pop_fitnesses = []
            index = 0
            while len(good_pop) < lim and index < len(sorted_fitnesses):
                key = str(strings[sorted_fitnesses[index]])
                if key not in uniques:
                    uniques[key] = 0
                    good_pop.append(strings[sorted_fitnesses[index]])
                    good_pop_fitnesses.append(
                        fitnesses[sorted_fitnesses[index]]
                        )
                index += 1
            if len(good_pop) == lim:
                return [good_pop,good_pop_fitnesses]
            else:
                while len(good_pop) < lim:
                    good_pop.append(self.generate_random_string(
                                        l=len(strings[0]))
                                    )
                    good_pop_fitnesses.append(self.fitness(good_pop[-1]))
                return [good_pop,good_pop_fitnesses]

    def RTR(self,
            population,
            sampled_population,
            population_fitnesses,
            sample_fitnesses,
            w=None):
        if w == None:
            w = len(population)/20
        _population = np.array(population)
        for ind_i,individual in enumerate(sampled_population):
            indexes = np.random.choice(len(_population), w, replace=False)
            distances = cdist(_population[indexes],[individual],"hamming")
            replacement = indexes[np.argmin(distances.flatten())]
            if population_fitnesses[replacement] < sample_fitnesses[ind_i]:
                _population[replacement] = individual
                population_fitnesses[replacement] = sample_fitnesses[ind_i]
        return _population

    def fitness_many(self,strings):
        return [self.fitness(s) for s in strings]

    def train_dA(self,
                 data,
                 corruption_level=0.2,
                 num_epochs=200,
                 lr=0.1,
                 output_folder="aemodels",
                 iteration=0):
        train_data = data
        train_set = SequenceDataset(train_data,batch_size=20,number_batches=None)
        sgd_optimizer(self.dA.params,[self.dA.input],self.dA.cost,train_set,
                      lr=lr,num_epochs=num_epochs,save=True,
                      output_folder=output_folder,iteration=iteration)

    def build_sample_dA(self):  
        self.sample_dA = theano.function([self.dA.input],self.dA.sample)

    def iterative_algorithm(
        self,
        name,
        pop_size=100,
        genome_length=20,
        lim_percentage=20,
        corruption_level=0.2,
        num_epochs=50,
        lr = 0.1,
        max_evaluations=200000,
        unique_training=False,
        hiddens=40,
        rtr = True
        ):
        self.mask = np.random.binomial(1,0.5,genome_length)
        trials = max_evaluations/pop_size
        population_limit = int(pop_size*(lim_percentage/100.0))
        self.dA = dA(n_visible=genome_length,n_hidden=hiddens)
        self.dA.build_dA(corruption_level)
        self.build_sample_dA()
        new_population = np.random.binomial(1,0.5,(pop_size,genome_length))
        self.population_fitnesses = self.fitness_many(new_population)
        for iteration in range(0,trials):
            print "iteration:",iteration
            population = new_population
            self.population = new_population
            rw = self.tournament_selection_replacement(population)
            good_strings,good_strings_fitnesses=self.get_good_strings(
                                          population,
                                          population_limit,
                                          unique=unique_training,
                                          fitnesses=self.population_fitnesses
                                        )
            print "training A/E"
            training_data = np.array(good_strings)
            self.train_dA(training_data,
                          num_epochs=num_epochs,
                          lr=lr)
            print "sampling..."
            sampled_population = np.array(self.sample_dA(rw),"b")
            self.sample_fitnesses = self.fitness_many(sampled_population)
            if rtr:
                new_population = self.RTR(
                              population,
                              sampled_population,
                              population_fitnesses=self.population_fitnesses,
                              sample_fitnesses=self.sample_fitnesses,
                              w=pop_size/20
                              )
            else:
                new_population = sampled_population
                new_population[0:1] = good_strings[0:1]
                self.population_fitnesses = self.sample_fitnesses
                self.population_fitnesses[0:1] = good_strings_fitnesses[0:1]
            print "{0},{1},{2}\n".format(np.mean(self.population_fitnesses),
                                         np.min(self.population_fitnesses),
                                         np.max(self.population_fitnesses))
            print "best from previous:",(
              self.fitness(new_population[np.argmax(self.population_fitnesses)])
                )
            if np.max(self.population_fitnesses) == self.optimum:
                pickle.dump({"pop":self.population,"fitnesses":self.population_fitnesses,"iteration":iteration},open("final_shit_ae.pkl","w"))
                break
        return new_population

if __name__ == '__main__':
    ae = AESolver("churchill")
    ae.iterative_algorithm(
        "churchill",
        pop_size=400,
        genome_length=32,
        lim_percentage=20,
        corruption_level=0.05,
        num_epochs=25,
        lr = 0.1,
        max_evaluations=20000,
        unique_training=True,
        hiddens=32,
        rtr = False
        )
