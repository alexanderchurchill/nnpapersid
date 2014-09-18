import random,pickle,copy,bisect,os,sys,pdb
import numpy as np
import theano
from denoising_autoencoder import dA
from custom_dataset import SequenceDataset
from optimizers import sgd_optimizer
import distance
from hiff import HIFF
from scipy.spatial.distance import pdist,cdist
from nade import NADE
from ga import KnapsackData
def ensure_dir(f):
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)

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
            self.knapsack = pickle.load(open("weing8.pkl"))
            self.fitness = self.knapsack_fitness
        elif self.FITNESS_F == "max_ones":
            self.fitness = self.max_ones_fitness
        elif self.FITNESS_F == "left_ones":
            self.fitness = self.left_ones
        elif self.FITNESS_F == "royal_road":
            self.fitness = self.royal_road
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

    def royal_road(self,string, order=8):
        """Royal Road Function R1 as presented by Melanie Mitchell in : 
        "An introduction to Genetic Algorithms".
        """
        individual = string^self.mask
        nelem = len(individual) / order
        max_value = int(2**order - 1)
        total = 0
        for i in xrange(nelem):
            value = int("".join(map(str, individual[i*order:i*order+order])), 2)
            total += int(order) * int(value/max_value)
        return total

    # def tournament_selection_replacement(self,
    #                                      population,
    #                                      fitnesses=None,
    #                                      pop_size=None):
    #     if pop_size == None:
    #         pop_size = len(population)
    #     if fitnesses == None:
    #         fitnesses = self.fitness_many(population)
    #     new_population = []
    #     while len(new_population) < pop_size:
    #         child_1 = int(np.random.random() * pop_size)
    #         child_2 = int(np.random.random() * pop_size)
    #         if fitnesses[child_1] > fitnesses[child_2]:
    #             new_population.append(copy.deepcopy(population[child_1]))
    #         else:
    #             new_population.append(copy.deepcopy(population[child_2]))
    #     return new_population

    def tournament_selection_replacement(self,
                                         population,
                                         fitnesses=None,
                                         pop_size=None,
                                         tournament_size=2):
        if pop_size == None:
            pop_size = len(population)
        if fitnesses == None:
            fitnesses = self.fitness_many(population)
        new_population = []
        while len(new_population) < pop_size:
            contenders=np.random.randint(0,len(population),tournament_size)
            # print "contenders:",contenders
            t_fitnesses = [fitnesses[c] for c in contenders]
            # print "fitnesses:",t_fitnesses
            # print "best_fitness:",np.argmax(t_fitnesses)
            # print "winner:",contenders[np.argmax(t_fitnesses)]
            winner = copy.deepcopy(population[contenders[np.argmax(t_fitnesses)]])
            new_population.append(winner)
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
                 output_folder="",
                 iteration=0):
        train_data = data
        train_set = SequenceDataset(train_data,batch_size=20,number_batches=None)
        sgd_optimizer(self.dA.params,[self.dA.input],self.dA.cost,train_set,
                      lr=lr,num_epochs=num_epochs,save=False,
                      output_folder=output_folder,iteration=iteration)

    def train_NADE(self,
                 data,
                 num_epochs=200,
                 lr=0.1,
                 output_folder="",
                 iteration=0):
        train_data = data
        train_set = SequenceDataset(train_data,batch_size=20,number_batches=None)
        sgd_optimizer(self.NADE.params,[self.NADE.v],self.NADE.cost,train_set,
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
        hiddens=300,
        rtr = True,
        w=10
        ):
        results_path = "results/autoencoder/{0}/".format(name)
        ensure_dir(results_path)
        fitfile = open("{0}fitnesses.dat".format(results_path),"w")
        self.mask = np.random.binomial(1,0.5,genome_length)
        trials = max_evaluations/pop_size
        population_limit = int(pop_size*(lim_percentage/100.0))
        # self.dA = dA(n_visible=genome_length,n_hidden=hiddens)
        # self.dA.build_dA(corruption_level)
        # self.build_sample_dA()
        self.NADE = NADE(n_visible=genome_length,n_hidden=hiddens)
        # self.NADE.build_NADE()
        new_population = np.random.binomial(1,0.5,(pop_size,genome_length))
        self.population_fitnesses = self.fitness_many(new_population)
        fitfile.write("{0},{1},{2},{3}\n".format(np.mean(self.population_fitnesses),np.min(self.population_fitnesses),np.max(self.population_fitnesses),np.std(self.population_fitnesses)))
        for iteration in range(0,trials):
            print "iteration:",iteration
            population = new_population
            self.population = new_population
            rw = self.tournament_selection_replacement(population,
                                                       fitnesses=self.population_fitnesses,
                                                       pop_size=population_limit,
                                                       tournament_size=4)
            if not rtr:
                good_strings,good_strings_fitnesses=self.get_good_strings(
                                              population,
                                              population_limit,
                                              unique=unique_training,
                                              fitnesses=self.population_fitnesses
                                            )
                training_data = np.array(good_strings)
            else:
                training_data = np.array(rw)
            print "training A/E"
            self.train_NADE(training_data,
                          num_epochs=num_epochs,
                          lr=lr)
            print "sampling..."
            # sampled_population = [np.array(self.NADE.sample(),"b") for i in range(len(self.population))]
            sampled_population = np.array(self.NADE.sample_multiple(n=len(new_population)),"b")
            # pdb.set_trace()
            self.sample_fitnesses = self.fitness_many(sampled_population)
            if rtr:
                new_population = self.RTR(
                              population,
                              sampled_population,
                              population_fitnesses=self.population_fitnesses,
                              sample_fitnesses=self.sample_fitnesses,
                              w=w
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
            fitfile.write("{0},{1},{2},{3}\n".format(np.mean(self.population_fitnesses),np.min(self.population_fitnesses),np.max(self.population_fitnesses),np.std(self.population_fitnesses)))
            fitfile.flush()
            if np.max(self.population_fitnesses) == self.optimum:
                pickle.dump({"pop":self.population,"fitnesses":self.population_fitnesses,"iteration":iteration},open("final_shit.pkl","w"))
                break
        fitfile.close()
        return new_population

if __name__ == '__main__':
    ae = AESolver("churchill")
    # ae.iterative_algorithm(
    #     "hiff-128",
    #     pop_size=2000,
    #     genome_length=128,
    #     lim_percentage=20,
    #     corruption_level=0.05,
    #     num_epochs=50,
    #     lr = 0.05,
    #     max_evaluations=300000,
    #     unique_training=True,
    #     hiddens=128,
    #     rtr = True,
    #     w=10
    #     )

    args = sys.argv
    pop_size = int(args[1])
    lim_percentage = int(args[2])
    num_epochs = int(args[3])
    lr = float(args[4])
    hiddens = int(args[5])
    rtr = int(args[6])
    if rtr == 1:
        rtr = True
    else:
        rtr = False
    w = int(args[7])
    trial = int(args[8])
    name = "royal_road-{0}".format("-".join([str(s) for s in [pop_size,
                                                            lim_percentage,
                                                            num_epochs,
                                                            lr,
                                                            hiddens,
                                                            rtr,
                                                            w,
                                                            trial]
                                                            ]))
    ae.iterative_algorithm(
        name,
        pop_size=pop_size,
        genome_length=32,
        lim_percentage=lim_percentage,
        corruption_level=0.05,
        num_epochs=num_epochs,
        lr = lr,
        max_evaluations=50000,
        unique_training=False,
        hiddens=hiddens,
        rtr = rtr,
        w=w
        )

