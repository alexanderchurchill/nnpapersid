import random,copy,os,pickle
from deap import base
from deap import creator
from deap import tools
from deap import benchmarks
import numpy as np

def ensure_dir(f):
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)

class KnapsackData(object):
    """docstring for Knapsack"""
    def __init__(self, name, knapsacks, items, values, capacities, constraints):
        super(KnapsackData, self).__init__()
        self.name = name
        self.knapsacks = knapsacks
        self.items = items
        self.values = values
        self.capacities = capacities
        self.constraints = constraints

class Genotypes(object):
    def __init__(self, min=True):
        super(Genotypes, self).__init__()
        self.genotype_array = []
        self.keys = {}
        self.min = min
        self.window = 100

    def add_genotypes(self,genotypes,uniques = False):
        if len(self.genotype_array) >= self.window:
            removing = self.genotype_array.pop()
            if uniques:
                hashed_g = str(self.genotype_array)
                for g in removing:
                    hashed_genotype = str(g)
                    if hashed_genotype not in hashed_g:
                        del self.keys[hashed_genotype]
        unique_genotypes = []
        for g in genotypes:
            if uniques:
                if str(g) not in self.keys:
                    unique_genotypes.append(g)
                    self.keys[str(g)] = 1
            else:
                unique_genotypes.append(g)
        self.genotype_array.insert(0,unique_genotypes)

    def flatten(self):
        return [item for sublist in self.genotype_array for item in sublist]

    def top_x_percent(self,x=0.2):
        s = self.flatten()
        s = sorted(s,key=lambda member:member.fitness.values)
        if self.min == False:
            s.reverse()
        self.top_x = s[0:int(len(s)*x)]
        return self.top_x

    def get_and_save_top_x(self,x=0.2,path="experiments",experiment=0,generation=0):
        top_x = self.top_x_percent(x)

    def top_x_genotypes_to_file(self,top_20,path="experiments",experiment=0,generation=0):
        np.savetxt("{0}_{1}/top_genotypes_{2}.dat".format(path,experiment,generation),top_20)
        np.savetxt("{0}_{1}/top_genotypes_fitnesses_{2}.dat".format(path,experiment,generation),[f.fitness.values[0] for f in top_20])

class GA(object):
    def fitness_function(self,individual):
        return sum(individual),

    def mutate(self,individual,indpb=0.05):
        return tools.mutFlipBit(individual,indpb)

    def __init__(self):
        super(GA, self).__init__()
        self.pop_size = 1000
        self.mut_rate = 0.2
        self.cross_rate = 0.9
        self.generations = 20
        self.tournament_size = 3
        self.N = 100
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        self.toolbox = base.Toolbox()
        # Attribute generator
        self.toolbox.register("attr_bool", random.randint, 0, 1)
        # Structure initializers
        self.toolbox.register("individual", tools.initRepeat, creator.Individual, 
            self.toolbox.attr_bool, 100)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        # Operator registering
        self.toolbox.register("evaluate", self.fitness_function)
        self.toolbox.register("mate", tools.cxTwoPoints)
        self.toolbox.register("mutate", self.mutate, indpb=0.05)
        self.toolbox.register("select", tools.selTournament, tournsize=self.tournament_size)

        self.population_snapshots = []
        self.genotypes_history = Genotypes(min=False)

    def run(self, path= "", experiment = 0):
        random.seed(random.uniform(0,1000))
        
        pop = self.toolbox.population(n=self.pop_size)
        CXPB, MUTPB, NGEN = self.cross_rate, self.mut_rate, self.generations
        
        print("Start of evolution")
        
        # Evaluate the entire population
        fitnesses = list(map(self.toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
        self.population_snapshots.append(copy.deepcopy(pop))
        self.genotypes_history.add_genotypes(pop)
        self.genotypes_history.get_and_save_top_x(0.2,"{0}experiment".format(path),experiment,0)
        #self.train_RBM()
        fitnesses_file = open("{0}experiment_{1}/fitnesses_temp.dat".format(path,experiment),"w")
        fitnesses_file.close()
        fitnesses_file = open("{0}experiment_{1}/fitnesses_temp.dat".format(path,experiment),"a")
        fitnesses_file.write(str([i.fitness.values[0] for i in pop]))
        fitnesses_file.write("\n")

        print("  Evaluated %i individuals" % len(pop))
        # Begin the evolution
        for g in range(NGEN):
            print("-- Generation %i --" % (g + 1))
            print "pop_size:",len(pop)
            elite = self.toolbox.clone(tools.selBest(pop, 1)[0])
            print "fitness elite:",elite.fitness.values
            # Select the next generation individuals
            offspring = self.toolbox.select(pop, len(pop) -1)
            # Clone the selected individuals
            offspring = list(map(self.toolbox.clone, offspring))

            # Apply crossover and mutation on the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < CXPB:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < MUTPB:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values
        
            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            
            print("  Evaluated %i individuals" % len(invalid_ind))
            
            # The population is entirely replaced by the offspring
            pop[:] = [elite] + offspring
            
            # Gather all the fitnesses in one list and print the stats
            fits = [ind.fitness.values[0] for ind in pop]
            self.statistics(fits)
            
            self.population_snapshots.append(copy.deepcopy(pop))
            self.genotypes_history.add_genotypes(pop)
            self.genotypes_history.get_and_save_top_x(0.2,"{0}experiment".format(path),experiment,g+1)
            fitnesses_file.write(str([i.fitness.values[0] for i in pop]))
            fitnesses_file.write("\n")
        print("-- End of (successful) evolution --")
        self.save_fitnesses(self.population_snapshots,"{0}experiment".format(path),experiment)
        return pop

    def statistics(self,fits):
        length = len(fits)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - mean**2)**0.5
        print("  Min %s" % min(fits))
        print("  Max %s" % max(fits))
        print("  Avg %s" % mean)
        print("  Std %s" % std)

    def get_fitnesses_from_population(self,all_populations):
        fitnesses = [[i.fitness.values[0] for i in p] for p in all_populations]
        return fitnesses

    def save_fitnesses(self,all_populations,path="",experiment=0):
        fitnesses = self.get_fitnesses_from_population(all_populations)
        avgs = np.mean(fitnesses,axis=1)
        maxs = np.max(fitnesses,axis=1)
        np.savetxt("{0}_{1}/mean_fitnesses.dat".format(path,experiment),avgs)
        np.savetxt("{0}_{1}/max_fitnesses.dat".format(path,experiment),maxs)

    def plot_mean_fitnesses(self,all_populations):
        fitnesses = self.get_fitnesses_from_population(all_populations)
        avgs = np.mean(fitnesses,axis=1)
        plt.plot(avgs)
        plt.show()

class MDimKnapsack(GA):
    def __init__(self,knapsack_file="weing8.pkl"):
        super(MDimKnapsack, self).__init__()
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        self.pop_size = 100
        self.mut_rate = 1.0
        self.cross_rate = 0.2
        self.generations = 2000

        self.knapsack = pickle.load(open(knapsack_file))
        self.knapsack.capacities = [[2000]]
        self.N = int(self.knapsack.items)
        self.toolbox = base.Toolbox()
        # Attribute generator
        self.toolbox.register("attr_bool", random.randint, 0, 1)
        # Structure initializers
        self.toolbox.register("individual", tools.initRepeat, creator.Individual, 
            self.toolbox.attr_bool, int(self.knapsack.items))
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        # Operator registering
        self.toolbox.register("evaluate", self.fitness_function, knapsack = self.knapsack)
        self.toolbox.register("mate", tools.cxTwoPoints)
        self.toolbox.register("mutate", self.mutate, indpb=float(1.0/self.N))
        self.toolbox.register("select", tools.selTournament, tournsize=self.tournament_size)

        self.genotypes_history = Genotypes(min=False)

    def fitness_function(self,individual,knapsack=None):
        weights = []
        for i,c in enumerate(knapsack.capacities):
            weights.append(np.sum(np.array(knapsack.constraints[i])*individual))
        over = 0
        for i,w in enumerate(weights):
            if w > knapsack.capacities[i]:
                over += (w - knapsack.capacities[i])
        if over > 0:
            return -over,
        else:
            return np.sum(np.array(knapsack.values)*individual),

class Sphere(GA):
    def __init__(self):
        super(Sphere, self).__init__()
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)
        self.mut_rate = 1.0
        self.cross_rate = 0.2
        self.pop_size = 200
        self.N = 50
        self.generations = 300000/self.pop_size
        self.toolbox = base.Toolbox()
        # Attribute generator
        self.toolbox.register("attr_float", random.uniform, -5, 5)
        self.toolbox.register("individual", tools.initRepeat, creator.Individual, self.toolbox.attr_float, self.N)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        # Operator registering
        self.toolbox.register("evaluate", benchmarks.sphere)
        self.toolbox.register("mate", tools.cxTwoPoints)
        self.toolbox.register("mutate", self.mutate, hi=1,lo=-1,mu=0,sigma=0.1,indpb=1.0/self.N)
        self.toolbox.register("select", tools.selTournament, tournsize=self.tournament_size)

        self.genotypes_history = Genotypes(min=True)

    def mutate(self,individual,hi=3,lo=-3,mu=0,sigma=0.1,indpb=0.05):
        for i in xrange(len(individual)):
            if random.random() < indpb:
                individual[i] += random.gauss(mu, sigma)
                if individual[i] > hi:
                    individual[i] = hi
                elif individual[i] < lo:
                    individual[i] = lo
        return individual,

class Rosenbrock(Sphere):
    """docstring for Rosenbrock"""
    def __init__(self):
        super(Rosenbrock, self).__init__()
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)
        self.mut_rate = 1.0
        self.cross_rate = 0.2
        self.pop_size = 500
        self.generations = 300000/self.pop_size
        self.N = 10

        self.toolbox = base.Toolbox()
        # Attribute generator
        self.toolbox.register("attr_float", random.uniform, -2.048, 2.048)
        self.toolbox.register("individual", tools.initRepeat, creator.Individual, self.toolbox.attr_float, self.N)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        # Operator registering
        self.toolbox.register("evaluate", benchmarks.rosenbrock)
        self.toolbox.register("mate", tools.cxTwoPoints)
        self.toolbox.register("mutate", self.mutate, hi=2.048,lo=-2.048,mu=0,sigma=0.01,indpb=1.0/self.N)
        self.toolbox.register("select", tools.selTournament, tournsize=self.tournament_size)

class Rastrigin(Sphere):
    """docstring for Rastrigin"""
    def __init__(self):
        super(Rastrigin, self).__init__()
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)
        self.mut_rate = 1.0
        self.cross_rate = 0.2
        self.pop_size = 500
        self.generations = 300000/self.pop_size
        self.N = 10

        self.toolbox = base.Toolbox()
        # Attribute generator
        self.toolbox.register("attr_float", random.uniform, -5.12, 5.12)
        self.toolbox.register("individual", tools.initRepeat, creator.Individual, self.toolbox.attr_float, self.N)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        # Operator registering
        self.toolbox.register("evaluate", benchmarks.rastrigin)
        self.toolbox.register("mate", tools.cxTwoPoints)
        self.toolbox.register("mutate", self.mutate, hi=5.12,lo=-5.12,mu=0,sigma=0.01,indpb=1.0/self.N)
        self.toolbox.register("select", tools.selTournament, tournsize=self.tournament_size)

class Experiment(object):
    def __init__(self, experiment_name,no_runs=5,start=0,end=None,test=None,ga=None):
        super(Experiment, self).__init__()
        self.experiment_name = experiment_name
        self.no_runs = no_runs
        self.start = start
        self.end = end
        if self.end == None:
            self.end = self.no_runs
        self.test = test
        if self.test == None:
            self.test = "sphere"
        ensure_dir("results/{0}".format(self.experiment_name))
        self.ga = None
        self.set_experiment(test,ga)

    def set_experiment(self,test=None,ga=None):
        if test == None:
            self.ga = ga
        else:
            if test == "sphere":
                self.ga = Sphere()
            elif test == "rosenbrock":
                self.ga = Rosenbrock()
            elif test == "rastrigin":
                self.ga = Rastrigin()
            elif test == "one_max":
                self.ga = GA()
            elif test == "knapsack_easy":
                self.ga = MDimKnapsack("weing1.pkl")
            elif test == "knapsack_hard":
                self.ga = MDimKnapsack("weing8.pkl")
            elif test == "knapsack_400":
                self.ga = MDimKnapsack("knapsack_400.pkl")
            elif test == "knapsack_500":
                self.ga = MDimKnapsack("knapsack_500.pkl")

    def run(self,test,ga):
        for i in range(self.start,self.end):
            self.ga = None
            self.set_experiment(test,ga)
            path = "results/{0}/".format(self.experiment_name)
            ensure_dir("{0}experiment_{1}/".format(path,i))
            self.ga.run(path = path,experiment=i)

if __name__ == "__main__":
    name = "sphere"
    test = "sphere"
    e = Experiment(name,no_runs=10,start=0,end=10,test=test)
    e.run(test=test,ga=None)
