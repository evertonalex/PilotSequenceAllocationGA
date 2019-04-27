########################################################################################################################
# Setting GA
########################################################################################################################
class SetupGA():
    def __init__(self, cromoSize, poulationSize, generationNumber, mutationRate):
        self.CHROMOSOME = cromoSize
        self.POPULATION_SIZE = poulationSize
        self.GENERATIONS_NUMBER = generationNumber
        self.MUTATION_RATE = mutationRate

########################################################################################################################
# Individuo
########################################################################################################################
class Individual():
    def __init__(self, pilotsSequence = [], geracao = 0):
        self.fitnessNote = 0
        self.chromosome = []
        self.geracao = geracao

        for i in range(len(pilotsSequence)):
            # for j in range(len(pilotsSequence[i])):
            self.chromosome.append(pilotsSequence[i])
        # print("CROMO ", pilotsSequence)

    def fnFitness(self):
        print('calcular fitness')

########################################################################################################################
# GA
########################################################################################################################
class GeneticAlgorithm():
    def __init__(self):
        self.population = []
        self.bestSolution = 0

    def inicializePopulation(self, pilotSequences):
        for i in range(len(pilotSequences)):
            self.population.append([])
            for j in range(len(pilotSequences[i])):
                # print("TESTE ", pilotSequences[i])
                self.population[i].append(Individual(pilotSequences[i][j]))
                # self.bestSolution = self.population[0]

    def printPopulation(self):
        for h in range(len(self.population)):
            for i in range(len(self.population[h])):
                print("PilotSequence Cell -> %s user: %s - chromosome: %s" % (h, i, self.population[h][i].chromosome))