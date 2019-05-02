import random
import numpy as np
import matplotlib.pyplot as plt
import math
########################################################################################################################
# Setting GA
########################################################################################################################
class SetupGA():
    def __init__(self, cromoSize, poulationSize, generationNumber, mutationRate):
        self.CHROMOSOME = cromoSize
        self.POPULATION_SIZE = poulationSize
        self.GENERATIONS_NUMBER = generationNumber
        self.MUTATION_RATE = mutationRate


CHROMOSOME = 0
POPULATION_SIZE = 0
GENERATIONS_NUMBER = 0
MUTATION_RATE = 0

########################################################################################################################
# Individual
########################################################################################################################
class Individual():
    def __init__(self, pilotsSequence = [], geracao = 0):
        self.fitnessNote = 0
        self.chromosome = []
        self.geracao = geracao

        for i in range(len(pilotsSequence)):
            # print("IDV --- %s -> %s" % (i, pilotsSequence[i]))
            # for j in range(len(pilotsSequence[i])):
            self.chromosome.append(pilotsSequence[i])
        # print("CROMO -> %s " % ( self.chromosome))

    # def fnFitness(self, phi, beta, sigma):
    def fnFitness(self, beta, sigma):
        print('calcular fitness')

        # l = self.chromosome[0]
        # k = self.chromosome[1]
        # sequence = self.chromosome[2]
        #
        # self.fitnessNote = (l + k)

        phi = self.chromosome

        f = np.zeros((len(phi), len(phi)))
        for ell in range(0, len(phi)):
            for k in range(0, len(phi)):
                f[k, ell] = beta[k, ell, ell]
                deno = 0
                for j in range(0, len(phi)):
                    for kline in range(0, len(phi)):
                        if j != ell:
                            # deno += np.inner(phi[k, :, ell], phi[kline, :, j])*beta[kline, j, ell]
                            deno += np.inner(phi[2], phi[2])*beta[kline, j, ell]
                deno += sigma
                f[k, ell] /= deno
        self.fitnessNote = np.sum(f)
        print("fitnessNote = ", self.fitnessNote)




    def crossover(self, otherIndvidual):
        children1 = self.chromosome[0:2]+otherIndvidual.chromosome[2::]
        children2 = otherIndvidual.chromosome[0:2] + self.chromosome[2::]

        childrens = [
            Individual(self.chromosome),
            Individual(self.chromosome)
        ]

        childrens[0].chromosome = children1
        childrens[1].chromosome= children2
        return childrens

    def mutation(self, rateMutation):
        randomSequence = random.randint(0,1)
        if randomSequence < rateMutation:
            self.chromosome[2] = random.randint(0,9)
        return self


########################################################################################################################
#Standard deviation
########################################################################################################################
def standardDeviation(fitnessList):
    media = sum(fitnessList) / (len(fitnessList))
    quadrados = []
    for i in range(len(fitnessList)):
        quadrados.append((fitnessList[i]-media) ** 2)
    somaQuadrados = sum(quadrados)
    varianca = somaQuadrados / ((len(fitnessList)) - 1)
    return math.sqrt(varianca)

########################################################################################################################

########################################################################################################################
# GA
########################################################################################################################
class GeneticAlgorithm():
    def __init__(self):
        self.population = []
        self.bestSolution = 0
        self.listSolution = [] #graphic

    def inicializePopulation(self, phi):
        for i in range(len(phi)):
            self.population.append([])
            for j in range(len(phi[i])):
                # print("TESTE ", phi[i])
                self.population[i].append(Individual(phi[i][j]))
                # self.bestSolution = self.population[0]
            print("Icializando população -> ", i)

    def printPopulation(self):
        for h in range(len(self.population)):
            for i in range(len(self.population[h])):
                print("PilotSequence Cell -> %s user: %s - chromosome: %s | fitnessNote: %s" % (
                    h, i, self.population[h][i].chromosome, self.population[h][i].fitnessNote))

    def sortPopulation(self):
        self.population = sorted(self.population, key=lambda population: population.fitnessNote, reverse=False)

    def sumFitnessRate(self, cell):
        total = 0
        # for population in range(len(self.population)):
        #     for individual in range(len(self.population[population])):
        # total += self.population[population][individual].fitnessNote
        for individual in cell:
            total += individual.fitnessNote
        return total

    def rouletteVitiate(self, sumFitness, cell):
        father = -1
        randomSort = random.randint(2,9)
        sumLoopsRoulette = 0
        i = 0
        while i < len(cell) and randomSort > sumLoopsRoulette:
            # print("cellFitnote",cell[i].fitnessNote)
            sumLoopsRoulette += cell[i].fitnessNote
            father += 1
            i += 1
        return father

    ########################################################################################################################
    # RUN GA
    ########################################################################################################################
    def runGA(self, pilotSequenceHipermatrix, generationsNumber, rateMutation, beta, sigma):
        self.inicializePopulation(pilotSequenceHipermatrix)
        for i in range(len(self.population)):
            for j in range(len(self.population[i])):
                self.population[i][j].fnFitness(beta, sigma)
        # self.printPopulation()
        # self.sortPopulation()
        self.printPopulation()

        for geracao in range(generationsNumber):
            newPopulation = []

            for cell in self.population:
                newCell = []
                sumEvaluations = self.sumFitnessRate(cell)
                # print("SUM Evaluation ", sumEvaluations)
                for individualGenerated in range(0, len(cell), 2):
                    father1 = self.rouletteVitiate(sumEvaluations, cell)
                    father2 = self.rouletteVitiate(sumEvaluations, cell)
                    # print("Father 1: %s | Father 2: %s " % (father1, father2))

                    #crossover
                    childrens = cell[father1].crossover(cell[father2])

                    #newPopulation
                    newCell.append(childrens[0].mutation(rateMutation))
                    newCell.append(childrens[1].mutation(rateMutation))
                    # print("new Cell ", newCell)
                newPopulation.append(newCell)


            self.population = list(newPopulation)
            for i in range(len(self.population)):
                fitnessList = [] #graphic

                for j in range(len(self.population[i])):
                    self.population[i][j].fnFitness(beta, sigma)

                    fitnessList.append(self.population[i][j].fitnessNote)

                    if self.bestSolution < self.population[i][j].fitnessNote:
                        self.bestSolution = self.population[i][j].fitnessNote

            print("list ", self.listSolution)
            self.listSolution.append(max(fitnessList)) #graphic


        print("------------------------------")
        print("desvioPadrao %s " % (standardDeviation(fitnessList)))
        print("\nmelhor solucao ", self.bestSolution)
        print("------------------------------")

        plt.figure()
        plt.plot(self.listSolution)


