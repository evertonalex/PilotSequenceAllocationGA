import random
import numpy as np
import matplotlib.pyplot as plt
import math
from array import array

CHROMOSOME = 0
POPULATION_SIZE = 0
GENERATIONS_NUMBER = 0
MUTATION_RATE = 0
CUTPOINT = 0

########################################################################################################################
# Individual
########################################################################################################################
class Individual():
    def __init__(self, phi = [], geracao = 0):
        self.fitnessNote = 0
        self.chromosome = phi
        self.geracao = geracao

    def crossover(self, otherIndvidual, cutPoint):

        sizePhi = len(self.chromosome)

        children1 = []
        for i in range(sizePhi):
            children1.append([])
            if i < cutPoint:
                for j in range(len(self.chromosome[i])):
                    children1[i].append(otherIndvidual.chromosome[i][j])
            else:
                for j in range(len(self.chromosome[i])):
                    children1[i].append(self.chromosome[i][j])
        children1 = np.asarray(children1)

        children2 = []
        for i in range(sizePhi):
            children2.append([])
            if i < cutPoint:
                for j in range(len(self.chromosome[i])):
                    children2[i].append(self.chromosome[i][j])
            else:
                for j in range(len(self.chromosome[i])):
                    children2[i].append(otherIndvidual.chromosome[i][j])
        children2 = np.asarray(children2)


        childrens = [
            Individual(self.chromosome),
            Individual(self.chromosome)
        ]

        childrens[0].chromosome = children1
        childrens[1].chromosome= children2

        # print("children 1 ", children1)
        # print("children 2 ", children2)

        return childrens

    def mutation(self, rateMutation):
        randomSequence = random.randint(0,1)
        if randomSequence < rateMutation:

            for ell in range(0, len(self.chromosome[0])):
                q = list(range(len(self.chromosome[0])))
                np.random.shuffle(q)
                # print("tsetes --->", q)
                # print(self.chromosome[2])
                if ell == 3 or ell == 4:
                    self.chromosome[range(0, len(self.chromosome[0])), q, ell] = 1
                    # print(self.chromosome)
        return self


########################################################################################################################
#Standard deviation
########################################################################################################################
def standardDeviation(fitnessList):
    media = sum(fitnessList) / (len(fitnessList))
    print("Media -> ", media)
    quadrados = []
    for i in range(len(fitnessList)):
        quadrados.append((fitnessList[i]-media) ** 2)
    somaQuadrados = sum(quadrados)
    varianca = somaQuadrados / ((len(fitnessList)) - 1)
    return math.sqrt(varianca)

########################################################################################################################
#Function fitness
########################################################################################################################
def fnFitness(phi, beta, sigma):

    f = np.zeros((len(phi), len(phi[0][0])))
    for ell in range(0, len(phi[0][0])):
        for k in range(0, len(phi)):
                f[k, ell] = beta[k, ell, ell]
                deno = 0
                for j in range(1, len(phi[0][0])):
                    for kline in range(0, len(phi)):
                        if j != ell:
                            # if k > ell:
                                deno += np.inner(phi[k, :, ell], phi[kline, :, j]) * beta[kline, j, ell]
                deno += sigma
                f[k, ell] /= deno
        # self.fitnessNote = np.sum(f)
        # print("fitnessNote = ", abs(np.sum(f)))
        return abs(np.sum(f))

########################################################################################################################
# First Generation
########################################################################################################################
def firstGeneration(phi):
    for ell in range(0, len(phi[0][0])):
        q = list(range(0, len(phi[0])))
        np.random.shuffle(q)
        phi[range(0, len(phi[0])), q, ell] = 1
    return phi


########################################################################################################################
# GA
########################################################################################################################
class GeneticAlgorithm():
    def __init__(self):
        self.population = []
        self.bestSolution = 0
        self.listSolution = [] #graphic

    def inicializePopulation(self, populationSize, K, Tp, L, beta):
        for p in range(populationSize):
            self.population.append([])
            for i in range(K):
                phi = np.zeros((len(beta), len(beta), len(beta[0]), int(populationSize)))
                self.population[p].append(Individual(firstGeneration(phi)))
        # print("population -> ", self.population)
        # print("population -> ", self.population)

    def printPopulation(self):
        for i in range(len(self.population)):
            for h in range(len(self.population[i])):
                # print("popualtion: %s user: %s - chromosome: %s | fitnessNote: %s" % (
                #  i, h, self.population[i], self.population[i][h].fitnessNote))
                print("popualtion: %s user: %s - | fitnessNote: %s" % (
                 i, h, self.population[i][h].fitnessNote))

    def sortPopulation(self):
        self.population = sorted(self.population, key=lambda population: population.fitnessNote, reverse=False)

    def sumFitnessRate(self, population):
        total = 0
        for pop in population:
            total += pop.fitnessNote
        return total

    def rouletteVitiate(self, sumFitness, populacaoAtual):
        father = -1
        randomSort = np.random.uniform(0,sumFitness)
        sumLoopsRoulette = 0
        i = 0
        while i < len(self.population[0]) and sumLoopsRoulette < randomSort:
            sumLoopsRoulette += populacaoAtual[i].fitnessNote
            father += 1
            i += 1
        return father

########################################################################################################################
# RUN GA
########################################################################################################################
    def runGA(self, generationsNumber, rateMutation, beta, sigma, cutPoint, populationSize, K, Tp, L):
        self.inicializePopulation(populationSize, K, Tp, L, beta)

        pop = np.zeros((len(beta), len(beta), len(beta[0]), int(populationSize)))

        for i in range(len(self.population)):

            # pop[:, :, :, i] = firstGeneration(pop[:, :, :, i])
            for j in range(len(self.population[i])):
                self.population[i][j].fitnessNote = fnFitness(self.population[i][j].chromosome[:, :, :, i], beta, sigma)

        self.printPopulation()

        for geracao in range(generationsNumber):
            newPopulation = []
            print("Progress: %s%% | generation: %s" % (((geracao*100)/generationsNumber), geracao))

            for pop in range(len(self.population)):
                newIndividual = []
                sumEvaluations = self.sumFitnessRate(self.population[pop])
                # print("SUM Evaluation ", sumEvaluations)
                for individualGenerated in range(0, len(self.population), 2):
                    father1 = self.rouletteVitiate(sumEvaluations, self.population[pop])
                    father2 = self.rouletteVitiate(sumEvaluations, self.population[pop])
                    # print("Father 1: %s | Father 2: %s " % (father1, father2))

                    #crossover
                    childrens = self.population[pop][father1].crossover(self.population[pop][father2], cutPoint)

                    #newPopulation
                    newIndividual.append(childrens[0].mutation(rateMutation))
                    newIndividual.append(childrens[1].mutation(rateMutation))
                    # print("new Cell ", newIndividual)
                newPopulation.append(newIndividual)
                # print("newPopulatio --> ", newPopulation)

            # print("selfPopulation --> ", self.population)
            # print(newPopulation)
            self.population = newPopulation
            for i in range(len(self.population)):
                fitnessList = [] #graphic

                for j in range(len(self.population[i])):
                    # self.population[i][j].fnFitness(beta, sigma)
                    self.population[i][j].fitnessNote = fnFitness(self.population[i][j].chromosome[:, :, :, i], beta,
                                                                  sigma)

                    fitnessList.append(self.population[i][j].fitnessNote)

                    if self.bestSolution < self.population[i][j].fitnessNote:
                        self.bestSolution = self.population[i][j].fitnessNote

            # print("fitnessLIst -> ", fitnessList)
            self.listSolution.append(max(fitnessList)) #graphic

        print("listSolution ", len(self.listSolution))
        print("listSolution ", self.listSolution)

        print("------------------------------")
        print("desvioPadrao %s " % (standardDeviation(fitnessList)))
        print("\nmelhor solucao ", self.bestSolution)
        print("------------------------------")

        plt.figure()
        plt.plot(self.listSolution)