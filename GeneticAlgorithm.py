import random
import numpy as np
import matplotlib.pyplot as plt
import math
from array import array
########################################################################################################################
# Setting GA
########################################################################################################################
# class SetupGA():
#     def __init__(self, cromoSize, poulationSize, generationNumber, mutationRate):
#         self.CHROMOSOME = cromoSize
#         self.POPULATION_SIZE = poulationSize
#         self.GENERATIONS_NUMBER = generationNumber
#         self.MUTATION_RATE = mutationRate


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

        # print("chromosome PHI -->: ", self.chromosome)

        # self.chromosome = phi

        # for i in range(len(pilotsSequence)):
        #     # print("IDV --- %s -> %s" % (i, pilotsSequence[i]))
        #     # for j in range(len(pilotsSequence[i])):
        #     self.chromosome.append(pilotsSequence[i])
        # print("CROMO -> %s " % ( self.chromosome))

    # def fnFitness(self, phi, beta, sigma):
    def fnFitness(self, beta, sigma):
        # print('calcular fitness')

        # l = self.chromosome[0]
        # k = self.chromosome[1]
        # sequence = self.chromosome[2]
        #
        # self.fitnessNote = (l + k)

        phi = self.chromosome

        # print("PHI fitnesss ", phi)

        # f = np.zeros((len(phi), len(phi)))
        # for ell in range(0, len(phi)):
        #     print("fn Lell - ", ell)
        #     for k in range(0, len(phi)):
        #         print("fn K - ", k)
        #         f[k, ell] = beta[k, ell, ell]
        #         deno = 0
        #         print("TESTEEEEEE ", len(phi))
        #         for j in range(0, len(phi)):
        #             print("fn J - ", j)
        #             for kline in range(0, len(phi)):
        #                 print("fn kline - ", kline)
        #                 if j != ell:
        #                     deno += np.inner(phi[k, :, ell], phi[kline, :, j])*beta[kline, j, ell]
        #                     # deno += np.inner(phi[2], phi[2])*beta[kline, j, ell]
        #         deno += sigma
        #         f[k, ell] /= deno
        f = np.zeros((len(phi), len(phi[0][0])))
        for ell in range(0, len(phi[0][0])):
            for k in range(0, len(phi)):
                if k < ell:
                    f[k, ell] = beta[k, ell, ell]
                    deno = 0
                    for j in range(1, len(phi[0][0])):
                        for kline in range(0, len(phi)):
                            # print("ell %s k %s j %s kline %s" % (ell, k, j, kline))
                            if j != ell:
                                if k > ell:
                                    deno += np.inner(phi[k, :, ell], phi[kline, :, j]) * beta[kline, j, ell]
                    deno += sigma
                    f[k, ell] /= deno
            # self.fitnessNote = np.sum(f)
        # print("fitnessNote = ", self.fitnessNote)
            return np.sum(f)




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

        # self.chromosome[0:2]+otherIndvidual.chromosome[2::]
        # children2 = otherIndvidual.chromosome[0:2] + self.chromosome[2::]

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
            # self.chromosome[2] = random.randint(0,9)

            # print("muta ", self.chromosome)

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
    quadrados = []
    for i in range(len(fitnessList)):
        quadrados.append((fitnessList[i]-media) ** 2)
    somaQuadrados = sum(quadrados)
    varianca = somaQuadrados / ((len(fitnessList)) - 1)
    return math.sqrt(varianca)

def fnFitness(phi, beta, sigma):

    f = np.zeros((len(phi), len(phi[0][0])))
    for ell in range(0, len(phi[0][0])):
        for k in range(0, len(phi)):
            # if k < ell:
                f[k, ell] = beta[k, ell, ell]
                deno = 0
                for j in range(1, len(phi[0][0])):
                    for kline in range(0, len(phi)):
                        # print("ell %s k %s j %s kline %s" % (ell, k, j, kline))
                        if j != ell:
                            # if k > ell:
                                deno += np.inner(phi[k, :, ell], phi[kline, :, j]) * beta[kline, j, ell]
                deno += sigma
                f[k, ell] /= deno
        # self.fitnessNote = np.sum(f)
        # print("fitnessNote = ", abs(np.sum(f)))
        return abs(np.sum(f))




########################################################################################################################

########################################################################################################################
# GA
########################################################################################################################

def firstGeneration(phi):
    for ell in range(0, len(phi[0][0])):
        q = list(range(0, len(phi[0])))
        np.random.shuffle(q)
        phi[range(0, len(phi[0])), q, ell] = 1
    return phi
class GeneticAlgorithm():
    def __init__(self):
        self.population = []
        self.bestSolution = 0
        self.listSolution = [] #graphic

    def inicializePopulation(self, populationSize, K, Tp, L, beta):
        # for p in range(populationSize):
        #     self.population.append([])
        #     for i in range(K):
        #         phi = np.zeros((int(K), int(Tp), int(L)))
        #         for ell in range(0, len(phi[0][0])):
        #             q = list(range(0, len(phi[0])))
        #             np.random.shuffle(q)
        #             phi[range(0, len(phi[0])), q, ell] = 1
        #         self.population[p].append(Individual(phi))
        # print("population -> ", self.population)
        # print("population -> ", self.population)


        # for p in range(populationSize):
        #     self.population.append([])
        #     # for i in range(K):
        #     phi = np.zeros((len(beta), len(beta), len(beta[0]), int(populationSize)))
        #     for ell in range(0, len(phi[0][0])):
        #         q = list(range(0, len(phi[0])))
        #         np.random.shuffle(q)
        #         phi[range(0, len(phi[0])), q, ell] = 1
        #         self.population[p].append(Individual(phi))
        # # print("population -> ", self.population)
        # # print("population -> ", self.population)


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
        # for population in range(len(self.population)):
        #     for individual in range(len(self.population[population])):
        # total += self.population[population][individual].fitnessNote

        # print("fitnessNoted",self.population[0].fitnessNote)

        for pop in population:
            # for individual in pop:
            total += pop.fitnessNote
        # print("total ", total)
        return total

    def rouletteVitiate(self, sumFitness, populacaoAtual):
        print("sumFItness ", sumFitness)
        # print("populationLen ", len(self.population[0]))
        father = -1
        randomSort = np.random.uniform(0,sumFitness)
        sumLoopsRoulette = 0
        i = 0
        while i < len(self.population[0]) and sumLoopsRoulette < randomSort:
            # print("cellFitnote",cell[i].fitnessNote)
            # print("abs fitness ", populacaoAtual[i].fitnessNote)
            sumLoopsRoulette += populacaoAtual[i].fitnessNote
            father += 1
            i += 1
        return father

    ########################################################################################################################
    # RUN GA
    ########################################################################################################################



    def runGA(self, generationsNumber, rateMutation, beta, sigma, cutPoint, populationSize, K, Tp, L):
        self.inicializePopulation(populationSize, K, Tp, L, beta)

        # print("population -------> ",self.population)
        print("population SIZE -------> ",len(self.population))

        pop = np.zeros((len(beta), len(beta), len(beta[0]), int(populationSize)))


        for i in range(len(self.population)):

            # print("fist ", firstGeneration(pop[:, :, :, i]))

            pop[:, :, :, i] = firstGeneration(pop[:, :, :, i])

            for j in range(len(self.population[i])):
                # print("fist chromo ", self.population[i][j].chromosome)
                #     self.population[i][j].fnFitness(beta, sigma)

                # self.population[i][j].fnFitness(beta, sigma)
                # print("FN fitness", fnFitness(pop[:, :, :, i], beta, sigma))
                print("FN fitness", fnFitness(self.population[i][j].chromosome[:, :, :, i], beta, sigma))
                # self.population[i][j].fitnessNote = fnFitness(pop[:, :, :, i], beta, sigma)
                self.population[i][j].fitnessNote = fnFitness(self.population[i][j].chromosome[:, :, :, i], beta, sigma)



            # print("pulation %s i %s" % (self.population[i].chromosome, i))

        # self.printPopulation()
        # self.sortPopulation()
        self.printPopulation()

        for geracao in range(generationsNumber):
            newPopulation = []
            print("GERACAO ", geracao)

            for pop in range(len(self.population)):
                print("individual ", pop)
                newIndividual = []
                sumEvaluations = self.sumFitnessRate(self.population[pop])
                # print("SUM Evaluation ", sumEvaluations)
                for individualGenerated in range(0, len(self.population), 2):
                    father1 = self.rouletteVitiate(sumEvaluations, self.population[pop])
                    father2 = self.rouletteVitiate(sumEvaluations, self.population[pop])
                    print("Father 1: %s | Father 2: %s " % (father1, father2))

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

            print("fitnessLIst -> ", fitnessList)
            self.listSolution.append(max(fitnessList)) #graphic

        print("listSolution ", len(self.listSolution))
        print("listSolution ", self.listSolution)


        print("------------------------------")
        print("desvioPadrao %s " % (standardDeviation(fitnessList)))
        print("\nmelhor solucao ", self.bestSolution)
        print("------------------------------")

        plt.figure()
        plt.plot(self.listSolution)


