import random

TEST_DATA = [random.randint(0, 10) for n in range(1000)]

class Genome():
    def __init__(self):
        self.genes = []
        self.lifetime = 0
        self.generation = 0
        self.score = 0
        self.parent = None
        self.geneSet = []

    def generate(self, length, geneSet, seed=0):
        self.geneSet = geneSet
        self.seed = seed
        random.seed(seed)
        while len(self.genes) < length:
            sampleSize = min(length - len(self.genes), len(self.geneSet))
            self.genes.extend(random.sample(self.geneSet, sampleSize))
        return self

    def mutate(self, parent):
        self.parent = parent
        self.geneSet = parent.geneSet
        self.genes = list(parent.genes)
        index = random.randrange(0, len(parent.genes))
        newGene, alternate = random.sample(self.geneSet, 2)
        self.genes[index] = alternate if newGene == self.genes[index] else newGene
        return self

    def crossbreeding(self, father, mother):
        self.geneSet = father.geneSet
        self.genes = [gene[random.randint(0, 1)] for gene in zip(father.genes, mother.genes)]
        return self

    def fitness(self, testset=TEST_DATA):
        for pos, gene in enumerate(self.genes):
            if gene == 1:
                self.score += testset[pos]
            else:
                self.score -= testset[pos]
        self.score = 1/(abs(self.score)+1)


class Population():
    def __init__(self):
        self.generation = 0
        self.species = []
        self.bank = []

    def addGenome(self, genome):
        if len(self.species) < self.sizeLimit:
            self.species.append(genome)
        else:
            print("Can't add genome, reached the size limit of the population")

    def generate(self, sizeLimit, genome):
        self.sizeLimit = sizeLimit
        while len(self.species) < self.sizeLimit:
            self.addGenome(genome)
        self.bank = self.species[0]
        return self

    def populate(self):
        self.generation += 1
        populationSize = len(self.species)
        for genome in self.species[:int(populationSize/2)]:
            genome.lifetime += 1
            self.addGenome(Genome().mutate(genome))
        for parents in zip(self.species[:int(populationSize/2)], self.species[int(populationSize/2):]):
            self.addGenome(Genome().crossbreeding(parents[0], parents[1]))

    def evaluate(self):
        for genome in self.species:
            genome.fitness()
        self.species.sort(key=lambda x: x.score)

    def genocide(self):
        self.evaluate()
        if self.species[0].score > self.bank.score:
            self.bank = self.species[0]
        self.species[int(len(self.species)/2):] = []

    def display_best(self):
        # self.evaluate()
        return self.bank.score

def main_cycle(specie):
    specie.display_best()
    print("Start")
    while True and specie.generation != 10000:
        specie.genocide()
        specie.populate()
        # specie.display_best()
        if specie.species[0].score == 1:
            break
        if specie.generation%100 == 0:
            print("Generation {} passed".format(specie.generation))
        if specie.generation%1000 == 0:
            print("Current best score: {}".format(specie.display_best()))
    print('End. Generation passed {}'.format(specie.generation))


if __name__ == "__main__":
    specie = Population().generate(8, Genome().generate(1000, [0, 1],))
    main_cycle(specie)




