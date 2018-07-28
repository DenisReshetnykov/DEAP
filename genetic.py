import random

class Genome():
    def __init__(self):
        self.genes = []
        self.lifetime = 0
        self.generation = 0
        self.score = 0

    def generate(self, length, geneSet, seed=0):
        self.geneSet = geneSet
        self.seed = seed
        random.seed(seed)
        while len(self.genes) < length:
            sampleSize = min(length - len(self.genes), len(self.geneSet))
            self.genes.extend(random.sample(self.geneSet, sampleSize))
        return self.genes

    def mutate(self, parent):
        self.parent = parent
        self.geneSet = parent.geneSet
        self.genes = parent.genes
        index = random.randrange(0, len(parent.genes))
        newGene, alternate = random.sample(self.geneSet, 2)
        self.genes[index] = alternate if newGene == self.genes[index] else newGene
        return self.genes

    def fitness(self):
        pass


class Population():
    def __init__(self):
        self.generation = 0
        self.species = []
        self.bank = []

    def addGenome(self, genome):
        if len(self.species) < self.sizeLimit:
            self.species.extend(genome)
        else:
            print("Can't add genome, reached the size limit of the population")

    def generate(self, sizeLimit, genome):
        self.sizeLimit = sizeLimit
        while len(self.species) < self.sizeLimit:
            self.addGenome(genome)


    def populate(self):


        pass

    def evaluate(self):
        self.species.sort(key=lambda x: x.score)

    def genocide(self):
        self.evaluate()
        self.species[len(self.species)/2 : ] = []


if __name__ == "__main__":
    daddy = Genome()
    print(daddy.generate(10, [0, 1]))
    son = Genome()
    son.mutate(daddy)
    print(son.genes)




