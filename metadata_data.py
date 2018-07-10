import pandas as pd
import numpy as np


class ParticipantRatings:
    def __init__(self, nParticipant=None, __filename=None):
        if __filename is None:
            self.__filename = 'metadata_csv/participant_ratings.csv'
        self.df = pd.read_csv(self.__filename)
        if nParticipant is not None:
            self.nParticipant = nParticipant
            # self.df = self.df.where(self.df['Participant_id'] == self.nParticipant).dropna(how='all')
            self.df = self.df[self.df['Participant_id'] == self.nParticipant]

    def getArousal(self):
        self.arousal = self.df['Arousal']
        return list(self.arousal)

    def getValence(self):
        self.valence = self.df['Valence']
        return list(self.valence)

    def getDominance(self):
        self.dominance = self.df['Dominance']
        return list(self.dominance)

    def getLiking(self):
        self.liking = self.df['Liking']
        return list(self.liking)

    def getFamiliarity(self):
        self.familiarity = self.df['Familiarity'].dropna()
        return list(self.familiarity)

    def filterByParam(self, trial=None, expID=None):
        if (trial is None) and (expID is None):
            print('Value error: trial and expID both are None. None returned')
            return None
        if trial is not None:
            self.filteredData = self.df.where(self.df['Trial'] == trial).dropna(how='all')
        if expID is not None:
            self.filteredData = self.df.where(self.df['Experiment_id'] == expID).dropna(how='all')
        return self.filteredData


class ProcessedParticipantRatings(ParticipantRatings):
    def __init__(self, nParticipant=None, __filename=None):
        super().__init__(nParticipant, __filename)

    def getLevels(self): #TODO Parametrize this method
        self.LALV = self.df[(self.df['Arousal'] < 5) & (self.df['Valence'] < 5)]
        self.LAHV = self.df[(self.df['Arousal'] < 5) & (self.df['Valence'] >= 5)]
        self.HALV = self.df[(self.df['Arousal'] >= 5) & (self.df['Valence'] < 5)]
        self.HAHV = self.df[(self.df['Arousal'] >= 5) & (self.df['Valence'] >= 5)]
        return self.LALV, self.LAHV, self.HALV, self.HAHV


class ExperimentData(ParticipantRatings):
    def __init__(self, nParticipant=None, __filename=None, experiment=None):
        super().__init__(nParticipant, __filename)
        if experiment is not None:
            self.df = self.filterByParam(expID=experiment)
        self.std = {}
        self.mean = {}

    def getRatingStd(self, rating):
        if rating not in self.std:
            self.std[rating] = self.df[rating].std()
        return self.std[rating]

    def getRatingMean(self, rating):
        if rating not in self.mean:
            self.mean[rating] = self.df[rating].mean()
        return self.mean[rating]


class ExperimentDataCollection:
    def __init__(self):
        # self.name = name
        self.collection = []
        self.expNumbers = []
        self.mean = {}
        self.stdmean = {}
        self.meanstd = {}
        self.stdstd = {}

    def addExperiment(self, experiment):
        self.collection.append(experiment)

    def getRatingMean(self, rating, recount=False):
        if rating not in self.mean or recount:
            self.mean[rating] = np.mean([exp.getRatingMean(rating) for exp in self.collection])
        return self.mean[rating]

    def getRatingStdMean(self, rating, recount=False):
        if rating not in self.stdmean or recount:
            self.stdmean[rating] = np.std([exp.getRatingMean(rating) for exp in self.collection])
        return self.stdmean[rating]

    def getRatingMeanStd(self, rating, recount=False):
        if rating not in self.meanstd or recount:
            self.meanstd[rating] = np.mean([exp.getRatingStd(rating) for exp in self.collection])
        return self.meanstd[rating]

    def getRatingStdStd(self, rating, recount=False):
        if rating not in self.stdstd or recount:
            self.stdstd[rating] = np.std([exp.getRatingStd(rating) for exp in self.collection])
        return self.stdstd[rating]

class AVSpaceDividedCollections:
    def __init__(self):
        [self.lalv, self.lahv, self.halv, self.hahv] = [ExperimentDataCollection() for n in range(4)]
        s = {}
        for i in range(1, 41):
            s[i] = ExperimentData(experiment=i)
            if s[i].getRatingMean('Arousal') < 5 and s[i].getRatingMean('Valence') < 5:
                self.lalv.addExperiment(s[i])
                self.lalv.expNumbers.append(i)
            if s[i].getRatingMean('Arousal') < 5 and s[i].getRatingMean('Valence') >= 5:
                self.lahv.addExperiment(s[i])
                self.lahv.expNumbers.append(i)
            if s[i].getRatingMean('Arousal') >= 5 and s[i].getRatingMean('Valence') < 5:
                self.halv.addExperiment(s[i])
                self.halv.expNumbers.append(i)
            if s[i].getRatingMean('Arousal') >= 5 and s[i].getRatingMean('Valence') >= 5:
                self.hahv.addExperiment(s[i])
                self.hahv.expNumbers.append(i)


if __name__ == "__main__":
    p = ParticipantRatings(2)
    if p.getFamiliarity() != []:
        print(p.familiarity)
    # print(p.familiarity.reset_index(drop=True).iloc[0])



