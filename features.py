from deapdata import EEG_CHANELS, PHYS_DATA_CHANELS, FREQ, STATISTICS
from metadata_data import ParticipantRatings
from phys_data import ParticipantSignalsFeatures, BANDS, ASYM_BANDS, ASYM_ELECTRODE_PAIRS
from eventlogger import EventLogger

from datetime import datetime
import numpy as np
import os
import pandas as pd
import random
from scipy.stats import skew

class ParticipantFeatureVectors:
    def __init__(self, nParticipant, from_file = False):
        self.nParticipant = nParticipant
        self.events = EventLogger()
        self.ratings = ParticipantRatings(self.nParticipant)
        self.featureVectors = {}  #  {1:{'Fp1_theta': 2453476, 'Fp1_slow_alpha': 482418, ... , 'avgSkinRes': 69'}}
        self.featureDF = pd.DataFrame  #  use special function
        self.Y = {}
        for trial in range(1,41):
            self.featureVectors[trial] = {}
            self.Y[trial] = 0
        #  seed and variables for splitting data
        self.randomSeed = random.randint(1, 1000000)
        self.X_train = {}
        self.X_test = {}
        self.X_validation = {}
        self.Y_train = {}
        self.Y_test = {}
        self.Y_validation = {}
        #  if we not use precomputed feature from file - compute it from raw signals
        if not from_file:
            self.ratings = ParticipantRatings(self.nParticipant)
            self.physSignalsFeatures = ParticipantSignalsFeatures(self.nParticipant)
            self.physSignalsFeatures.computeFeatures(range(1, 41), range(1, 33), BANDS, FREQ, 0, 8063,
                                                     ASYM_ELECTRODE_PAIRS, ASYM_BANDS)
        else:
            self.loadFeatureVectorsFromCSV()
            self.convertFeatureVectorsToDataFrame()

    def fillFeatureVectors(self):
        self.addEEGSpectralToFeatureVector()
        self.addEEGAsymetryToFeatureVector()
        self.addGSRToFeatureVector()

    def createYVector(self, yType = 'f'):
        self.Y = {}
        if yType == 'f':
            if self.ratings.getFamiliarity() != []:
                for trial in self.featureVectors.keys():
                    self.Y[trial] = self.ratings.familiarity.reset_index(drop=True).iloc[trial-1]
        elif yType == 'a':
            self.ratings.getArousal()
            for trial in self.featureVectors.keys():
                self.Y[trial] = self.ratings.arousal.reset_index(drop=True).iloc[trial - 1]
        elif yType == 'v':
            self.ratings.getValence()
            for trial in self.featureVectors.keys():
                self.Y[trial] = self.ratings.valence.reset_index(drop=True).iloc[trial - 1]
        elif yType == 'l':
            self.ratings.getLiking()
            for trial in self.featureVectors.keys():
                self.Y[trial] = self.ratings.liking.reset_index(drop=True).iloc[trial - 1]
        elif yType == 'd':
            self.ratings.getDominance()
            for trial in self.featureVectors.keys():
                self.Y[trial] = self.ratings.dominance.reset_index(drop=True).iloc[trial - 1]
        elif yType == 'save':
            for type in ['f', 'a', 'v', 'd', 'l']:
                self.createYVector(yType=type)
                self.saveYVectorToCSV(yType=type)
        else:
            print('No such Y vector type')

    def convertFeatureVectorsToDataFrame(self):
        self.featureDF = pd.DataFrame.from_dict(self.featureVectors, orient='index')

    def addEEGSpectralToFeatureVector(self):
        for trial in self.featureVectors.keys():
            for electrode in range(len(EEG_CHANELS)):
                for band in BANDS.keys():
                    feature_name = EEG_CHANELS[electrode] + '_' + band
                    self.featureVectors[trial][feature_name] = \
                    self.physSignalsFeatures.spectralEEGFeatures[trial][electrode+1][band]

    def addEEGAsymetryToFeatureVector(self):
        for trial in self.featureVectors.keys():
            for electrodePair in ASYM_ELECTRODE_PAIRS:
                leftE, rightE = electrodePair
                for band in ASYM_BANDS.keys():
                    feature_name = EEG_CHANELS[leftE-1] + '-' + EEG_CHANELS[rightE-1] + '_' + band
                    self.featureVectors[trial][feature_name] = \
                    self.physSignalsFeatures.spectralEEGAsymetry[trial][leftE][band]

    def addGSRToFeatureVector(self):
        for trial in self.featureVectors.keys():
            self.featureVectors[trial]['avgSkinRes'] = self.physSignalsFeatures.averageSkinResistance[trial]

    def randomSplitSetForTraining(self, train=70, test=30, validation=0, seed=None):
        '''
        Split self.featureVectors and self.Y in random train, test and validation parts in a given proportions
        :param train: proportion of train part, default 70
        :param test: proportion of test part, default 30
        :param validation: proportion of validation part, default 0
        :param seed: seed for random, if None - there will be self.randomSeed used
        :return: self.X_train, self.X_test, self.X_validation, self.Y_train, self.Y_test, self.Y_validation - feature
        and target variable set divided in a given proportion
        '''
        #  init and fill proportion variables
        self.trainPart = train
        self.testPart = test
        self.validationPart = validation

        #  get random sample from feature vector index of test proportion length
        if seed is None:
            seed = self.randomSeed
        random.seed(seed)
        train_index = self.featureVectors.keys()
        test_index = random.sample(train_index,
                                   round(len(self.featureVectors.keys()) * test / (train + test + validation)))
        test_index.sort() #  to have ordered index
        train_index = [item for item in train_index if item not in test_index]

        #  Not all model requires validation set, so we could skip it creation in such case
        if validation != 0:
            validation_index = random.sample(train_index, round(
                len(self.featureVectors.keys()) * validation / (train + test + validation)))
            validation_index.sort()
            train_index = [item for item in train_index if item not in validation_index]

        #  create dict by created index
        self.X_train = {key: self.featureVectors[key] for key in train_index}
        try:
            self.Y_train = {key: self.Y[key] for key in train_index}
        except KeyError:
            errorMsg = 'Participant {} self.Y is empty, so no data for {}'.format(str(self.nParticipant),
                                                                                  'self.Y_train')
            self.events.addEvent(204, errorMsg)
            print(errorMsg)
        self.X_test = {key: self.featureVectors[key] for key in test_index}
        try:
            self.Y_test = {key: self.Y[key] for key in test_index}
        except KeyError:
            errorMsg = 'Participant {} self.Y is empty, so no data for {}'.format(str(self.nParticipant), 'self.Y_text')
            self.events.addEvent(204, errorMsg)
            print(errorMsg)
        if validation != 0:
            self.X_validation = {key: self.featureVectors[key] for key in validation_index}
            try:
                self.Y_validation = {key: self.Y[key] for key in validation_index}
            except KeyError:
                errorMsg = 'Participant {} self.Y is empty, so no data for {}'.format(str(self.nParticipant),
                                                                                      'self.Y_validation')
                self.events.addEvent(204, errorMsg)
                print(errorMsg)
        return self.X_train, self.Y_train, self.X_test, self.Y_test, self.X_validation, self.Y_validation

    def saveSplitedSetToCSV(self, seed=None):
        if seed is None:
            seed = self.randomSeed
        names = ['X_train', 'X_test', 'X_validation', 'Y_train', 'Y_test', 'Y_validation']
        sets = [self.X_train, self.X_test, self.X_validation, self.Y_train, self.Y_test, self.Y_validation]
        pathname = 'training_data/seed={}&train={}&test={}&val={}/'.format(str(seed), str(self.trainPart),
                                                                           str(self.testPart), str(self.validationPart))
        if not os.path.isdir(pathname):
            os.makedirs(pathname)
        for name, set in zip(names, sets):
            if set != {}:
                file_name = '{1}_{2}_{0}.csv'.format(str(seed), str(self.nParticipant), name)
                pd.DataFrame.from_dict(set, orient='index').to_csv(pathname+file_name)

    def saveFeatureVectorToCSV(self):
        filename = 'feature_vectors/FV{}.csv'.format(str(self.nParticipant))
        pd.DataFrame.from_dict(self.featureVectors, orient='index').to_csv(filename)

    def saveYVectorToCSV(self, yType):
        filename = 'feature_vectors/YV'+str(self.nParticipant)+yType+'.csv'
        pd.DataFrame.from_dict(self.Y, orient='index').to_csv(filename)

    #  in most cases we no need to recalculate features from data, so it's necessary to load the previously computed
    #  (and saved to *.csv) featureVectors
    def loadFeatureVectorsFromCSV(self):
        filename = 'feature_vectors/FV{}.csv'.format(str(self.nParticipant))
        self.featureVectors = pd.DataFrame.from_csv(filename).to_dict(orient='index')


def createHugeFeatureVector(seed, from_file=False):
    if from_file:
        if not os.path.isfile('feature_vectors/huge/'+str(seed)+'X_train.npy'):
            saveHugeFeatureVector(seed)
        X_huge_train = np.load('feature_vectors/huge/' + str(seed) + 'X_train.npy')
        Y_huge_train = np.load('feature_vectors/huge/' + str(seed) + 'Y_train.npy')
        X_huge_test = np.load('feature_vectors/huge/' + str(seed) + 'X_test.npy')
        Y_huge_test = np.load('feature_vectors/huge/' + str(seed) + 'Y_test.npy')
    else:
        X_huge_train = []
        Y_huge_train = []
        X_huge_test = []
        Y_huge_test = []
        for part in range(1, 33):
            #create participant object, split feature vectors,
            p = ParticipantFeatureVectors(part, from_file=True)
            p.createYVector()
            X_train, Y_train, X_test, Y_test, X_validation, Y_validation = p.randomSplitSetForTraining(70, 30, seed=seed)
            p.saveSplitedSetToCSV(seed=seed)
            X_train = np.array([list(X_train[trial].values()) for trial in X_train.keys()])
            Y_train = np.array([Y_train[trial] for trial in Y_train.keys()])
            X_test = np.array([list(X_test[trial].values()) for trial in X_test.keys()])
            Y_test = np.array([Y_test[trial] for trial in Y_test.keys()])
            # use only Participant with necessary label vector
            if Y_train != []:
                X_huge_train.extend(X_train)
                Y_huge_train.extend(Y_train)
                X_huge_test.extend(X_test)
                Y_huge_test.extend(Y_test)
            else:
                print('Y_train for {} is empty'.format(part))
    return X_huge_train, Y_huge_train, X_huge_test, Y_huge_test

def saveHugeFeatureVector(seed):
    X_huge_train, Y_huge_train, X_huge_test, Y_huge_test = createHugeFeatureVector(seed)
    np.save('feature_vectors/huge/' + str(seed) + 'X_train', X_huge_train)
    np.save('feature_vectors/huge/' + str(seed) + 'Y_train', Y_huge_train)
    np.save('feature_vectors/huge/' + str(seed) + 'X_test', X_huge_test)
    np.save('feature_vectors/huge/' + str(seed) + 'Y_test', Y_huge_test)

def convertCategorialFeatureTo01(series):
    print('There such unique categorial values: {}'.format(sorted(series.unique())))
    convert_dict = {entry: [0] * sorted(series.unique()).index(entry) + [1] + [0] * (
                len(series.unique()) - 1 - sorted(series.unique()).index(entry)) for entry in series.unique()}
    series = series.map(convert_dict)
    return series

def combineFV():
    # features = []
    for type in ['arousal', 'dominance', 'liking', 'valence']:
        Ys = []
        for part in range(1, 33):
        # features.append(pd.DataFrame.from_csv('feature_vectors/FV'+str(part)+'.csv'))
            Ys.append(pd.DataFrame.from_csv('feature_vectors/YV' + str(part) + type[0] + '.csv'))
        result = pd.concat(Ys)
        print(result)
        filename = 'feature_vectors/huge/YV_' + type + '.csv'
        result.to_csv(filename)
    return result


class Features:
    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.normDF = pd.DataFrame
        self.normCoef = {}
        self.features = self.dataframe.columns
        self.statistics = {}
        self.statisticsDF = pd.DataFrame

    def getFeatures(self):
        return self.features

    def setStatistics(self):
        for feature in self.features:
            featureData = self.dataframe[feature]
            zipper = zip(STATISTICS, [np.min(featureData), np.max(featureData), np.round(np.mean(featureData)),
                                      np.round(np.median(featureData)), np.round(np.std(featureData)),
                                      np.round(skew(featureData), 3)])
            self.statistics[feature] = {statKey: statValue for (statKey, statValue) in zipper}
        self.statisticsDF = pd.DataFrame.from_dict(self.statistics, orient='index', columns=STATISTICS)

    def getStatistics(self, feature):
        return self.statistics[feature]

    def getFeatureVector(self, feature):
        return self.dataframe[feature]

    def normalizeData(self):
        self.normDF = self.dataframe
        self.setStatistics()
        for feature in self.features:
            range = self.statistics[feature]['max'] - self.statistics[feature]['min']
            self.normDF[feature] = (self.dataframe[feature] - self.statistics[feature]['mean']) / range
        return self.normDF

    def convertCategorialFeatureTo01(self, feature):
        series = pd.Series(self.dataframe[feature])
        print('There such unique categorial values: {}'.format(sorted(series.unique())))
        convert_dict = {entry: [0] * sorted(series.unique()).index(entry) + [1] + [0] * (
                len(series.unique()) - 1 - sorted(series.unique()).index(entry)) for entry in series.unique()}
        series = series.map(convert_dict)
        return series

    def covarianceMatrix(self, features):
        pass # TODO визуализация матрицы ковариации


    #  TODO Класс делегирует методы визуализации в модуль plotting. Класс делегирует методы сериализации и
    #  TODO десереализации своих атрибутов в служебный класс ObjectSerialize.


if __name__ == "__main__":
    pass
    # combineFV()
    # y = pd.DataFrame.from_csv('feature_vectors/huge/YV_familiarity.csv').astype(int)


    # p = ParticipantFeatureVectors(4, from_file=True)
    # p.createYVector()
    # ser = pd.Series(p.Y)

    # p.randomSplitSetForTraining(seed=111)
    # p.saveSplitedSetToCSV(seed=111)


    # starttime = datetime.today()
    # print('Star for {} participant at {}'.format(str(2), str(starttime)))
    # p = ParticipantFeatureVectors(2)
    # finishtime = datetime.today()
    # print('ended after '+str(finishtime-starttime))
    #
    # firststarttime = datetime.today()
    # print('Star at {}'.format(str(firststarttime)))
    # for i in range(1, 33):
    #     starttime = datetime.today()
    #     print('Star for {} participant at {}'.format(str(i), str(starttime)))
    #     p = ParticipantFeatureVectors(i)
    #     p.fillFeatureVectors()
    #     p.saveFeatureVectorToCSV()
    #     p.createYVector('save')
    #     finishtime = datetime.today()
    #     print('ended after ' + str(finishtime - starttime))
    # print('Last ended after ' + str(finishtime - firststarttime))

