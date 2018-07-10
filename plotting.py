import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import skew

from metadata_data import ParticipantRatings, AVSpaceDividedCollections
from features import ParticipantFeatureVectors, createHugeFeatureVector, saveHugeFeatureVector, Features

RATINGS = ['Liking', 'Valence', 'Arousal', 'Dominance', 'Familiarity']
SPACES = ['LALV', 'HALV', 'LAHV', 'HAHV']


def getExpIDData(expList=range(1, 41)):
    '''
    Return participant data for list of experiment(video)
    :param expList: list of experiment(video numbers), e.g [1, 5, 12, 33, 22], default [1, ..., 40]
    :return: dictionary where key is experiment number and value is pandas 32-row dataframe
    '''
    expID = {}
    for i in expList:
        expID[i] = ParticipantRatings().filterByParam(expID=i)
    return expID


def calculateRatingMinMaxMean(expID):
    expIDRatingsMinMaxMean = {}
    for key in expID.keys():
        v = expID[key]['Valence']
        a = expID[key]['Arousal']
        d = expID[key]['Dominance']
        l = expID[key]['Liking']
        expIDRatingsMinMaxMean[key] = {'V': [np.min(v), np.max(v), np.mean(v)],
                                       'A': [np.min(a), np.max(a), np.mean(a)],
                                       'D': [np.min(d), np.max(d), np.mean(d)],
                                       'L': [np.min(l), np.max(l), np.mean(l)], }
    return expIDRatingsMinMaxMean


def plotRatings(expIDRatingsMinMaxMean):
    valence = []
    arousal = []
    dominance = []
    liking = []
    expIDMMM = expIDRatingsMinMaxMean
    for key in expIDMMM.keys():
        valence.append(expIDMMM[key]['V'][2])
        arousal.append(expIDMMM[key]['A'][2])
        dominance.append(expIDMMM[key]['D'][2])
        liking.append(expIDMMM[key]['L'][2])
    plt.scatter(arousal, valence, s=[((d - 2) * 5) ** 2 for d in dominance], marker='.', c=liking, cmap='cool')
    if False:  # plot vertical and horizontal line in mean arousal and mean valence
        plt.vlines(np.mean(arousal), np.min(valence), np.max(valence))
        plt.hlines(np.mean(valence), np.min(arousal), np.max(arousal))
    if True:  # plot vertical and horizontal line in middle=5 of arousal and valence scales
        plt.vlines(5, np.min(valence), np.max(valence))
        plt.hlines(5, np.min(arousal), np.max(arousal))
    plt.text(np.min(arousal), np.min(valence), 'LALV', fontsize=15)
    plt.text(np.min(arousal), np.max(valence)-0.5, 'LAHV', fontsize=15)
    plt.text(np.max(arousal)-0.5, np.min(valence), 'HALV', fontsize=15)
    plt.text(np.max(arousal)-0.5, np.max(valence)-0.5, 'HAHV', fontsize=15)
    plt.xlabel('Arousal')
    plt.ylabel('Valence')
    plt.show()


#  Rating distributions for the emotion induction conditions
#  Fig. 7. The distribution of the participants’ subjective ratings per scale (L - liking, V - valence, A - arousal, D -
#  dominance, F - familiarity) for the 4 affect elicitation conditions (LALV, HALV, LAHV, HAHV)
def boxPlotSubjectiveRating():
    AV = AVSpaceDividedCollections()
    data = []
    for col in (AV.lalv, AV.halv, AV.lahv, AV.hahv):
        ratingdata = []
        for rating in RATINGS:
            experimentdata = []
            for experiment in col.collection:
                experimentdata.append(round(experiment.getRatingMean(rating), 1))
            ratingdata.append(experimentdata)
        data.append(ratingdata)
    plt.figure(figsize=(15,8))
    for n in range(4):
        plt.subplot(1, 4, n+1)
        plt.boxplot(data[n], notch=True, sym='.', labels=[Letter[0] for Letter in RATINGS])
        x1, x2, y1, y2 = plt.axis()
        plt.axis((x1, x2, 0, 10))
        plt.xlabel(SPACES[n])
    plt.suptitle('Rating distributions for the emotion induction conditions')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def basicalDataAnalysis(nParticipant):
    p = ParticipantFeatureVectors(nParticipant, from_file=True)
    print('There DF with shape {}.\nThere some stats for features:\n'.format(str(p.featureDF.shape)))
    print('Feature'+' '*20+'min'+' '*10+'max'+' '*8+'mean'+' '*8+'median'+' '*8+'std'+' '*7+'skew')

    for feature in list(p.featureDF.columns):
        FDF = p.featureDF[feature]
        print('{0}{1}{2}{3}{4}{5}{6}'.format('{:18}'.format(feature), '{:13}'.format(np.min(FDF)),
                                             '{:13}'.format(np.max(FDF)), '{:13}'.format(round(np.mean(FDF))),
                                             '{:13}'.format(round(np.median(FDF))), '{:13}'.format(round(np.std(FDF))),
                                             '{:9}'.format(round(skew(FDF), 3))))
    f = 1
    plt.figure(figsize=(15, 15))
    for feature in list(p.featureDF.columns)[0:36]:
        plt.subplot(6, 6, f)
        plt.boxplot(p.featureDF[feature], notch=True, sym='.')
        plt.title(feature)
        f += 1
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    #  TODO визуализация (ящик с усами)

    #  TODO нарисовать гистограммы по каждой фиче

    #  TODO визуализация написать метод строящий корелляцию межды признаками в датафрейме


def plottingData(signal, y):
    plt.figure(figsize=(15, 15))
    plt.scatter(signal, y)
    plt.show()

def plotScaterMatrix():
    X = Features(pd.DataFrame.from_csv('feature_vectors/huge/FV_familiarity.csv'))
    X.normalizeData()

    # positive = X.normDF['Fp1_theta'].where(X.normDF['Fp1_theta'] > 0).dropna()
    # negative = X.normDF['Fp1_theta'].where(X.normDF['Fp1_theta'] < 0).dropna()
    # print(negative)


    y = pd.DataFrame.from_csv('feature_vectors/huge/YV_familiarity.csv').astype(int)

    grr = pd.plotting.scatter_matrix(X.normDF.iloc[:, 210:], c=y['0'].tolist(), figsize=(15, 15), marker='.',
                            hist_kwds={'bins': 20}, alpha=.8)

    # plt.scatter(negative, negative.index.values)


    plt.show()




if __name__ == "__main__":
    plotScaterMatrix()


    # basicalDataAnalysis(2)
    # print(plotRatings(calculateRatingMinMaxMean(getExpIDData())))
