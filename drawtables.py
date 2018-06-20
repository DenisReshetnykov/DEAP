from metadata_data import ParticipantRatings, ProcessedParticipantRatings, AVSpaceDividedCollections

import pandas as pd
import numpy as np
from scipy import stats

CORRELATION_PARAM = ['Liking', 'Valence', 'Arousal', 'Dominance', 'Familiarity', 'Trial']

def drawRatingsTableInAVSpace(strformat='{0:.1f}'):
    pp = ProcessedParticipantRatings()
    LALV, LAHV, HALV, HAHV = pp.getLevels()
    for space in [LALV, HALV, LAHV, HAHV]:
        print('liking: {0}({1}) valence: {2}({3}) arousal: {4}({5}) dominance: {6}({7}) familiarity: {8}({9})'.format(
            strformat.format(space['Liking'].mean()), strformat.format(space['Liking'].std()),
            strformat.format(space['Valence'].mean()), strformat.format(space['Valence'].std()),
            strformat.format(space['Arousal'].mean()), strformat.format(space['Arousal'].std()),
            strformat.format(space['Dominance'].mean()), strformat.format(space['Dominance'].std()),
            strformat.format(space['Familiarity'].mean()), strformat.format(space['Familiarity'].std()),
        ))

def drawMeanRatingTableInAVSpace(strformat='{0:.1f}'):
    AV = AVSpaceDividedCollections()
    for col in (AV.lalv, AV.halv, AV.lahv, AV.hahv):
        print('liking: {0}({1}) valence: {2}({3}) arousal: {4}({5}) dominance: {6}({7}) familiarity: {8}({9})'.format(
            strformat.format(col.getRatingMean('Liking')), strformat.format(col.getRatingStdMean('Liking')),
            strformat.format(col.getRatingMean('Valence')), strformat.format(col.getRatingStdMean('Valence')),
            strformat.format(col.getRatingMean('Arousal')), strformat.format(col.getRatingStdMean('Arousal')),
            strformat.format(col.getRatingMean('Dominance')), strformat.format(col.getRatingStdMean('Dominance')),
            strformat.format(col.getRatingMean('Familiarity')), strformat.format(col.getRatingStdMean('Familiarity')),
        ))

def drawSubjectiveRatingsCorrelationTable():
    ratings = {participant: ParticipantRatings(nParticipant=participant).df[CORRELATION_PARAM] for participant in
               range(1, 41) if
               not ParticipantRatings(nParticipant=participant).df[CORRELATION_PARAM].dropna().index.empty}
    corr = {key: calculateCorrelationMatrix(ratings[key]).round(2) for key in ratings.keys()}
    meancorr = np.zeros((6, 6))
    fisher = np.zeros((6, 6))
    for col in range(6):
        for row in range(6):
            all_pvalues = [corr[n][col, row] for n in corr.keys()]
            meancorr[col, row] = np.round(pd.DataFrame(all_pvalues).mean(), 2)
            # print(all_pvalues)
            if col != row :
                fisher[col, row] = calculateCorrelationSignificance(all_pvalues)
    meancorr = pd.DataFrame(meancorr, index=CORRELATION_PARAM, columns=CORRELATION_PARAM)
    fisher = pd.DataFrame(fisher, index=CORRELATION_PARAM, columns=CORRELATION_PARAM)
    print(meancorr)
    print(fisher)


def calculateFisherPValueCombine(pvalues):
    #  https://brainder.org/2012/05/11/the-logic-of-the-fisher-method-to-combine-p-values/
    X = -2*np.sum(np.log(pvalues))
    return X


def rTozFisherTransform(r):
    return 0.5 * np.log((1 + r) / (1 - r))


def zTorFisherTransform(z):
    return (np.e**z-np.e**(-z)) / (np.e**z+np.e**(-z))


def zGeneralized(corr, freedom_degree):
    return np.sum([rTozFisherTransform(r)*freedom_degree for r in corr]) / np.sum([freedom_degree for r in corr])


def calculateChiSquared(corr, freedom_degree):
    return np.sum([(rTozFisherTransform(r) - zGeneralized(corr, freedom_degree))**2 * freedom_degree for r in corr])


def isChiIn5Perc(corr):
    chi = calculateChiSquared(corr, 37)
    if chi <= 52.2:
        return True
    else:
        return False


def calculateCorrelationSignificance(corr):
    if isChiIn5Perc(corr):
        return zTorFisherTransform(zGeneralized(corr, 37))
    else:
        return False


def calculateCorrelations(xlist, ylist):
    xmean = np.mean(xlist)
    ymean = np.mean(ylist)
    r = np.sum([xi*yi for xi,yi in zip([(x - xmean) for x in xlist], [(y - ymean) for y in ylist])]) / np.sqrt(
        np.sum([(x - xmean) ** 2 for x in xlist]) * np.sum([(y - ymean) ** 2 for y in ylist]))
    return r


def calculateCorrelationMatrix(df):
    corrmatrix = np.zeros((6, 6))
    for col1 in range(df.shape[1]):
        for col2 in range(df.shape[1]):
            xlist = list(df.iloc[:, col1])
            ylist = list(df.iloc[:, col2])
            corrmatrix[col1, col2] = calculateCorrelations(xlist, ylist)
    return corrmatrix



if __name__ == "__main__":
    # drawSubjectiveRatingsCorrelationTable()
    # print(calculateFisherSignificance())


    # drawRatingsTableInAVSpace(strformat='{0:.2f}')
    # drawMeanRatingTableInAVSpace()