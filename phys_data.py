import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from participant_data import ParticipantData
from deapdata import EEG_CHANELS, PHYS_DATA_CHANELS, FREQ, BANDS, ASYM_BANDS, ASYM_ELECTRODE_PAIRS

class ParticipantSignalsFeatures(ParticipantData):
    def __init__(self, nParticipant, __filename=None):
        super().__init__(nParticipant, __filename)
        self.spectralEEGFeatures = {}  # {1: {1: {'theta': 297233, 'alpha': 151385}}}
        self.spectralEEGAsymetry = {}  # {1: {1: {'theta': -2431634, 'alpha': -729425}}}
        self.averageSkinResistance = {}  # {1: -1297, 2: 817, ... , 40: 19011}
        self.averageOfDerivative = {}

    def computeFeatures(self, trials, electrodes, bands, freq, epochStart, epochStop, electrodePairs, asymBands):
        for trial in trials:
            self.calculateAverageSkinResistance(trial)
            for electrode in electrodes:
                self.getEEGSpectralPower(trial, electrode, bands, freq, epochStart, epochStop)
            for electrodePair in electrodePairs:
                self.getSpectralPowerAsymmetry(trial, electrodePair, asymBands)

    def computeSpectralPower(self, trial, electrode, bands, signalFreq=FREQ, epochStart=0, epochStop=None):
        '''
        :param trial: integer, number of video data in dataset (starting from 1)
        :param electrode: integer, number of channel in dataset (starting from 1)
        :param epochStart: integer, starting datapoint (starting from 0)
        :param epochStop: integer, ending datapoint (starting from 0)
        :param signalFreq: integer, signal physical frequency
        :param bands: dict of tuple, where key is band name and value(tuple) is real frequencies border (in Hz)
                      Each element of tuple is a physical frequency and shall not exceed the Nyquist frequency, i.e.,
                      half of sampling frequency. ex.: {'slow_alpha': (8, 10), 'alpha': (8, 12)}
        :return: Power: dict, where key is band name and value is power. ex.:{'slow_alpha': 10, 'alpha': 20}
        '''
        if epochStop is None:
            epochStop = self.data.shape[2]
        signal = self.data[trial-1, electrode-1][epochStart:epochStop]
        fftSignal = abs(np.fft.fft(signal))
        Power = {}
        for bandKey in bands.keys():
            FreqStart, FreqEnd = bands[bandKey]
            Power[bandKey] = int(sum(fftSignal[int(np.floor(FreqStart / signalFreq * len(signal))): int(
                np.floor(FreqEnd / signalFreq * len(signal)))]))
        return Power

    def getEEGSpectralPower(self, trial, electrode, bands, signalFreq=FREQ, epochStart=0, epochStop=8063, recount=False):
        params = (trial, electrode, bands, signalFreq, epochStart, epochStop)
        if trial not in self.spectralEEGFeatures:
            self.spectralEEGFeatures[trial] = {}
            self.spectralEEGFeatures[trial][electrode] = self.computeSpectralPower(*params)
        else:
            if electrode not in self.spectralEEGFeatures[trial] or recount:
                self.spectralEEGFeatures[trial][electrode] = self.computeSpectralPower(*params)
        return self.spectralEEGFeatures[trial][electrode]

    def getSpectralPowerAsymmetry(self, trial, electrodePair, bands, *args, recount=False, **kwargs):
        leftElectrode, rightElectrode = electrodePair
        argsLeft = (trial, leftElectrode, bands) + args
        argsRight = (trial, rightElectrode, bands) + args
        kw = {'recount': recount}
        kw.update(kwargs)
        if trial not in self.spectralEEGAsymetry:
            self.spectralEEGAsymetry[trial] = {}
            self.spectralEEGAsymetry[trial][leftElectrode] = {
            band: self.getEEGSpectralPower(*argsLeft, **kw)[band] - self.getEEGSpectralPower(*argsRight, **kw)[band] for
            band in bands.keys()}
        elif leftElectrode not in self.spectralEEGAsymetry[trial] or recount:
            self.spectralEEGAsymetry[trial][leftElectrode] = {
            band: self.getEEGSpectralPower(*argsLeft, **kw)[band] - self.getEEGSpectralPower(*argsRight, **kw)[band] for
            band in bands.keys()}
        return self.spectralEEGAsymetry[trial][leftElectrode]

    def clearTrendFromSignal(self, signal):
        #  TODO написать функцию вычисления тренда в сигнале и очистки сигнала от него
        return signal

    def calculateAverageSkinResistance(self, trial, recount=False):
        if trial not in self.averageSkinResistance or recount:
            signal = self.data[trial-1, 36]
            self.averageSkinResistance[trial] = int(np.mean(signal))
        return self.averageSkinResistance[trial]

    def calculateAverageOfDerivative(self, trial, recount=False):
        if trial not in self.averageOfDerivative or recount:
            signal = self.data[trial-1, 36]
            plt.figure(figsize=(15, 8))
            plt.plot(signal)
            plt.show()
            # print(signal)
        return averageOfDerivative

class ClearedEEGParticipantFeatures(ParticipantSignalsFeatures):
    def __init__(self, nParticipant, __filename=None):
        super().__init__(nParticipant, __filename)
        self.data = self.data[:, :32, :]

    def outliersToMean(self, data, sigma=2):
        mean = np.mean(data)
        std = np.std(data)
        data = np.array([x_i if (mean - sigma * std < x_i) and (x_i < mean + sigma * std) else mean for x_i in data])
        return data

    def clearOutliers(self):
        for trial in range(40):
            for chanel in range(32):
                self.data[trial, chanel] = self.outliersToMean(self.data[trial, chanel])

    def computeEEGSpectralPower(self, timesize=4):
        self.removeBaseline()
        self.clearOutliers()
        parts = int(self.data.shape[2]/FREQ/timesize)
        for trial in range(1, self.data.shape[0]+1):
            self.spectralEEGFeatures[trial] = {}
            for chanel in range(1, self.data.shape[1]+1):
                self.spectralEEGFeatures[trial][chanel] = {}
                for timeSegment in range(parts):
                    self.spectralEEGFeatures[trial][chanel][(timeSegment + 1) * timesize] = self.computeSpectralPower(
                        trial, chanel, ASYM_BANDS, FREQ, FREQ * timeSegment, FREQ * (timeSegment + 1))
        return self.spectralEEGFeatures


def substractBaseline(data, baseline):
    return data-np.mean(baseline)

def replaceOutliersByMedian(data, sigma=2):
    mean = np.mean(data)
    std = np.std(data)
    print("mean is {} ane std is {}".format(round(mean, 2), round(std, 2)))
    data = np.array([x_i if (mean - sigma*std < x_i) and (x_i < mean + sigma*std) else mean for x_i in data])
    return data

def plotEEG(signals):
    for n in range(4):
        plt.figure(n)
        fig, axes = plt.subplots(8, 1, figsize=(15, 15))
        ax = axes.ravel()
        for i in range(8):
            ax[i].plot(signals[n*8+i])
        # ax[0].set_xlabel("Значение признака")
        # ax[0].set_ylabel("Частота")
        # ax[0].legend(["0", "1"], loc="best")
        fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    p = ClearedEEGParticipantFeatures(4)
    print(p.computeEEGSpectralPower())




    # plotEEG(p.data[0])
    # print(np.mean(p.data[3][6][0:384]))
    # print(substractBaseline(p.data[3][6], p.data[3][6][0:384]))

    # p.computeFeatures(range(1, 41), range(1, 33), BANDS, FREQ, 0, 8063, ASYM_ELECTRODE_PAIRS, ASYM_BANDS)
    # for trial in range(1, 41):
    #     p.calculateAverageSkinResistance(trial)
    # print(p.averageSkinResistance)
    # for trial in range(1,41):
    #     p.calculateAverageSkinResistance(trial)
    #     for electrode in range(1,33):
    #         p.getEEGSpectralPower(trial, electrode, BANDS, FREQ, 0, 8063)
    #     for electrodePair in ASYM_ELECTRODE_PAIRS:
    #         p.getSpectralPowerAsymmetry(trial, electrodePair, ASYM_BANDS)
    # print(p.spectralEEGFeatures)
    # print(p.spectralEEGAsymetry)
    # print(p.averageSkinResistance)

