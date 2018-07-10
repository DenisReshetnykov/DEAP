import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from participant_data import ParticipantData
from deapdata import EEG_CHANELS, PHYS_DATA_CHANELS, FREQ

BANDS = {'theta': (4, 8), 'slow_alpha': (8, 10), 'alpha': (8, 12), 'beta': (12, 30), 'gamma': (30, FREQ)}
ASYM_BANDS = {'theta': (4, 8), 'alpha': (8, 12), 'beta': (12, 30), 'gamma': (30, FREQ)}
ASYM_ELECTRODE_PAIRS = [(1, 17), (2, 18), (3, 20), (4, 21), (5, 22), (6, 23), (7, 25),
                        (8, 26), (9, 27), (10, 28), (11, 29), (12, 30), (13, 31), (14, 32)]



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

    def computeSpectralPower(self, trial, electrode, bands, signalFreq=FREQ, epochStart=0, epochStop=8063):
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





if __name__ == "__main__":
    p = ParticipantSignalsFeatures(3)
    p.calculateAverageOfDerivative(3)
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

