import numpy as np
import pandas as pd

from participant_data import ParticipantData
from deapdata import EEG_CHANELS, PHYS_DATA_CHANELS, FREQ

BANDS = {'theta': (4, 8), 'slow_alpha': (8, 10), 'alpha': (8, 12), 'beta': (12, 30), 'gamma': (30, FREQ)}
ASYM_BANDS = {'theta': (4, 8), 'alpha': (8, 12), 'beta': (12, 30), 'gamma': (30, FREQ)}
ASYM_ELECTRODE_PAIRS = [(1, 17), (2, 18), (3, 20), (4, 21), (5, 22), (6, 23), (7, 25),
                        (8, 26), (9, 27), (10, 28), (11, 29), (12, 30), (13, 31), (14, 32)]



class ParticipantSignalsFeatures(ParticipantData):
    def __init__(self, nParticipant, __filename=None):
        super().__init__(nParticipant, __filename)
        self.spectralEEGFeatures = {}  # {1: {1: {'theta': 297233.62, 'alpha': 151385.36}}}
        self.spectralEEGAsymetry = {}  # {1: {1: {}}}

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
            Power[bandKey] = sum(fftSignal[int(np.floor(FreqStart / signalFreq * len(signal))): int(
                np.floor(FreqEnd / signalFreq * len(signal)))])
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

    def getSpectralPowerAsymmetry(self, trial, electrodePair, bands, recount=False):
        leftElectrode, rightElectrode = electrodePair
        self.spectralEEGAsymetry[trial] = {}
        self.spectralEEGAsymetry[trial][leftElectrode] = {}
        for band in bands.keys():
            #  TODO придумать как передавать произвольный набор параметров
            paramsLeft = (trial, leftElectrode, bands)
            paramsRight = (trial, rightElectrode, bands)
            self.spectralEEGAsymetry[trial][leftElectrode][band] = self.getEEGSpectralPower(*paramsLeft)[band] - \
                                                                   self.getEEGSpectralPower(*paramsRight)[band]
        return self.spectralEEGAsymetry[trial][leftElectrode]


if __name__ == "__main__":
    p = ParticipantSignalsFeatures(3)
    for trial in range(1,41):
        for electrode in range(1,33):
            p.getEEGSpectralPower(trial, electrode, BANDS, FREQ, 0, 8063)
    # print(p.getEEGSpectralPower(1, 1, BANDS, FREQ, 0, 8063))
    print(p.spectralEEGFeatures)
    print(p.getSpectralPowerAsymmetry(1, (1, 2), ASYM_BANDS))

