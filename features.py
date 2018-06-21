from deapdata import EEG_CHANELS, PHYS_DATA_CHANELS, FREQ
from metadata_data import ParticipantRatings
from phys_data import ParticipantSignalsFeatures, BANDS, ASYM_BANDS, ASYM_ELECTRODE_PAIRS

import pandas as pd

class ParticipantFeatureVector:
    def __init__(self, nParticipant, yType='Familiarity'):
        self.nParticipant = nParticipant
        self.ratings = ParticipantRatings(self.nParticipant)
        self.physSignalsFeatures = ParticipantSignalsFeatures(self.nParticipant)
        self.physSignalsFeatures.computeFeatures(range(1, 41), range(1, 33), BANDS, FREQ, 0, 8063, ASYM_ELECTRODE_PAIRS,
                                                 ASYM_BANDS)
        self.featureVector = {}
        self.Y = {}

        # self.df = pd.DataFrame


    def createXVector(self):
        for trial in range(1,41):
            self.featureVector[trial] = {}
            for electrode in range(len(EEG_CHANELS)):
                for band in BANDS.keys():
                    feature_name = EEG_CHANELS[electrode] + '_' + band
                    self.featureVector[trial][feature_name] = \
                    self.physSignalsFeatures.spectralEEGFeatures[trial][electrode+1][band]
            for electrodePair in ASYM_ELECTRODE_PAIRS:
                leftE, rightE = electrodePair
                for band in ASYM_BANDS.keys():
                    feature_name = EEG_CHANELS[leftE-1] + '-' + EEG_CHANELS[rightE-1] + '_' + band
                    self.featureVector[trial][feature_name] = \
                    self.physSignalsFeatures.spectralEEGAsymetry[trial][leftE][band]
            self.featureVector[trial]['avgSkinRes'] = self.physSignalsFeatures.averageSkinResistance[trial]
        return pd.DataFrame.from_dict(self.featureVector, orient='index')  # TODO развернуть






if __name__ == "__main__":
    p = ParticipantFeatureVector(2)
    print(p.createXVector())
    # print(p.ratings.df.head(100))
