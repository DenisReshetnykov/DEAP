import pandas as pd
import pickle

from metadata_data import ParticipantRatings


#  loading data and creating label vectors
class ParticipantData:
    def __init__(self, nParticipant, __filename=None):
        self.nParticipant = nParticipant
        if __filename is None:
            self.__filename = 'data_preprocessed_python/s'+'{:02}'.format(nParticipant)+'.dat'
        self.df = pickle.load(open(self.__filename, 'rb'), encoding='latin1')
        self.labels = self.df['labels']
        self.data = self.df['data']

    def getValence(self):
        return list(self.labels[:, 0])

    def getArousal(self):
        return list(self.labels[:, 1])

    def getDominance(self):
        return list(self.labels[:, 2])

    def getLiking(self):
        return list(self.labels[:, 3])

    #  read subjective data from participant_ratings.csv and set participant familiarity
    def setFamiliarity(self):
        self.familiarity = ParticipantRatings(self.nParticipant).getFamiliarity()

    def getFamiliarity(self):
        self.setFamiliarity()
        if list(self.familiarity):
            return list(self.familiarity)
        else:
            print('No Familiarity data')
            return None


if __name__ == "__main__":
    p = ParticipantData(3)
    # print(p.data[1, 3].shape)