# from contracts import contract

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import scipy.io
import pickle


NPARTICIPANT = 32  #  peoples
NTRIAL = 40  #  video
LABEL_TYPES = ['valence', 'arousal', 'dominance', 'liking', 'familiarity', 'order']
EEG_CHANELS = ['Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'P3', 'P7', 'PO3', 'O1', 'Oz', 'Pz',
              'Fp2', 'AF4', 'Fz', 'F4', 'F8', 'FC6', 'FC2', 'Cz', 'C4', 'T8', 'CP6', 'CP2', 'P4', 'P8', 'PO4', 'O2']
PHYS_DATA_CHANELS = ['hEOG', 'vEOG', 'zEMG', 'tEMG', 'GSR', 'Resp', 'Plethysmograph', 'Temp']
FREQ = 128  #  Hz
STATISTICS = ['min', 'max', 'mean', 'median', 'std', 'skew']


if __name__ == "__main__":
    pass






