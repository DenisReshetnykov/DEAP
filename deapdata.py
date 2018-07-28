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
BANDS = {'theta': (4, 8), 'slow_alpha': (8, 10), 'alpha': (8, 12), 'beta': (12, 30), 'gamma': (30, FREQ)}
ASYM_BANDS = {'theta': (4, 8), 'alpha': (8, 12), 'beta': (12, 30), 'gamma': (30, FREQ)}
ASYM_ELECTRODE_PAIRS = [(1, 17), (2, 18), (3, 20), (4, 21), (5, 22), (6, 23), (7, 25),
                        (8, 26), (9, 27), (10, 28), (11, 29), (12, 30), (13, 31), (14, 32)]


if __name__ == "__main__":
    pass






