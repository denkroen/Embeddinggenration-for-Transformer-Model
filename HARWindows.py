'''
Created on May 18, 2019

@author: fmoya
'''

import os

from torch.utils.data import Dataset


import pandas as pd
import pickle
import torch
import numpy as np

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")



class HARWindows(Dataset):
    '''
    classdocs
    '''


    def __init__(self, csv_file, root_dir):
        """
        Args:
            csv_file (string): Path to the csv file with list of annotated sequences.
            root_dir (string): Directory with all the sequences.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.harwindows = pd.read_csv(csv_file)
        self.root_dir = root_dir

    def __len__(self):
        return len(self.harwindows)

    def __getitem__(self, idx):
        '''
        get single item

        @param data: index of item in List
        @return window_data: dict with sequence window, label of window, and labels of each sample in window
        '''
        window_name = os.path.join(self.root_dir, self.harwindows.iloc[idx, 0])
        f = open(window_name, 'rb')
        data = pickle.load(f, encoding='bytes')
        f.close()

    
        X = np.float32(data['data'])
        y = data['label'][0]
        Y = np.float32(data['label'][1:])

        if 'identity' in data.keys():
            i = data['identity']
            window_data = {"data": X, "label_class": y, "label_attr": Y, "identity": i}
        else:
            window_data = {"data": X, "label_class": y, "label_attr": Y}

        return window_data
