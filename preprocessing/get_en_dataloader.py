import pickle
import random

import pandas as pd
import numpy as np
from transformers import BertTokenizer


class ENMedicDialogueDataset(torch.utils.data.Dataset):
    def __init__(self, descriptions, patients, doctors, neg_samples=9, max_length=256):
        super().__init__()
        self.descriptions = descriptions.copy()
        self.patients = patients.copy()
        self.doctors = doctors.copy()
        
        # Usually description length might be shorter 
        self.__trim_or_pad_input(self.descriptions, max_length)
        self.__trim_or_pad_input(self.patients, max_length)
        self.__trim_or_pad_input(self.doctors, max_length)
            
        
        self.neg_samples = neg_samples
        

    def __trim_or_pad_input(self, inp, max_length):
        '''
        Trim or pad the input dict into the specified length
        This modifies the dictionary inplace
        '''
        current_len = len(inp['input_ids'][0])
        if current_len < max_length:
            for key, val in inp.items():
                for idx in range(self.__len__()):
                    inp[key][idx] += (max_length - current_len) * [0]
        elif current_len > max_length:
            for key, val in inp.items():
                for idx in range(self.__len__()):
                    inp[key][idx] = inp[key][idx][:max_length]
                    
        
    def __getitem__(self, idx):
        '''
        The item at the specific idx is positive,
        We return a number of negative samples as initialized in self.neg_samples
        
        '''
        
        # 2 x MAX_LENGTH tensor of description and patients response
        item = {key: torch.tensor([self.descriptions[key][idx], self.patients[key][idx]]) for key, val in self.descriptions.items()}
        idxs = np.random.randint(0, self.__len__(), self.neg_samples)
        
        # (1 + neg_sample) x MAX_LENGTH tensor of doctor responses 
        doctor_choices = {key: [val[idx]] for key, val in self.doctors.items()}
        for idx in idxs:
            for key in self.doctors.keys():
                doctor_choices[key].append(self.doctors[key][idx])
                
        doctor_choices = {f"doctor_{key}": torch.tensor(val) for key, val in doctor_choices.items()}
        
        return {**item, **doctor_choices}

    def __len__(self):
        return len(self.descriptions)