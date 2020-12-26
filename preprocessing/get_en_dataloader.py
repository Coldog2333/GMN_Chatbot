from typing import Dict, Tuple, Union, List
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

from transformers import BertTokenizer
from transformers.tokenization_utils_base import BatchEncoding


class ENMedicDialogueDataset(Dataset):
    '''
    A dataloader for english medical dataset
    This dataloader will return
    - Description and Patient response as 2-turn dialogue
    - 1 positive response and 9(default) negative sampled responses
      as the targets


    '''

    def __init__(
        self,
        descriptions: BatchEncoding,
        patients: BatchEncoding,
        doctors: BatchEncoding,
        neg_samples: int = 9,
        max_length: int = 256
    ):
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
            for key, _ in inp.items():
                for idx in range(self.__len__()):
                    inp[key][idx] += (max_length - current_len) * [0]
        elif current_len > max_length:
            for key, _ in inp.items():
                for idx in range(self.__len__()):
                    inp[key][idx] = inp[key][idx][:max_length]

    def __getitem__(self, idx):
        '''
        The item at the specific idx is positive,
        We return a number of negative samples as initialized in self.neg_samples

        '''

        # 2 x MAX_LENGTH tensor of description and patients response
        item = {key: torch.tensor([self.descriptions[key][idx], self.patients[key][idx]])
                for key, val in self.descriptions.items()}
        idxs = np.random.randint(0, self.__len__(), self.neg_samples)

        # (1 + neg_sample) x MAX_LENGTH tensor of doctor responses
        doctor_choices = {key: [val[idx]] for key, val in self.doctors.items()}
        for idx in idxs:
            for key in self.doctors.keys():
                doctor_choices[key].append(self.doctors[key][idx])

        doctor_choices = {f"doctor_{key}": torch.tensor(
            val) for key, val in doctor_choices.items()}

        return {**item, **doctor_choices}

    def __len__(self):
        return len(self.descriptions)


def merge_batchencoder(
    left_batchencoder: BatchEncoding,
    right_batchencoders: Union[BatchEncoding, List[BatchEncoding]]
) -> None:
    '''
    Merge the batchencoder from the right into the left (inplace)
    '''
    for key in left_batchencoder:
        if isinstance(right_batchencoders, list):
            for right_batchencoder in right_batchencoders:
                left_batchencoder[key] += right_batchencoder[key]
        else:
            left_batchencoder[key] += right_batchencoders[key]


def get_training_dev_test_dataloader(debugging=False, max_length=256) -> Tuple[BatchEncoding, BatchEncoding, BatchEncoding]:
    '''
    Build the dataloader train, dev, and test
    For simplicity, we will specify

    Train
    - healthcaremagic_splitted_idname_1
    - healthcaremagic_splitted_idname_2
    - healthcaremagic_splitted_idname_3

    Dev
    - healthcaremagic_splitted_idname_4

    Test
    - icliniq_splitted_idname
    '''
    import pickle
    MAX_LENGTH = 256
    data_path = "data"
    roles = ['description', 'patient', 'doctor']
    filename_list = {
        'train': [
            'healthcaremagic_splitted_idname_1',
            'healthcaremagic_splitted_idname_2',
            'healthcaremagic_splitted_idname_3',
        ],
        'dev': ['healthcaremagic_splitted_idname_4', ],
        'test': ['icliniq_splitted_idname', ]
    }
    if debugging:
        filename_list = {
            'train': [
                'icliniq_splitted_idname',
            ],
            'dev': ['icliniq_splitted_idname', ],
            'test': ['icliniq_splitted_idname', ]
        }

    batchencoder_bucket = {
        'train': {f'{role}s': None for role in roles},
        'dev': {f'{role}s': None for role in roles},
        'test': {f'{role}s': None for role in roles}
    }

    for data_type, filenames in filename_list.items():
        for filename in filenames:
            for role in roles:
                with open(f"{data_path}/{filename}_{MAX_LENGTH}_{role}.pkl", 'rb') as f:
                    batch_encoder = pickle.load(f)
                if batchencoder_bucket[data_type][f'{role}s'] is None:
                    batchencoder_bucket[data_type][f'{role}s'] = batch_encoder
                else:
                    merge_batchencoder(
                        batchencoder_bucket[data_type][f'{role}s'], batch_encoder)

    train_loader = ENMedicDialogueDataset(
        **batchencoder_bucket['train'],
        max_length=max_length
    )
    dev_loader = ENMedicDialogueDataset(
        **batchencoder_bucket['dev'],
        max_length=max_length
    )
    test_loader = ENMedicDialogueDataset(
        **batchencoder_bucket['test'],
        max_length=max_length
    )

    return train_loader, dev_loader, test_loader
