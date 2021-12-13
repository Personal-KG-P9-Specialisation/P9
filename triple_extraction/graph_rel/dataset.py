from torch.utils.data import Dataset
import spacy, json
import torch as T
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import numpy as np
from tqdm import tqdm_notebook as tqdm
class DS(Dataset):
    def __init__(self, data):
        super(DS,self).__init__()

        self.data = data
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        return self.data[idx]

