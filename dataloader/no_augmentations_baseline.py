import numpy as np
import torch
import random

def DataTransform(training_data, config, logger):    
    # To hold the augmented samples
    n = training_data.shape[0]
    augmented_data = torch.empty(n, 1, 3000) 

    for i in range(n):
        aug = training_data[i]
        augmented_data[i] = aug
    return augmented_data 
    # >> (n, 1, 3000) tensor holding the augmented samples
