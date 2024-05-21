import torch
from torch.utils.data import Dataset
import os
import numpy as np
from .augmentations import DataTransform, bandstop_filter


class Load_Dataset(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, dataset, config, training_mode, logger):
        # Dataset comes in the form of train.pt, val.pt, or test.pt
        super(Load_Dataset, self).__init__()
        self.training_mode = training_mode

        X_train = dataset["samples"]
        # X_train is now a 3 dimensional tensor where each row is a 30 second recording, i.e. X_train[0, :, :] returns the first 30 second recording.
         
        y_train = dataset["labels"]
        # y_train is a one dimensional vector with #of rows total entries. It contains a label for each row in the X_train. 

        if len(X_train.shape) < 3: # adds an extra third dimension to the tensor if it doesn't already have it
            X_train = X_train.unsqueeze(2)

        if X_train.shape.index(min(X_train.shape)) != 1:  # make sure the Channels is in the second dimension
            X_train = X_train.permute(0, 2, 1)

        if isinstance(X_train, np.ndarray): # if the data happens to be in numpy format, then change it into torch format
            self.x_data = torch.from_numpy(X_train)
            self.y_data = torch.from_numpy(y_train).long()
        else:
            if config.bandstop_filter: # Run data through a bandstop filter, removing frequencies above 31 hz.
                n = X_train.shape[0]
                filtered_data = torch.empty(n, 1, 3000) 
                for i in range(n):
                    filtered_data[i] = bandstop_filter(X_train[i])
                self.x_data = filtered_data 
                self.y_data = y_train
                logger.debug("Bandstop filter applied")
            else:
                self.x_data = X_train
                self.y_data = y_train

        self.len = X_train.shape[0]

        if training_mode == "self_supervised":  # no need to apply Augmentations in other modes
            self.aug = DataTransform(self.x_data, config, logger) # calls the augmentation function, see augmentations.py
            # --> returns training data augmented

    def __getitem__(self, index):
        if self.training_mode == "self_supervised":
            return self.x_data[index], self.y_data[index], self.aug[index]
        else:
            return self.x_data[index], self.y_data[index], self.y_data[index] # Must return a third value not to throw errors in trainer, model_train(.)

    def __len__(self):
        return self.len


def data_generator(data_path, configs, training_mode, logger):
    # example data_path: "./data/sleepEDF", contains the train,val,test
    train_dataset = torch.load(os.path.join(data_path, "train.pt"))
    valid_dataset = torch.load(os.path.join(data_path, "val.pt"))
    test_dataset = torch.load(os.path.join(data_path, "test.pt"))

    # Calls custom function on torch.load objects
    train_dataset = Load_Dataset(train_dataset, configs, training_mode, logger)
    valid_dataset = Load_Dataset(valid_dataset, configs, training_mode, logger)
    test_dataset = Load_Dataset(test_dataset, configs, training_mode, logger)
    # train_dataset, valid_dataset, test_dataset, are now all instances of the class Load_Dataset, inherits all of its methods
    # can get_item in self-supervised mode to retrieve training, test data and the augmented training data 

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=configs.batch_size,
                                               shuffle=True, drop_last=configs.drop_last,
                                               num_workers=0)
    
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=configs.batch_size,
                                               shuffle=False, drop_last=configs.drop_last,
                                               num_workers=0)
    
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=configs.batch_size,
                                              shuffle=False, drop_last=False,
                                              num_workers=0)

    return train_loader, valid_loader, test_loader
    # --> returns the tuple (train_loader, valid_loader, test_loader),
    # where each instance is an iterable DataLoader instance, each item is a bath of configs.batch_size (128 for sleepEDF),
    # a batch consists of the Load_Dataset objects, which has the initialization, getitem and length methods.   
    # shuffle=true means that the data will be moved around every epoch, allegedly good practice to avoid overfitting. 
    # configs.drop_last=True which means that drop_last=True, this makes sure that if the last batch is not large enough it will be dropped, not large enough when not divisible by batchsize 128.
    # num_workers=0 means everything will be done in the main thread, can be increased to parallelize
 
