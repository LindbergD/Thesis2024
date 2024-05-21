import numpy as np
import torch
import random

def DataTransform(training_data, config, logger):

    # Set of possible augmentations to sample from
    # augmentation_set = [horizontal_flip, masking, permutation, gaussian_noise, amplitude_scaling, dc_shift] # frequency_gaussian_noise, bandstop_filter
    
    # To hold the augmented samples
    n = training_data.shape[0]
    augmented_data = torch.empty(n, 1, 3000) 

    # Transform each sample with a randomly chosen composite of augmentations (each sample also receives random hyperparameters within specified range)
    for i in range(n):

        # augmentations = random.sample(augmentation_set, 2)
        # aug = augmentations[0](augmentations[1](training_data[i]))

        aug = training_data[i]

        augmented_data[i] = aug
    
    return augmented_data 
    # >> (n, 1, 3000) tensor holding the augmented samples

# Frequency domain gaussian noise
def frequency_gaussian_noise(x, ratio=0.2):
    # FFT
    frequencies = np.fft.fft(x) # >> (1, 3000)

    # Select a random std, then generate the noise
    std = np.random.uniform(low=0, high=ratio)
    augs = np.random.normal(loc=0, scale=std, size=(frequencies.shape))

    # Add the noise to original sample 
    augmented_frequencies = np.array(frequencies + augs, dtype=complex)

    # Recover the time domain signal by taking the real-valued part of the inverse FFT
    recovered_signal = np.fft.ifft(augmented_frequencies)
    recovered_signal = np.real(recovered_signal)

    return torch.tensor(recovered_signal)

# Gaussian noise
def gaussian_noise(x, ratio=0.8): # was 0.2
    # x is of shape (1, 3000)
    # Select a random std for each batch
    std = np.random.uniform(low=0, high=ratio)
    augs = np.random.normal(loc=0, scale=std, size=(x.shape))
    
    return x + augs


# Horizontal flip (backwards signal)
def horizontal_flip(x):
    # Flip the array alongst dimension 1 (flip the columns)
    return torch.flip(x, dims=[1])


# Masking
def masking(x, max_masking=150):

    L = x.shape[1]
    x_aug = x.clone()

    starts = torch.randint(0, L - max_masking, size=(1,)) # returns a random start point
    lengths = torch.randint(0, max_masking, size=(1,)) # returns a length of masking 

    x_aug[:, starts:starts+lengths] = 0
    
    return x_aug


# Amplitude scaling
def amplitude_scaling(x, min_ratio=0.5, max_ratio=2):
    # For a continuous signal f(x), amplitude scaling is defined as A*f(x) where A is a scalar 
    # https://www.tutorialspoint.com/signals-and-systems-amplitude-scaling-of-signals#:~:text=What%20is%20Amplitude%20Scaling%3F,is%20known%20as%20amplitude%20scaling.

    # scaled = x*np.random.uniform(min_ratio, max_ratio, size=(x.shape))

    scaling_factor = np.random.normal(loc=2, scale=1.5)

    return x*scaling_factor


# DC shift
def dc_shift(x, min_shift=-0.00001, max_shift=0.00001): 
    shifted = x + np.random.uniform(min_shift, max_shift)
    return shifted


# Permutation
def permutation(x, max_segments=12):
    orig_steps = np.arange(x.shape[1]) 
    # --> Returns timesteps, in shape (3000,) which contains the values 0 to 2999 in that order
    num_segs = np.random.randint(1, max_segments)
    # --> Returns a np.array that contains the number of segments for each sample (30-second epoch), a number between 1 and 12
    ret = np.zeros_like(x)

    if num_segs > 1:
        splits = np.array_split(orig_steps, num_segs)
        random.shuffle(splits)
        splits = np.concatenate(splits)
        ret = x[0, splits]
        ret = ret.unsqueeze(0)
    else:
        ret = x
    return ret


def bandstop_filter(x):
    # Compute FFT
    fft_result = np.fft.fft(x) # >> (1, 3000)

    # Compute the frequencies
    fs = 100 # Sampling rate
    freqs = np.fft.fftfreq(fft_result.shape[1], 1/fs) # >> (3000, )

    # Filter out all frequencies larger than 31. Motivated by sleep-relevant frequencies.
    # Then select a random set of frequecies to remove, a width of 5 is used by default.
    
    # Contruct a mask
    filter_mask = np.ones(fft_result.shape[1])
    filter_mask[(freqs > 31)] = 0
    filter_mask[(freqs < -31)] = 0
    # Apply mask, removing frequencies larger than 31
    fft_result_filtered = fft_result[0]*filter_mask

    # Select a random center between 0 and 31hz. Remove frequencies in a window of width = 5
    # https://hal.science/hal-03853329/document for center
    # https://proceedings.mlr.press/v136/mohsenvand20a.html for window size 5
    
    width = 5
    center_freq = np.random.uniform(low=0, high=31)

    # Contruct a mask to remove frequencies in the range
    low, high = center_freq-(width/2), center_freq+(width/2)

    filter_mask = np.ones(fft_result.shape[1])
    filter_mask[(freqs >= low) & (freqs <= high)] = 0
    filter_mask[(freqs <= -low) & (freqs >= -high)] = 0
    # Apply mask
    fft_result_filtered = fft_result_filtered*filter_mask

    # Recover the original signal with the inverse FFT and extract the real part
    recovered_signal = np.fft.ifft(fft_result_filtered)
    recovered_signal = np.real(recovered_signal)

    return torch.tensor(recovered_signal).unsqueeze(0)