import numpy as np
import torch
import random

def DataTransform(training_data, config, logger):
    
    # Gather the two augmentations and their parameters
    aug1, aug2 = determine_augmentation_set(config, logger) # aug1 is of the format: [augmentation, parameters]

    # Initialize array to hold the augmented samples
    n = training_data.shape[0]
    augmented_data = torch.empty(n, 1, 3000) 

    # Apply a single augmentation OR a composite of two augmentations
    if aug2[0] == None:
        for i in range(n):
            aug = aug1[0](training_data[i], aug1[1])    # aug1[0], aug1[1] >> augmentation, parameters
            augmented_data[i] = aug
    else:
        for i in range(n):
            aug = aug1[0](aug2[0](training_data[i], aug2[1]), aug1[1])
            augmented_data[i] = aug

    return augmented_data 
    # >> (n, 1, 3000) tensor holding the augmented samples


# Frequency domain gaussian noise
def frequency_scaling(x, *args):
    min_scale, max_scale = args[0][0], args[0][1]
    frequencies = np.fft.fft(x) # >> (1, 3000)
    # Select a scalar multiplier
    scalar = np.random.uniform(min_scale, max_scale, size=(1,))
    # Scale the frequency representation
    augmented_frequencies = np.array(frequencies*scalar, dtype=complex)
    # Recover the time domain signal by taking the real-valued part of the inverse FFT
    recovered_signal = np.fft.ifft(augmented_frequencies)
    recovered_signal = np.real(recovered_signal)
    return torch.tensor(recovered_signal)


# Gaussian noise
def gaussian_noise(x, *args):
    # x is of shape (1, 3000)
    std = args[0]
    noise = np.random.normal(loc=0, scale=std, size=(x.shape))
    return x + noise


# Horizontal flip (backwards signal)
def horizontal_flip(x, *args):
    # Flip the array alongst dimension 1 (flip the columns)
    return torch.flip(x, dims=[1])


# Masking
def masking(x, *args):
    max_masking = args[0]
    L = x.shape[1]
    x_aug = x.clone()

    starts = torch.randint(0, L - max_masking, size=(1,)) # returns a random start point
    lengths = torch.randint(0, max_masking, size=(1,)) # returns a length of masking 

    x_aug[:, starts:starts+lengths] = 0
    return x_aug


# Amplitude scaling
def amplitude_scaling(x, *args):
    # For a continuous signal f(x), amplitude scaling is defined as A*f(x) where A is a scalar 
    # https://www.tutorialspoint.com/signals-and-systems-amplitude-scaling-of-signals#:~:text=What%20is%20Amplitude%20Scaling%3F,is%20known%20as%20amplitude%20scaling.
    min_ratio, max_ratio = args[0][0], args[0][1]
    scaling_factor = np.random.uniform(min_ratio, max_ratio, size=(1,))

    return x*scaling_factor


# DC shift
def dc_shift(x, *args):
    min_shift, max_shift = args[0][0], args[0][1]
    shifted = x + np.random.uniform(min_shift, max_shift)
    return shifted


# Permutation
def permutation(x, *args):
    max_segments = args[0]
    orig_steps = np.arange(x.shape[1]) 
    # >> timesteps, in shape (3000,) which contains the values 0 to 2999 in that order
    num_segs = np.random.randint(1, max_segments)
    # >> np.array that contains the number of segments for each sample (30-second epoch), a number between 1 and 12
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


def bandstop_filter(x, *args):
    cutoff = args[0]
    # Compute FFT
    fft_result = np.fft.fft(x) # >> (1, 3000)

    # Compute the frequencies
    fs = 100 # Sampling rate
    freqs = np.fft.fftfreq(fft_result.shape[1], 1/fs) # >> (3000, )

    # Filter out all frequencies larger than 31. Motivated by sleep-relevant frequencies.
    # Then select a random set of frequecies to remove, a width of 5 is used by default.
    
    # Contruct a mask
    filter_mask = np.ones(fft_result.shape[1])
    filter_mask[(freqs > cutoff)] = 0
    filter_mask[(freqs < -cutoff)] = 0
    # Apply mask, removing frequencies larger than 31
    fft_result_filtered = fft_result[0]*filter_mask

    # Recover the original signal with the inverse FFT and extract the real part
    recovered_signal = np.fft.ifft(fft_result_filtered)
    recovered_signal = np.real(recovered_signal)

    return torch.tensor(recovered_signal).unsqueeze(0)

def determine_augmentation_set(config, logger):
    if config.run_description == "gaussian_gaussian":
        logger.debug("Selected gaussian-gaussian")
        return [gaussian_noise, config.augmentation.gaussian_std], [None]
    elif config.run_description == "gaussian_scaling":
        logger.debug("Selected gaussian-scaling")
        return [gaussian_noise, config.augmentation.gaussian_std], [amplitude_scaling, config.augmentation.scale]
    elif config.run_description == "gaussian_shift":
        logger.debug("Selected gaussian-shift")
        return [gaussian_noise, config.augmentation.gaussian_std], [dc_shift, config.augmentation.shift]
    elif config.run_description == "gaussian_mask":
        logger.debug("Selected gaussian-mask")
        return [gaussian_noise, config.augmentation.gaussian_std], [masking, config.augmentation.mask]
    elif config.run_description == "gaussian_permutation":
        logger.debug("Selected gaussian-permutation")
        return [gaussian_noise, config.augmentation.gaussian_std], [permutation, config.augmentation.max_seg]
    elif config.run_description == "gaussian_flip":
        logger.debug("Selected gaussian-flip")
        return [gaussian_noise, config.augmentation.gaussian_std], [horizontal_flip, 0]
    elif config.run_description == "gaussian_frequency":
        logger.debug("Selected gaussian-frequency")
        return [gaussian_noise, config.augmentation.gaussian_std], [frequency_scaling, config.augmentation.scale]
    elif config.run_description == "gaussian_bandstop":
        logger.debug("Selected gaussian-bandstop")
        return [gaussian_noise, config.augmentation.gaussian_std], [bandstop_filter, config.augmentation.bandstop]

    elif config.run_description == "scaling_scaling":
        logger.debug("Selected scaling-scaling")
        return [amplitude_scaling, config.augmentation.scale], [None]
    elif config.run_description == "scaling_shift":
        logger.debug("Selected scaling-shift")
        return [amplitude_scaling, config.augmentation.scale], [dc_shift, config.augmentation.shift]
    elif config.run_description == "scaling_mask":
        logger.debug("Selected scaling-mask")
        return [amplitude_scaling, config.augmentation.scale], [masking, config.augmentation.mask]
    elif config.run_description == "scaling_permutation":
        logger.debug("Selected scaling-permutation")
        return [amplitude_scaling, config.augmentation.scale], [permutation, config.augmentation.max_seg]
    elif config.run_description == "scaling_flip":
        logger.debug("Selected scaling-flip")
        return [amplitude_scaling, config.augmentation.scale], [horizontal_flip, 0]
    elif config.run_description == "scaling_frequency":
        logger.debug("Selected scaling-frequency")
        return [amplitude_scaling, config.augmentation.scale], [frequency_scaling, config.augmentation.scale]
    elif config.run_description == "scaling_bandstop":
        logger.debug("Selected scaling-bandstop")
        return [amplitude_scaling, config.augmentation.scale], [bandstop_filter, config.augmentation.bandstop]
    elif config.run_description == "scaling_gaussian":
        logger.debug("Selected scaling-gaussian")
        return [amplitude_scaling, config.augmentation.scale], [gaussian_noise, config.augmentation.gaussian_std]

    elif config.run_description == "shift_shift":
        logger.debug("Selected shift-shift")
        return [dc_shift, config.augmentation.shift], [None]
    elif config.run_description == "shift_mask":
        logger.debug("Selected shift-mask")
        return [dc_shift, config.augmentation.shift], [masking, config.augmentation.mask]
    elif config.run_description == "shift_permutation":
        logger.debug("Selected shift-permutation")
        return [dc_shift, config.augmentation.shift], [permutation, config.augmentation.max_seg]
    elif config.run_description == "shift_flip":
        logger.debug("Selected shift-flip")
        return [dc_shift, config.augmentation.shift], [horizontal_flip, 0]
    elif config.run_description == "shift_frequency":
        logger.debug("Selected shift-frequency")
        return [dc_shift, config.augmentation.shift], [frequency_scaling, config.augmentation.scale]
    elif config.run_description == "shift_bandstop":
        logger.debug("Selected shift-bandstop")
        return [dc_shift, config.augmentation.shift], [bandstop_filter, config.augmentation.bandstop]
    elif config.run_description == "shift_gaussian":
        logger.debug("Selected shift-gaussian")
        return [dc_shift, config.augmentation.shift], [gaussian_noise, config.augmentation.gaussian_std]
    elif config.run_description == "shift_scaling":
        logger.debug("Selected shift-scaling")
        return [dc_shift, config.augmentation.shift], [amplitude_scaling, config.augmentation.scale]

    elif config.run_description == "mask_mask":
        logger.debug("Selected mask-mask")
        return [masking, config.augmentation.mask], [None]
    elif config.run_description == "mask_permutation":
        logger.debug("Selected mask-permutation")
        return [masking, config.augmentation.mask], [permutation, config.augmentation.max_seg]
    elif config.run_description == "mask_flip":
        logger.debug("Selected mask-flip")
        return [masking, config.augmentation.mask], [horizontal_flip, 0]
    elif config.run_description == "mask_frequency":
        logger.debug("Selected mask-frequency")
        return [masking, config.augmentation.mask], [frequency_scaling, config.augmentation.scale]
    elif config.run_description == "mask_bandstop":
        logger.debug("Selected mask-bandstop")
        return [masking, config.augmentation.mask], [bandstop_filter, config.augmentation.bandstop]
    elif config.run_description == "mask_gaussian":
        logger.debug("Selected mask-gaussian")
        return [masking, config.augmentation.mask], [gaussian_noise, config.augmentation.gaussian_std]
    elif config.run_description == "mask_scaling":
        logger.debug("Selected mask-scaling")
        return [masking, config.augmentation.mask], [amplitude_scaling, config.augmentation.scale]
    elif config.run_description == "mask_shift":
        logger.debug("Selected mask-shift")
        return [masking, config.augmentation.mask], [dc_shift, config.augmentation.shift]

    elif config.run_description == "permutation_permutation":
        logger.debug("Selected permutation-permutation")
        return [permutation, config.augmentation.max_seg], [None]
    elif config.run_description == "permutation_flip":
        logger.debug("Selected permutation-flip")
        return [permutation, config.augmentation.max_seg], [horizontal_flip, 0]
    elif config.run_description == "permutation_frequency":
        logger.debug("Selected permutation-frequency")
        return [permutation, config.augmentation.max_seg], [frequency_scaling, config.augmentation.scale]
    elif config.run_description == "permutation_bandstop":
        logger.debug("Selected permutation-bandstop")
        return [permutation, config.augmentation.max_seg], [bandstop_filter, config.augmentation.bandstop]
    elif config.run_description == "permutation_gaussian":
        logger.debug("Selected permutation-gaussian")
        return [permutation, config.augmentation.max_seg], [gaussian_noise, config.augmentation.gaussian_std]
    elif config.run_description == "permutation_scaling":
        logger.debug("Selected permutation-scaling")
        return [permutation, config.augmentation.max_seg], [amplitude_scaling, config.augmentation.scale]
    elif config.run_description == "permutation_shift":
        logger.debug("Selected permutation-shift")
        return [permutation, config.augmentation.max_seg], [dc_shift, config.augmentation.shift]
    elif config.run_description == "permutation_mask":
        logger.debug("Selected permutation-mask")
        return [permutation, config.augmentation.max_seg], [masking, config.augmentation.mask]

    elif config.run_description == "flip_flip":
        logger.debug("Selected flip-flip")
        return [horizontal_flip, 0], [None]
    elif config.run_description == "flip_frequency":
        logger.debug("Selected flip-frequency")
        return [horizontal_flip, 0], [frequency_scaling, config.augmentation.scale]
    elif config.run_description == "flip_bandstop":
        logger.debug("Selected flip-bandstop")
        return [horizontal_flip, 0], [bandstop_filter, config.augmentation.bandstop]
    elif config.run_description == "flip_gaussian":
        logger.debug("Selected flip-gaussian")
        return [horizontal_flip, 0], [gaussian_noise, config.augmentation.gaussian_std]
    elif config.run_description == "flip_scaling":
        logger.debug("Selected flip-scaling")
        return [horizontal_flip, 0], [amplitude_scaling, config.augmentation.scale]
    elif config.run_description == "flip_shift":
        logger.debug("Selected flip-shift")
        return [horizontal_flip, 0], [dc_shift, config.augmentation.shift]
    elif config.run_description == "flip_mask":
        logger.debug("Selected flip-mask")
        return [horizontal_flip, 0], [masking, config.augmentation.mask]
    elif config.run_description == "flip_permutation":
        logger.debug("Selected flip-mask")
        return [horizontal_flip, 0], [permutation, config.augmentation.max_seg]

    elif config.run_description == "frequency_frequency":
        logger.debug("Selected frequency-frequency")
        return [frequency_scaling, config.augmentation.scale], [None]
    elif config.run_description == "frequency_bandstop":
        logger.debug("Selected frequency-bandstop")
        return [frequency_scaling, config.augmentation.scale], [bandstop_filter, config.augmentation.bandstop]
    elif config.run_description == "frequency_gaussian":
        logger.debug("Selected frequency-gaussian")
        return [frequency_scaling, config.augmentation.scale], [gaussian_noise, config.augmentation.gaussian_std]
    elif config.run_description == "frequency_scaling":
        logger.debug("Selected frequency-scaling")
        return [frequency_scaling, config.augmentation.scale], [amplitude_scaling, config.augmentation.scale]
    elif config.run_description == "frequency_shift":
        logger.debug("Selected frequency-shift")
        return [frequency_scaling, config.augmentation.scale], [dc_shift, config.augmentation.shift]
    elif config.run_description == "frequency_mask":
        logger.debug("Selected frequency-mask")
        return [frequency_scaling, config.augmentation.scale], [masking, config.augmentation.mask]
    elif config.run_description == "frequency_permutation":
        logger.debug("Selected frequency-mask")
        return [frequency_scaling, config.augmentation.scale], [permutation, config.augmentation.max_seg]
    elif config.run_description == "frequency_flip":
        logger.debug("Selected frequency-mask")
        return [frequency_scaling, config.augmentation.scale], [horizontal_flip, 0]

    elif config.run_description == "bandstop_bandstop":
        logger.debug("Selected bandstop-bandstop")
        return [bandstop_filter, config.augmentation.bandstop], [None]
    elif config.run_description == "bandstop_gaussian":
        logger.debug("Selected bandstop-gaussian")
        return [bandstop_filter, config.augmentation.bandstop], [gaussian_noise, config.augmentation.gaussian_std]
    elif config.run_description == "bandstop_scaling":
        logger.debug("Selected bandstop-scaling")
        return [bandstop_filter, config.augmentation.bandstop], [amplitude_scaling, config.augmentation.scale]
    elif config.run_description == "bandstop_shift":
        logger.debug("Selected bandstop-shift")
        return [bandstop_filter, config.augmentation.bandstop], [dc_shift, config.augmentation.shift]
    elif config.run_description == "bandstop_mask":
        logger.debug("Selected bandstop-mask")
        return [bandstop_filter, config.augmentation.bandstop], [masking, config.augmentation.mask]
    elif config.run_description == "bandstop_permutation":
        logger.debug("Selected bandstop-mask")
        return [bandstop_filter, config.augmentation.bandstop], [permutation, config.augmentation.max_seg]
    elif config.run_description == "bandstop_flip":
        logger.debug("Selected bandstop-mask")
        return [bandstop_filter, config.augmentation.bandstop], [horizontal_flip, 0]
    elif config.run_description == "bandstop_frequency":
        logger.debug("Selected bandstop-frequency")
        return [bandstop_filter, config.augmentation.bandstop], [frequency_scaling, config.augmentation.scale]
 
    else:
        raise AssertionError(f"Run description '{config.run_description}' does not match any existing augmentation sets")