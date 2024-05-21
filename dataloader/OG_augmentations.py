import numpy as np
import torch
import random # had to import this for new code in permutations


def DataTransform(sample, config):

    aug = jitter(scaling(sample, config.augmentation.jitter_scale_ratio), config.augmentation.jitter_ratio)
    aug = jitter(permutation(sample, max_segments=config.augmentation.max_seg, seg_mode="not_random"), config.augmentation.jitter_ratio)

    return sample


def jitter(x, sigma=0.8):
    # https://arxiv.org/pdf/1706.00527.pdf
    stds = []
    for i in range(x.shape[0]):
        # Select a random std for the batch
        std = np.random.uniform(low=0, high=sigma)
        stds.append(np.random.normal(loc=0, scale=std, size=(1, x.shape[1], x.shape[2])))
    augs = np.concatenate(stds, axis=0)
    return x + augs
    # return x + np.random.normal(loc=0., scale=sigma, size=x.shape)


def scaling(x, sigma=1.1):
    # https://arxiv.org/pdf/1706.00527.pdf
    factor = np.random.normal(loc=2., scale=sigma, size=(x.shape[0], x.shape[2]))
    ai = []
    for i in range(x.shape[1]):
        xi = x[:, i, :]
        ai.append(np.multiply(xi, factor[:, :])[:, np.newaxis, :])
    return np.concatenate((ai), axis=1)


def permutation(x, max_segments=5, seg_mode="random"):
    # x contains samples of 30-second recordings in the shape: (training_size, 1, 3000)
    # max_segments is M, i.e. the maximum number of splits --> Set to 12 for sleepEDF data
    orig_steps = np.arange(x.shape[2]) 
    # --> Returns timesteps, in shape (3000,) which contains the values 0 to 2999 in that order
    num_segs = np.random.randint(1, max_segments, size=(x.shape[0]))
    # --> Returns a np.array that contains the number of segments for each sample (30-second epoch), a number between 1 and 12
        
    ret = np.zeros_like(x)

    for i, pat in enumerate(x):
        if num_segs[i] > 1:
            if seg_mode == "random":
                split_points = np.random.choice(x.shape[2] - 2, num_segs[i] - 1, replace=False)
                split_points.sort()
                splits = np.split(orig_steps, split_points)
            
            else:
                splits = np.array_split(orig_steps, num_segs[i])
                random.shuffle(splits)
                splits = np.concatenate(splits)
                ret[i] = pat[0, splits]
        else:
            ret[i] = pat

    return torch.from_numpy(ret)
