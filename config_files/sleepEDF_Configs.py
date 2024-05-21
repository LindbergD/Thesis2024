class Config(object):
    def __init__(self):
        # model configs
        self.input_channels = 1
        self.final_out_channels = 128
        self.num_classes = 5
        self.dropout = 0.35

        self.kernel_size = 25
        self.stride = 3
        self.features_len = 127

        # training configs
        self.num_epoch = 100
        self.run_description = None

        # optimizer parameters
        self.optimizer = 'adam'
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.lr = 3e-4

        # data parameters
        self.drop_last = True
        self.batch_size = 128
        self.bandstop_filter = False

        self.Context_Cont = Context_Cont_configs()
        self.TC = TC()
        self.augmentation = augmentations()

class augmentations(object):
    def __init__(self):
        self.gaussian_std = 1       # Gaussian noise standard deviation
        self.shift = [-1, 1]        # DC-shift [min, max]
        self.mask = 300             # Masking max timesteps
        self.scale = [0.8, 1.2]     # Amplitude & frequency scaling [min, max] 
        self.max_seg = 12           # Permutation max segments
        self.bandstop = 31          # Lowpass filter cutoff in hz

class Context_Cont_configs(object):
    def __init__(self):
        self.temperature = 0.2
        self.use_cosine_similarity = True


class TC(object):
    def __init__(self):
        self.hidden_dim = 64
        self.timesteps = 50
        self.normalize_preds = False