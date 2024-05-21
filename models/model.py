from torch import nn


class base_Model(nn.Module):
    def __init__(self, configs):
        super(base_Model, self).__init__()

        """
        Hyperparameter options:
            congifs.input_channels = 1
            configs.kernel_size = 25
            configs.stride = 3
            configs.dropout = 0.35
            configs.final_out_channels = 128
            configs.features_len = 127
            configs.num_classes = 5

        Kernel/filters are matrices that slide over the input image to extract features.
        The stride determines how large those slides will be, stride=1 means the kernel window moves one pixel at a time.
        If necessary, padding modifies the input such that the kernel window fits, e.g. by adding 0's.

        """
        
        # Applies 32 kernels of size 25,
        # Normalizes each feature map,
        # Applies ReLU,
        # Halves dimension of features maps,
        # Dropout of 0.35. 
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(configs.input_channels, 32, kernel_size=configs.kernel_size,
                      stride=configs.stride, bias=False, padding=(configs.kernel_size//2)),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(configs.dropout)
        )

        # Applies 64 kernels of size 8,
        # Normalizes each kernel,
        # Applies ReLU,
        # Reduces feature map by half.
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        # Applies 128 kernels of size 8,
        # Normalizes each map,
        # Applies ReLU,
        # Reduces feature maps by half.
        self.conv_block3 = nn.Sequential(
            nn.Conv1d(64, configs.final_out_channels, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(configs.final_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

        # Only to produce class predictions for model evaluation
        model_output_dim = configs.features_len
        self.logits = nn.Linear(model_output_dim * configs.final_out_channels, configs.num_classes)

    def forward(self, x_in):
        x = self.conv_block1(x_in)
        x = self.conv_block2(x)
        x = self.conv_block3(x)

        # Logits returns class predictions
        x_flat = x.reshape(x.shape[0], -1)
        logits = self.logits(x_flat)
        return logits, x