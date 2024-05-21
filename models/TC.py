import torch
import torch.nn as nn
import numpy as np
from .attention import Seq_Transformer


class TC(nn.Module):
    def __init__(self, configs, device):
        super(TC, self).__init__()
        self.num_channels = configs.final_out_channels # 128
        self.timestep = configs.TC.timesteps # 50
        self.Wk = nn.ModuleList([nn.Linear(configs.TC.hidden_dim, self.num_channels) for i in range(self.timestep)]) # configs.TC.hidden_dim = 64
        self.lsoftmax = nn.LogSoftmax(dim=1) 
        self.device = device
        self.normalize_preds = configs.TC.normalize_preds
        
        self.projection_head = nn.Sequential(
            nn.Linear(configs.TC.hidden_dim, configs.final_out_channels // 2),
            nn.BatchNorm1d(configs.final_out_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(configs.final_out_channels // 2, configs.final_out_channels // 4),
        )

        self.seq_transformer = Seq_Transformer(patch_size=self.num_channels, dim=configs.TC.hidden_dim, depth=4, heads=4, mlp_dim=64)

    def forward(self, features_source, features_aug):
        # Obtain latent space augmentations
        # features are (batch_size=128, out_channels=128, feature_len=127)
        z_aug = features_aug
        z_aug = z_aug.transpose(1, 2) # (batch_size, feature_len, out_channels)
        z_source = features_source
        z_source = z_source.transpose(1, 2) # (batch_size, feature_len, out_channels)
        
        seq_len = z_aug.shape[2] # out_channels=128 
        batch = z_aug.shape[0] # batch_size=128
        
        # randomly select the number of features to construct the context vectors
        t_samples = torch.randint(seq_len - (self.timestep+2), size=(1,)).long().to(self.device)
        # >> (1), contains a single random number
        
        forward_seq_aug = z_aug[:, :t_samples + 1, :]
        forward_seq_source = z_source[:, :t_samples + 1, :]
        # Takes all batches, their t_samples+1 first features, and all channels
        # >> (128, t_samples+1, 128) contains the features that will be used to create context vector 

        # Holds source features that will be predicted
        encode_samples = torch.empty((self.timestep, batch, self.num_channels)).float().to(self.device)

        for i in np.arange(1, self.timestep + 1):
            encode_samples[i - 1] = z_source[:, t_samples + i, :].view(batch, self.num_channels)
            # >> (50, 128, 128) contains 50 features to be predicted 
                
        # Create the contextual vectors using attention module
        c_aug = self.seq_transformer(forward_seq_aug)
        c_source = self.seq_transformer(forward_seq_source)
        # >> (128, 64) each batch is encoded with a class token of 64 features

        # Make predictions
        pred = torch.empty((self.timestep, batch, self.num_channels)).float().to(self.device)
        # >> (50, 128, 128) same shape as encode_samples but holds predictions
        for i in np.arange(0, self.timestep):
            linear = self.Wk[i]   # (64, 128) log-bilinear model
            pred[i] = linear(c_aug)
        
        nce = 0
        for i in np.arange(0, self.timestep):
            total = torch.mm(encode_samples[i], torch.transpose(pred[i], 0, 1))
            # >> (128, 128), (Batch, Batch)
            nce += torch.sum(torch.diag(self.lsoftmax(total)))
        
        # Average over batches and timesteps 
        nce /= -1 * batch * self.timestep
        return nce, self.projection_head(c_source), self.projection_head(c_aug)
        # self.projection_head(.) is the nonlinear projection head before the NTXENT contrastive loss
