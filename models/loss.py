import torch
import numpy as np

class NTXentLoss(torch.nn.Module):

    def __init__(self, device, batch_size, temperature, use_cosine_similarity):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size # 128
        self.temperature = temperature # 0.2
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity) # use_cosine_similarity=True
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size) 
        # >> (256, 256), filled with ones along the main diagonal, zeros elsewhere
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        # >> (256, 256) filled with ones along shifted diagonal, starting at (128, 1), going to (256, 128)
        # Ones begin to appear in cell (128, 128), and then go down (129, 129), (130, 130), ..., (256, 256)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        # Ones begin to appear in cell (0, 128), and then go down (1, 129), (2, 130), ..., (128, 256)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0)) # calls torch.nn.CosineSimilarity(dim=-1)
        # x.unsqueeze(1) >> (2N, 1, 32)
        # y.unsqueeze(0) >> (1, 2N, 32)
        # This approach ensures that we compute cosine similarity between ALL samples, positive as well as negative
        
        return v
    # >> (256, 256) == (2N, 2N)

    def forward(self, zis, zjs):
        # zis, zjs are both (128, 32) == (N, attention_num_out_features) where N is the number of batches
        representations = torch.cat([zjs, zis], dim=0)
        # (256, 32) == (2N, 32)

        # Calls _cosine_simililarity(..., dim=-1)
        similarity_matrix = self.similarity_function(representations, representations)
        # >> (256, 256) == (2N, 2N)

        # Filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        # >> (128)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        # >> (128)

        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)
        # >> (256, 1)
        # Makes sense as each input sample receives one augmented equivalent, we have 2N positive sampels

        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)
        # >> (256, 254)

        logits = torch.cat((positives, negatives), dim=1)
        # >> (256, 255)
        logits /= self.temperature

        labels = torch.zeros(2 * self.batch_size).to(self.device).long()
        # >> (256) zeros

        loss = self.criterion(logits, labels) # calls cross entropy

        return loss / (2 * self.batch_size)
