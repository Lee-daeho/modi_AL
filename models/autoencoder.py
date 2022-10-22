import torch
import torch.nn as nn
import torch.nn.functional as F

class AutoEncoder(nn.Module):
    """
    Build a SimSiam model.
    """
    def __init__(self, base_encoder, dim=512, pred_dim=256):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super(AutoEncoder, self).__init__()

        # create the encoder
        # num_classes is the output fc dimension, zero-initialize last BNs
        self.encoder = base_encoder

        # build a 3-layer projector
        prev_dim = self.encoder.fc.weight.shape[1]
        # self.encoder.fc = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
        #                                 nn.BatchNorm1d(prev_dim),
        #                                 nn.ReLU(inplace=True), # first layer
        #                                 nn.Linear(prev_dim, prev_dim, bias=False),
        #                                 nn.BatchNorm1d(prev_dim),
        #                                 nn.ReLU(inplace=True), # second layer 
        #                                 nn.Linear(prev_dim, dim, bias=False),
        #                                 nn.BatchNorm1d(dim, affine=False)) # output layer
        self.encoder.fc = nn.Linear(prev_dim, prev_dim)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, padding=1, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, padding=1, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, padding=1, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 3, padding=1, stride=1)
        )

    def forward(self, x):

        _, out, _ = self.encoder(x)
        out = out.view(out.size(0), 512, 1, 1)
        out = F.interpolate(out, scale_factor=4)
        out = self.decoder(out)

        return out