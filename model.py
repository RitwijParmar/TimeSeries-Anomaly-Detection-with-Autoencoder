import torch.nn as nn

class Autoencoder(nn.Module):
    """
    A simple Autoencoder architecture with Linear layers.
    """
    def __init__(self, sequence_length, embedding_dim=64):
        super(Autoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(sequence_length, 128),
            nn.ReLU(True),
            nn.Linear(128, embedding_dim),
            nn.ReLU(True)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, sequence_length),
            nn.Tanh() # Using Tanh since input is scaled between 0-1 (can also use Sigmoid)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x