from torch import nn
import torch


### Convolutional Autoencoder network. Note that:
#   torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding) – applies convolution
#   torch.nn.MaxPool2d(kernel_size, stride, padding) – applies max pooling
class AutoencoderNet(nn.Module):
    
    def __init__(self, encoded_space_dim):
        
        super(AutoencoderNet, self).__init__()
        
        ### Encoder 
        self.encoder_cnn = nn.Sequential(
            
            # input size (1,28,28) -> output size (16,10,10)
            nn.Conv2d(1, 16, 3, stride=3, padding=1),
            # apply activation function
            nn.ReLU(True),
            # input size (16, 10, 10) -> output size (32, 5, 5)
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            # apply activation function
            nn.ReLU(True),
            # input size (32, 5, 5) -> output size (32, 4, 4)
            nn.MaxPool2d(2, stride=1),
        )
        
        ## fully connected linear layers
        self.encoder_lin = nn.Sequential(
            
            # 512 input features, 64 output features 
            nn.Linear(32 * 4 * 4, 64),
            # apply activation function
            nn.ReLU(True),
            # 64 input features, varying latent space output size 
            nn.Linear(64, encoded_space_dim)
        )
        
        ### Decoder
        self.decoder_lin = nn.Sequential(
            # varying latent space input size, 64 output size 
            nn.Linear(encoded_space_dim, 64),
            # apply activation function
            nn.ReLU(True),
            # 64 input features, 512 output features
            nn.Linear(64, 32 * 4 * 4),
            # apply activation function
            nn.ReLU(True)
        )
        # reconstruct the data trough transpose layers
        self.decoder_conv = nn.Sequential(    
            # input size (32, 4, 4) -> output size (16, 8, 8)
            nn.ConvTranspose2d(32, 16, 3, stride=2,  padding=1, output_padding=1),
            # apply activation function
            nn.ReLU(True),
            # input size (16, 8, 8) -> output size (8, 14, 14)
            nn.ConvTranspose2d(16, 8, 3, stride=2,  padding=2, output_padding=1),
             # apply activation function
            nn.ReLU(True),
            # input size (8, 14, 14) -> output size (1, 28, 28)
            nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1, output_padding=1)
        )

    # forward propagation
    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

    def encode(self, x):
        # Apply convolutions
        x = self.encoder_cnn(x)
        # Flatten
        x = x.view([x.size(0), -1])
        # Apply linear layers
        x = self.encoder_lin(x)
        return x
    
    def decode(self, x):
        # Apply linear layers
        x = self.decoder_lin(x)
        # Reshape
        x = x.view([-1, 32, 4, 4])
        # Apply transposed convolutions
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)
        return x