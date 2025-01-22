from typing import Iterator
import torch
from torch import nn
from torch.nn.parameter import Parameter
from torchvision.models import resnet18

class ResNetEncoder(nn.Module):
    def __init__(self, in_channels, latent_dim):
        super().__init__()
        resnet = resnet18(pretrained=True)
        if in_channels != 3:
            resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        self.fc_mu = nn.Linear(resnet.fc.in_features, latent_dim)
        self.fc_logvar = nn.Linear(resnet.fc.in_features, latent_dim)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1) # Flatten the tensor
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

class ResNetVAE(nn.Module):
    def __init__(self, in_channels=1, latent_dim=512, img_size=256):
        super().__init__()
        self.encoder = ResNetEncoder(in_channels, latent_dim)

        s = img_size // 8

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, s*s*64),  # First fully connected layer
            nn.ReLU(),
        )
        self.reshape = nn.Unflatten(1, (64, s, s))  # Reshape layer
        self.conv_layers = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # Convolutional layer 1
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),  # Convolutional layer 2
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1),  # Convolutional layer 3
            nn.Sigmoid()  # To get pixel values in [0, 1]
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        z = self.decoder(z)
        z = self.reshape(z)
        z = self.conv_layers(z)
        return z, mu, logvar

    def anomaly_score(self, x, device):
        self.eval()
        with torch.no_grad():
            x = x.to(device)
            recon_x, mu, logvar = self(x)
            BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='none').sum((1,2,3))
            KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), 1)
            return BCE + KLD

def train_step(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for x in train_loader:
        x = x.to(device)
        optimizer.zero_grad()
        recon_x, mu, logvar = model(x)
        # Compute reconstruction loss and KL divergence
        BCE = criterion(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = BCE + KLD
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(train_loader.dataset), None

def val_step(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for x in val_loader:
            x = x.to(device)
            recon_x, mu, logvar = model(x)
            BCE = criterion(recon_x, x, reduction='sum')
            KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = BCE + KLD
            running_loss += loss.item()
    return running_loss / len(val_loader.dataset), None

def train(model, train_loader, val_loader, epochs, lr, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        train_loss = train_step(model, train_loader, optimizer, device)
        val_loss = val_step(model, val_loader, device)
        print(f"Epoch {epoch+1}, Train Loss: {train_loss}, Val Loss: {val_loss}")


# import the necessary libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

# define a class for sampling
# this class will be used in the encoder for sampling in the latent space
class Sampling(nn.Module):
    def forward(self, z_mean, z_log_var):
        # get the shape of the tensor for the mean and log variance
        batch, dim = z_mean.shape
        # generate a normal random tensor (epsilon) with the same shape as z_mean
        # this tensor will be used for reparameterization trick
        epsilon = Normal(0, 1).sample((batch, dim)).to(z_mean.device)
        # apply the reparameterization trick to generate the samples in the
        # latent space
        return z_mean + torch.exp(0.5 * z_log_var) * epsilon
    
# define the encoder
class Encoder(nn.Module):
    def __init__(self, image_size, embedding_dim):
        super(Encoder, self).__init__()
        # define the convolutional layers for downsampling and feature
        # extraction
        self.conv1 = nn.Conv2d(1, 32, 7, stride=1, padding='same')
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, 5, stride=1, padding='same')
        self.pool2 = nn.MaxPool2d(2, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=1, padding='same')
        self.pool3 = nn.MaxPool2d(2, stride=2)
        self.conv4 = nn.Conv2d(128, 256, 3, stride=1, padding='same')
        self.pool4 = nn.MaxPool2d(2, stride=2)
        # define a flatten layer to flatten the tensor before feeding it into
        # the fully connected layer
        self.flatten = nn.Flatten()
        # define fully connected layers to transform the tensor into the desired
        # embedding dimensions
        o, r = 256, 16
        self.fc_mean = nn.Linear(
            o * (image_size // r) * (image_size // r), embedding_dim
        )
        self.fc_log_var = nn.Linear(
            o * (image_size // r) * (image_size // r), embedding_dim
        )
        # initialize the sampling layer
        self.sampling = Sampling()
    def forward(self, x):
        # apply convolutional layers with relu activation function
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = F.relu(self.conv4(x))
        x = self.pool4(x)
        # flatten the tensor
        x = self.flatten(x)
        # get the mean and log variance of the latent space distribution
        z_mean = self.fc_mean(x)
        z_log_var = self.fc_log_var(x)
        # sample a latent vector using the reparameterization trick
        z = self.sampling(z_mean, z_log_var)
        return z_mean, z_log_var, z

# define the decoder
class Decoder(nn.Module):
    def __init__(self, embedding_dim, shape_before_flattening):
        super(Decoder, self).__init__()
        # define a fully connected layer to transform the latent vector back to
        # the shape before flattening
        self.fc = nn.Linear(
            embedding_dim,
            shape_before_flattening[0]
            * shape_before_flattening[1]
            * shape_before_flattening[2],
        )
        # define a reshape function to reshape the tensor back to its original
        # shape
        self.reshape = nn.Unflatten(1, shape_before_flattening)
        # define the transposed convolutional layers for the decoder to upsample
        # and generate the reconstructed image
        self.deconv4 = nn.ConvTranspose2d(
            256, 128, 3, stride=2, padding=1, output_padding=1
        )
        self.deconv1 = nn.ConvTranspose2d(
            128, 64, 3, stride=2, padding=1, output_padding=1
        )
        self.deconv2 = nn.ConvTranspose2d(
            64, 32, 3, stride=2, padding=1, output_padding=1
        )
        self.deconv3 = nn.ConvTranspose2d(
            32, 1, 3, stride=2, padding=1, output_padding=1
        )
    
    def forward(self, x):
        # pass the latent vector through the fully connected layer
        x = self.fc(x)
        # reshape the tensor
        x = self.reshape(x)
        # apply transposed convolutional layers with relu activation function
        x = F.relu(self.deconv4(x))
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        # apply the final transposed convolutional layer with a sigmoid
        # activation to generate the final output
        x = torch.sigmoid(self.deconv3(x))
        return x

# define the vae class
class VAESmall(nn.Module):
    def __init__(self, in_channels=1, latent_dim=512, image_size=256):
        super(VAESmall, self).__init__()
        shape_before_flattening = (256, image_size // 16, image_size // 16)
        encoder = Encoder(image_size, latent_dim)
        decoder = Decoder(latent_dim, shape_before_flattening)
        # initialize the encoder and decoder
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        # pass the input through the encoder to get the latent vector
        z_mean, z_log_var, z = self.encoder(x)
        # pass the latent vector through the decoder to get the reconstructed
        # image
        reconstruction = self.decoder(z)
        # return the mean, log variance and the reconstructed image
        return reconstruction, z_mean, z_log_var
    
    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        return list(self.encoder.parameters(recurse=recurse)) + list(self.decoder.parameters(recurse=recurse))
    
class ConvVAE(nn.Module):
    def __init__(self, in_channels=1, latent_dim=100, img_size=256):
        super(ConvVAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),  # 128x128
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 64x64
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 32x32
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 16x16
            nn.ReLU(),
            nn.Flatten(),  # Flatten the tensor
        )

        s = img_size // 16

        self.fc_mu = nn.Linear(256*s*s, latent_dim)
        self.fc_logvar = nn.Linear(256*s*s, latent_dim)

        self.fc_decode = nn.Linear(latent_dim, 256*s*s)
        self.unflatten = nn.Unflatten(1, (256, s, s))

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 64x64
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 128x128
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),  # 256x256
            nn.Sigmoid(),
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        z = self.fc_decode(z)
        z = self.unflatten(z)
        return self.decoder(z), mu, logvar

class ConvVAE2(nn.Module):
    def __init__(self, in_channels=1, latent_dim=100, img_size=256):
        super(ConvVAE2, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),  # 256x256
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 128x128
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 128x128
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 64x64
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 64x64
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 32x32
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # 32x32
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 16x16
            nn.Flatten(),  # Flatten the tensor
        )

        s = img_size // 16

        self.fc_mu = nn.Linear(256*s*s, latent_dim)
        self.fc_logvar = nn.Linear(256*s*s, latent_dim)

        self.fc_decode = nn.Linear(latent_dim, 256*s*s)
        self.unflatten = nn.Unflatten(1, (256, s, s))

        self.decoder = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),  # 16x16
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),  # 32x32
            nn.Conv2d(128, 64, kernel_size=3, padding=1),  # 32x32
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),  # 64x64
            nn.Conv2d(64, 32, kernel_size=3, padding=1),  # 64x64
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),  # 128x128
            nn.Conv2d(32, 1, kernel_size=3, padding=1),  # 128x128
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),  # 256x256
            nn.ReLU()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        z = self.fc_decode(z)
        z = self.unflatten(z)
        return self.decoder(z), mu, logvar

