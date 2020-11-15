#%% lib
import torch.nn as nn
import torch
import numpy as np

class VAE(nn.Module):
    def __init__(self, input_size):
        super(VAE, self).__init__()

        self.E_hidden = nn.Linear(input_size,400)
        self.E_mu = nn.Linear(400,200)
        self.E_logvar = nn.Linear(400, 200)
        self.D_hidden = nn.Linear(200,400)
        self.D_mu = nn.Linear(400, input_size)
        self.D_logvar = nn.Linear(400, input_size)
        self.relu = nn.ReLU(inplace=True)

    def encode(self,x):
        x = self.relu(self.E_hidden(x))
        return self.E_mu(x) , self.E_logvar(x)

    def decode(self,z):
        z = self.relu(self.D_hidden(z))
        return self.D_mu(z) , self.D_logvar(z)

    def forward(self,x):
        x = x.view(x.size(0),-1)
        mu, logvar = self.encode(x)
        z_sample = self.sample_from_gaussian(mu,logvar)
        x_mean, x_logvar = self.decode(z_sample)
        x_mean = x_mean.view(x_mean.size(0),1,28,28)
        x_logvar = x_logvar.view(x_logvar.size(0),1,28,28)

        return x_mean, x_logvar, mu, logvar

    def reconstruction_probability(self,x):
        x = x.view(x.size(0), -1)
        mu, logvar = self.encode(x)
        z_samples = self.samples_from_gaussian(mu,logvar)
        likelyhoods = torch.zeros((10,x.size(0)))
        for ind , z_sample in enumerate(z_samples):
            x_mean, x_logvar = self.decode(z_sample)
            likelyhood_per_pixel = torch.exp(-0.5 * torch.square(x - x_mean) * torch.exp(-x_logvar))/(torch.sqrt(2*np.pi*torch.exp(x_logvar)))
            likelyhood = torch.mean(likelyhood_per_pixel,axis=[1])
            likelyhoods[ind] = likelyhood
        return torch.mean(likelyhoods,axis=0)

    @staticmethod
    def sample_from_gaussian(mean,logvar):
        eps = torch.randn_like(mean)
        return eps.mul(torch.exp(logvar/2)).add_(mean)

    @staticmethod
    def samples_from_gaussian(mean, logvar, T=10):
        z_samples = []
        for _ in range(T):
            eps = torch.randn_like(mean)
            z_samples.append(eps.mul(torch.exp(logvar/2)).add_(mean))
        return z_samples


class AE(nn.Module):
    def __init__(self, input_size):
        super(AE, self).__init__()

        self.E_hidden1 = nn.Linear(input_size,400)
        self.E_hidden2 = nn.Linear(400,200)
        self.relu = nn.ReLU(inplace=True)
        self.D_hidden1 = nn.Linear(200, 400)
        self.D_hidden2 = nn.Linear(400, input_size)

    def forward(self,x):
        x = x.view(x.size(0),-1)
        x = self.relu(self.E_hidden1(x))
        x = self.relu(self.E_hidden2(x))
        x = self.relu(self.D_hidden1(x))
        x = self.relu(self.D_hidden2(x))
        x = x.view(x.size(0),1,28,28)

        return x

