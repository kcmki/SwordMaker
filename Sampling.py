
from typing import Dict, Tuple
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
from diffusion_utilities import *
from ContextUnet import ContextUnet


class Sampler():
    def __init__(self):
        # hyperparameters
        # diffusion hyperparameters
        self.timesteps = 500
        beta1 = 1e-4
        beta2 = 0.02

        # network hyperparameters
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else torch.device('cpu'))
        n_feat = 64 # 64 hidden dimension feature
        n_cfeat = 5 # context vector is of size 5
        self.height = 16 # 16x16 image
        save_dir = './weights/'

        # construct DDPM noise schedule
        self.b_t = (beta2 - beta1) * torch.linspace(0, 1, self.timesteps + 1, device=self.device) + beta1
        self.a_t = 1 - self.b_t
        self.ab_t = torch.cumsum(self.a_t.log(), dim=0).exp()    
        self.ab_t[0] = 1


        self.nn_model = ContextUnet(in_channels=3, n_feat=n_feat, n_cfeat=n_cfeat, height=self.height).to(self.device)
        self.nn_model.load_state_dict(torch.load(f"{save_dir}/model_31.pth", map_location=self.device))
        print("Loaded in Model")


    # helper function; removes the predicted noise (but adds some noise back in to avoid collapse)
    def denoise_add_noise(self,x, t, pred_noise, z=None):
        if z is None:
            z = torch.randn_like(x)
        noise = self.b_t.sqrt()[t] * z
        mean = (x - pred_noise * ((1 - self.a_t[t]) / (1 - self.ab_t[t]).sqrt())) / self.a_t[t].sqrt()
        return mean + noise


    # sample using standard algorithm
    @torch.no_grad()
    def sample_ddpm(self,n_sample, save_rate=20):
        # x_T ~ N(0, 1), sample initial noise
        samples = torch.randn(n_sample, 3, self.height, self.height).to(self.device)  

        # array to keep track of generated steps for plotting
        intermediate = [] 
        for i in range(self.timesteps, 0, -1):
            print(f'sampling timestep {i:3d}', end='\r')

            # reshape time tensor
            t = torch.tensor([i / self.timesteps])[:, None, None, None].to(self.device)

            # sample some random noise to inject back in. For i = 1, don't add back in noise
            z = torch.randn_like(samples) if i > 1 else 0

            eps = self.nn_model(samples, t)    # predict noise e_(x_t,t)
            samples = self.denoise_add_noise(samples, i, eps, z)
            if i % save_rate ==0 or i==self.timesteps or i<8:
                intermediate.append(samples.detach().cpu().numpy())

        intermediate = np.stack(intermediate)
        return samples, intermediate