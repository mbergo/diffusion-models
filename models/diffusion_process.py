import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class NoiseScheduler:
    """
    Manages the noise schedule for the diffusion process.
    
    This class implements a cosine-based noise schedule as described in the
    'Improved Denoising Diffusion Probabilistic Models' paper.
    """
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02):
        self.num_timesteps = num_timesteps
        
        # Create cosine-based noise schedule
        betas = torch.linspace(beta_start, beta_end, num_timesteps)
        alphas = 1 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        # Pre-compute values for sampling and loss calculation
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)
        
    def add_noise(self, x_0, t):
        """
        Adds noise to the input according to the forward process q(x_t|x_0).
        
        Args:
            x_0 (torch.Tensor): Original input image
            t (torch.Tensor): Timesteps
            
        Returns:
            tuple: (noised_image, noise) containing the noised image and the actual noise added
        """
        noise = torch.randn_like(x_0)
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        
        # Formula: x_t = √(αt)x_0 + √(1-αt)ε
        noised_image = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
        return noised_image, noise

class UNet(nn.Module):
    """
    U-Net architecture for noise prediction in the reverse process.
    
    This implementation includes time embeddings and residual connections
    for better gradient flow and temporal understanding.
    """
    def __init__(self, in_channels=1, time_emb_dim=256):
        super().__init__()
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionalEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.GELU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim)
        )
        
        # Encoder
        self.conv1 = self._make_conv_block(in_channels, 64)
        self.conv2 = self._make_conv_block(64, 128)
        self.conv3 = self._make_conv_block(128, 256)
        self.conv4 = self._make_conv_block(256, 512)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        
        # Decoder
        self.upconv4 = self._make_upconv_block(512 + 512, 256)
        self.upconv3 = self._make_upconv_block(256 + 256, 128)
        self.upconv2 = self._make_upconv_block(128 + 128, 64)
        self.upconv1 = self._make_upconv_block(64 + 64, 64)
        
        # Final layer
        self.final_conv = nn.Conv2d(64, in_channels, 1)
        
    def _make_conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
    
    def _make_upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    
    def forward(self, x, t):
        # Time embedding
        t_emb = self.time_mlp(t)
        
        # Encoder
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        
        # Add time information
        x4 = x4 * (t_emb[..., None, None] + 1)
        
        # Bottleneck
        x4 = self.bottleneck(x4)
        
        # Decoder with skip connections
        x = self.upconv4(torch.cat([x4, x4], dim=1))
        x = self.upconv3(torch.cat([x, x3], dim=1))
        x = self.upconv2(torch.cat([x, x2], dim=1))
        x = self.upconv1(torch.cat([x, x1], dim=1))
        
        return self.final_conv(x)

class SinusoidalPositionalEmbedding(nn.Module):
    """
    Sinusoidal positional embeddings for time steps.
    
    This embedding scheme helps the model understand the temporal nature
    of the diffusion process through periodic functions.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings
