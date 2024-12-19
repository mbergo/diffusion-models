import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from forward_reverse_process import NoiseScheduler, UNet
import matplotlib.pyplot as plt

class DiffusionTrainer:
    """
    Trainer class for the diffusion model.
    
    This class handles the training loop, loss calculation, and sampling
    from the model during training.
    """
    def __init__(self, model, noise_scheduler, device='cuda'):
        self.model = model.to(device)
        self.noise_scheduler = noise_scheduler
        self.device = device
        self.optimizer = optim.AdamW(model.parameters(), lr=2e-4)
        
    def train_step(self, batch):
        """
        Performs a single training step on a batch of data.
        
        Args:
            batch (torch.Tensor): Batch of images
            
        Returns:
            float: Loss value for this batch
        """
        self.optimizer.zero_grad()
        
        # Sample random timesteps
        batch_size = batch.shape[0]
        timesteps = torch.randint(0, self.noise_scheduler.num_timesteps, (batch_size,),
                                device=self.device)
        
        # Add noise to images
        noised_images, noise = self.noise_scheduler.add_noise(batch, timesteps)
        
        # Predict noise
        predicted_noise = self.model(noised_images, timesteps)
        
        # Calculate loss
        loss = F.mse_loss(predicted_noise, noise)
        
        # Backpropagate
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def sample(self, num_images=1, image_size=28, channels=1):
        """
        Samples new images from the trained model.
        
        Args:
            num_images (int): Number of images to generate
            image_size (int): Size of the images to generate
            channels (int): Number of channels in the images
            
        Returns:
            torch.Tensor: Generated images
        """
        self.model.eval()
        with torch.no_grad():
            # Start from pure noise
            x = torch.randn(num_images, channels, image_size, image_size).to(self.device)
            
            # Gradually denoise
            for t in reversed(range(self.noise_scheduler.num_timesteps)):
                timesteps = torch.full((num_images,), t, device=self.device)
                predicted_noise = self.model(x, timesteps)
                
                # Update sample using predicted noise
                alpha_t = self.noise_scheduler.alphas_cumprod[t]
                alpha_t_prev = self.noise_scheduler.alphas_cumprod[t-1] if t > 0 else torch.tensor(1.0)
                
                # Equation for reverse process step
                beta_t = 1 - alpha_t / alpha_t_prev
                x = (1 / torch.sqrt(1 - beta_t)) * (x - (beta_t / torch.sqrt(1 - alpha_t)) * predicted_noise)
                
                if t > 0:
                    noise = torch.randn_like(x)
                    x = x + torch.sqrt(beta_t) * noise
        
        self.model.train()
        return x

def main():
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    dataset = datasets.MNIST('./data', train=True, transform=transform, download=True)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    
    # Initialize models
    model = UNet(in_channels=1)
    noise_scheduler = NoiseScheduler()
    trainer = DiffusionTrainer(model, noise_scheduler, device)
    
    # Training loop
    num_epochs = 50
    for epoch in range(num_epochs):
        total_loss = 0
        for batch, _ in dataloader:
            batch = batch.to(device)
            loss = trainer.train_step(batch)
            total_loss += loss
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
        
        # Generate and save sample images
        if (epoch + 1) % 5 == 0:
            samples = trainer.sample(num_images=16)
            samples = (samples + 1) / 2  # Denormalize
            
            plt.figure(figsize=(10, 10))
            for i in range(16):
                plt.subplot(4, 4, i+1)
                plt.imshow(samples[i, 0].cpu(), cmap='gray')
                plt.axis('off')
            plt.savefig(f'samples_epoch_{epoch+1}.png')
            plt.close()

if __name__ == '__main__':
    main()
