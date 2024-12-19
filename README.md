# Educational Diffusion Model Implementation

This repository contains an educational implementation of a Diffusion Model using PyTorch. The code is designed to be clear and well-documented, making it ideal for learning about diffusion models and their implementation.

## 📚 Background

Diffusion Models are a class of generative models that learn to generate data by gradually denoising a pure noise input. The process involves two main steps:

1. **Forward Process (Diffusion)**: Gradually adds noise to the data according to a fixed schedule
2. **Reverse Process (Denoising)**: Learns to gradually remove noise to generate new data samples

## 🏗️ Project Structure

- `forward_reverse_process.py`: Contains the core implementations of:
  - `NoiseScheduler`: Manages the noise schedule for both forward and reverse processes
  - `UNet`: Neural network architecture for noise prediction
  - `SinusoidalPositionalEmbedding`: Time embedding module
- `train_diffusion.py`: Contains the training logic and sampling procedures
- `requirements.txt`: Lists all necessary Python packages

## 🚀 Getting Started

1. **Setup Environment**
```bash
# Create and activate virtual environment
python -m venv diffusion_env
source diffusion_env/bin/activate  # On Windows: diffusion_env\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

2. **Run Training**
```bash
python train_diffusion.py
```

## 🔧 Implementation Details

### Forward Process

The forward process is implemented in the `NoiseScheduler` class, which:
- Uses a cosine-based noise schedule
- Implements q(x_t|x_0) for adding noise
- Pre-computes important values for efficiency

### Reverse Process

The reverse process uses a U-Net architecture that:
- Takes both the noisy image and timestep as input
- Uses skip connections for better gradient flow
- Incorporates time embeddings using sinusoidal positional encoding

### Training

The training process:
1. Samples random timesteps for each batch
2. Adds noise according to the schedule
3. Predicts the noise using the U-Net
4. Optimizes using MSE loss between predicted and actual noise

## 📈 Results

The model will save generated samples every 5 epochs in the format `samples_epoch_N.png`. You can observe how the generation quality improves over time.

## 🎓 Learning Resources

To deepen your understanding of diffusion models, consider reading:
1. "Denoising Diffusion Probabilistic Models" (Ho et al., 2020)
2. "Improved Denoising Diffusion Probabilistic Models" (Nichol & Dhariwal, 2021)
3. "Understanding Diffusion Models: A Unified Perspective" (Yang, 2022)

## 