import torch
import torch.nn as nn

def weights_init(m):
    """
    Custom weights initialization for Generator and Discriminator/Critic models.
    As recommended in the DCGAN paper:
    - Convolutional and Transposed Convolutional weights are initialized from a normal distribution with mean=0, std=0.02.
    - BatchNorm weights are initialized from a normal distribution with mean=1.0, std=0.02, and bias=0.
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
    """
    Generator Network for 64x64 RGB Face Image Generation.
    Maps a latent vector z in R^128 to a 64x64x3 RGB image.
    Uses a convolutional/transposed-convolutional architecture.
    """
    def __init__(self, latent_dim=128, features_g=64):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # Layer 1: Input latent vector z (batch, latent_dim, 1, 1) -> (batch, features_g * 8, 4, 4)
            nn.ConvTranspose2d(latent_dim, features_g * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(features_g * 8),
            nn.ReLU(True),
            
            # Layer 2: (batch, features_g * 8, 4, 4) -> (batch, features_g * 4, 8, 8)
            nn.ConvTranspose2d(features_g * 8, features_g * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(features_g * 4),
            nn.ReLU(True),
            
            # Layer 3: (batch, features_g * 4, 8, 8) -> (batch, features_g * 2, 16, 16)
            nn.ConvTranspose2d(features_g * 4, features_g * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(features_g * 2),
            nn.ReLU(True),
            
            # Layer 4: (batch, features_g * 2, 16, 16) -> (batch, features_g, 32, 32)
            nn.ConvTranspose2d(features_g * 2, features_g, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(features_g),
            nn.ReLU(True),
            
            # Layer 5: (batch, features_g, 32, 32) -> (batch, 3, 64, 64)
            # Output values are squashed to [-1, 1] using Tanh
            nn.ConvTranspose2d(features_g, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        # If input z is 2D (batch_size, latent_dim), unsqueeze it to 4D (batch_size, latent_dim, 1, 1)
        if z.dim() == 2:
            z = z.unsqueeze(-1).unsqueeze(-1)
        return self.main(z)

class Discriminator(nn.Module):
    """
    Discriminator/Critic Network for 64x64 RGB images.
    Maps a 64x64x3 RGB image to a scalar output (logit).
    Note: To support Vanilla GAN, LSGAN, and WGAN concurrently, the final sigmoid
    activation is OMITTED from this network and is handled dynamically inside the loss function.
    """
    def __init__(self, features_d=64):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # Layer 1: (batch, 3, 64, 64) -> (batch, features_d, 32, 32)
            # No BatchNorm in the first layer as recommended in DCGAN
            nn.Conv2d(3, features_d, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 2: (batch, features_d, 32, 32) -> (batch, features_d * 2, 16, 16)
            nn.Conv2d(features_d, features_d * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(features_d * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 3: (batch, features_d * 2, 16, 16) -> (batch, features_d * 4, 8, 8)
            nn.Conv2d(features_d * 2, features_d * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(features_d * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 4: (batch, features_d * 4, 8, 8) -> (batch, features_d * 8, 4, 4)
            nn.Conv2d(features_d * 4, features_d * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(features_d * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 5: (batch, features_d * 8, 4, 4) -> (batch, 1, 1, 1) -> squeezed to (batch, 1)
            nn.Conv2d(features_d * 8, 1, kernel_size=4, stride=1, padding=0, bias=False)
        )

    def forward(self, x):
        output = self.main(x)
        # Squeeze output to shape (batch_size, 1)
        return output.view(-1, 1)

if __name__ == "__main__":
    # Quick sanity check
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running sanity check on device: {device}")
    
    netG = Generator().to(device)
    netG.apply(weights_init)
    netD = Discriminator().to(device)
    netD.apply(weights_init)
    
    # Test G forward
    test_z = torch.randn(8, 128, device=device)
    fake_img = netG(test_z)
    print(f"Generator output shape: {fake_img.shape} (Expected: torch.Size([8, 3, 64, 64]))")
    assert fake_img.shape == (8, 3, 64, 64), "Incorrect Generator output dimensions!"
    
    # Test D forward
    score = netD(fake_img)
    print(f"Discriminator output shape: {score.shape} (Expected: torch.Size([8, 1]))")
    assert score.shape == (8, 1), "Incorrect Discriminator output dimensions!"
    print("Sanity checks passed successfully!")
