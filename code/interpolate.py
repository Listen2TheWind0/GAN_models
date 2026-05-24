import os
import argparse
import torch
import torchvision.utils as vutils
from networks import Generator
import utils

def main():
    parser = argparse.ArgumentParser(description="DTS410TC CW2 - Latent Space Interpolation")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the trained generator checkpoint (.pth)")
    parser.add_argument("--output_path", type=str, default="../results/interpolation.png",
                        help="Path to save the output interpolation grid image")
    parser.add_argument("--latent_dim", type=int, default=128,
                        help="Dimension of latent space z")
    parser.add_argument("--num_lines", type=int, default=5,
                        help="Number of interpolation sequences (rows) to generate")
    parser.add_argument("--num_steps", type=int, default=8,
                        help="Number of interpolation steps (columns) per row")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading generator from: {args.checkpoint} on device: {device}")
    
    # Initialize Generator
    netG = Generator(latent_dim=args.latent_dim).to(device)
    
    # Load state dict
    state = torch.load(args.checkpoint, map_location=device)
    if 'generator' in state:
        netG.load_state_dict(state['generator'])
    else:
        netG.load_state_dict(state)
        
    netG.eval()
    
    print(f"Generating {args.num_lines} interpolation sequences with {args.num_steps} steps each...")
    
    # List to store all generated images
    all_images = []
    
    with torch.no_grad():
        for r in range(args.num_lines):
            # Sample two distinct latent vectors z1 and z2 from standard normal distribution
            z1 = torch.randn(1, args.latent_dim, device=device)
            z2 = torch.randn(1, args.latent_dim, device=device)
            
            row_images = []
            for step in range(args.num_steps):
                # Calculate interpolation coefficient alpha in [0, 1]
                alpha = step / (args.num_steps - 1)
                z_alpha = (1 - alpha) * z1 + alpha * z2
                
                # Generate face image
                fake_img = netG(z_alpha)
                row_images.append(fake_img.squeeze(0))
                
            # Concatenate row images along width (dimension 2) or add to global list
            all_images.extend(row_images)
            
    # Stack all images into a single tensor of shape (num_lines * num_steps, 3, 64, 64)
    grid_tensor = torch.stack(all_images)
    
    # Save image grid using the save_image_grid helper from utils
    utils.save_image_grid(grid_tensor, args.output_path, nrow=args.num_steps, padding=2)
    print(f"Successfully saved interpolation grid to: {args.output_path}")

if __name__ == "__main__":
    main()
