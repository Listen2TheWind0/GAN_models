import os
import argparse
import shutil
from PIL import Image
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from networks import Generator
from dataset import CelebADataset
import utils

# Import calculate_fid_given_paths from pytorch_fid package
try:
    from pytorch_fid.fid_score import calculate_fid_given_paths
except ImportError:
    print("WARNING: pytorch-fid is not installed. Please install it using 'pip install pytorch-fid'")

def generate_fake_images(netG, output_dir, num_samples, latent_dim, batch_size, device):
    """
    Generates num_samples fake images from the Generator and saves them to output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    # Clear directory if it has old fake images
    for filename in os.listdir(output_dir):
        file_path = os.path.join(output_dir, filename)
        if os.path.isfile(file_path):
            os.unlink(file_path)
            
    print(f"Generating {num_samples} fake images...")
    netG.eval()
    
    generated_count = 0
    with torch.no_grad():
        while generated_count < num_samples:
            cur_batch = min(batch_size, num_samples - generated_count)
            z = torch.randn(cur_batch, latent_dim, device=device)
            fake_imgs = netG(z)
            # Denormalize to [0, 1]
            fake_imgs = utils.denormalize(fake_imgs.cpu())
            
            for j in range(cur_batch):
                img_tensor = fake_imgs[j]
                # Convert tensor to PIL image
                ndarr = img_tensor.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to(torch.uint8).numpy()
                im = Image.fromarray(ndarr)
                im.save(os.path.join(output_dir, f"fake_{generated_count + j:05d}.png"))
                
            generated_count += cur_batch
            
    print(f"Fake images saved to: {output_dir}")

def prepare_real_test_images(celeba_dir, output_dir, num_samples):
    """
    Finds the official test split images in CelebA, applies center crop and resize to 64x64,
    and saves them to output_dir as PNG files.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if we already have the required number of preprocessed real images
    existing_files = [f for f in os.listdir(output_dir) if f.endswith('.png')]
    if len(existing_files) >= num_samples:
        print(f"Real test directory already contains {len(existing_files)} images. Skipping preparation.")
        return
        
    print(f"Preparing {num_samples} preprocessed real test images...")
    
    # Load dataset split = 'test'
    test_dataset = CelebADataset(root_dir=celeba_dir, split='test')
    
    limit = min(num_samples, len(test_dataset))
    for i in tqdm(range(limit)):
        img_tensor = test_dataset[i]
        # Denormalize image tensor from [-1, 1] to [0, 1]
        img_tensor = utils.denormalize(img_tensor.cpu())
        
        # Convert tensor to PIL image
        ndarr = img_tensor.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to(torch.uint8).numpy()
        im = Image.fromarray(ndarr)
        im.save(os.path.join(output_dir, f"real_{i:05d}.png"))
        
    print(f"Real test images saved to: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="DTS410TC CW2 - FID Evaluation Script")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to Generator checkpoint file (.pth)")
    parser.add_argument("--celeba_dir", type=str, default="../celeba",
                        help="Path to CelebA dataset root folder")
    parser.add_argument("--num_samples", type=int, default=5000,
                        help="Number of samples to generate for FID (default: 5000)")
    parser.add_argument("--batch_size", type=int, default=50,
                        help="Batch size for generating images and feature extraction")
    parser.add_argument("--latent_dim", type=int, default=128,
                        help="Latent vector dimension z")
    parser.add_argument("--temp_dir", type=str, default="../results/fid_temp",
                        help="Temporary directory to store evaluation images")
    parser.add_argument("--clean", action="store_true",
                        help="If set, removes temporary generated fake images after calculation")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("FID Evaluation")
    print(f"Generator Checkpoint: {args.checkpoint}")
    print(f"Num Samples: {args.num_samples} | Batch Size: {args.batch_size}")
    print(f"Device: {device}")
    print("=" * 60)
    
    # Initialize Generator
    netG = Generator(latent_dim=args.latent_dim).to(device)
    
    # Load state dict
    state = torch.load(args.checkpoint, map_location=device)
    if 'generator' in state:
        netG.load_state_dict(state['generator'])
    else:
        netG.load_state_dict(state)
        
    netG.eval()
    
    # Resolve folders
    model_name = os.path.basename(os.path.dirname(args.checkpoint))
    if not model_name:
        model_name = "eval_model"
        
    real_dir = os.path.join(args.temp_dir, "real_test")
    fake_dir = os.path.join(args.temp_dir, f"fake_{model_name}")
    
    # 1. Prepare real test images (preprocessed center-cropped and resized to 64x64)
    prepare_real_test_images(args.celeba_dir, real_dir, args.num_samples)
    
    # 2. Generate fake images from Generator
    generate_fake_images(netG, fake_dir, args.num_samples, args.latent_dim, args.batch_size, device)
    
    # 3. Compute FID score
    print("Computing FID score via Inception-V3 feature extraction...")
    fid_value = calculate_fid_given_paths([real_dir, fake_dir],
                                          batch_size=args.batch_size,
                                          device=device,
                                          dims=2048)
                                          
    print("\n" + "*" * 50)
    print(f"FID Score for {model_name}: {fid_value:.4f}")
    print("*" * 50 + "\n")
    
    # Write FID score to results folder
    result_folder = os.path.dirname(args.checkpoint).replace("checkpoints", "results")
    os.makedirs(result_folder, exist_ok=True)
    with open(os.path.join(result_folder, "fid_score.txt"), "w") as f:
        f.write(f"FID: {fid_value:.4f}\n")
        
    # Clean up fake images folder if requested
    if args.clean:
        print(f"Cleaning up temporary fake images in {fake_dir}...")
        shutil.rmtree(fake_dir)
        
    print("Evaluation completed successfully!")

if __name__ == "__main__":
    main()
