import os
import argparse
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from networks import Generator, Discriminator, weights_init
from dataset import CelebADataset
import utils

def main():
    parser = argparse.ArgumentParser(description="DTS410TC CW2 - GAN Training Script")
    parser.add_argument("--objective", type=str, required=True, choices=["vanilla", "lsgan", "wgan"],
                        help="GAN objective formulation to train: vanilla, lsgan, or wgan")
    parser.add_argument("--celeba_dir", type=str, default="d:/GAN_models/celeba",
                        help="Path to CelebA dataset root directory")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size during training")
    parser.add_argument("--lr_g", type=float, default=-1.0,
                        help="Generator learning rate (default: 2e-4 for vanilla/lsgan, 5e-5 for wgan)")
    parser.add_argument("--lr_d", type=float, default=-1.0,
                        help="Discriminator learning rate (default: 2e-4 for vanilla/lsgan, 5e-5 for wgan)")
    parser.add_argument("--latent_dim", type=int, default=128,
                        help="Latent vector dimension z")
    parser.add_argument("--clip_value", type=float, default=0.01,
                        help="Weight clipping threshold c for WGAN critic")
    parser.add_argument("--n_critic", type=int, default=-1,
                        help="Critic updates per Generator update (default: 5 for WGAN, 1 for vanilla/lsgan)")
    parser.add_argument("--checkpoint_dir", type=str, default="d:/GAN_models/checkpoints",
                        help="Directory to save weights/checkpoints")
    parser.add_argument("--result_dir", type=str, default="d:/GAN_models/results",
                        help="Directory to save training curves and image grids")
    parser.add_argument("--save_interval", type=int, default=5,
                        help="Checkpoint saving interval in epochs")
    parser.add_argument("--log_interval", type=int, default=100,
                        help="Batch logging interval")
    parser.add_argument("--num_workers", type=int, default=2,
                        help="Dataloader number of workers")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training from")
    parser.add_argument("--max_steps", type=int, default=-1,
                        help="Maximum training steps per epoch (default: -1, all steps)")
    
    args = parser.parse_args()
    
    # 1. Resolve Hyperparameter Defaults based on GAN variant
    if args.objective == "wgan":
        lr_g = 5e-5 if args.lr_g < 0 else args.lr_g
        lr_d = 5e-5 if args.lr_d < 0 else args.lr_d
        n_critic = 5 if args.n_critic < 0 else args.n_critic
    else:
        lr_g = 2e-4 if args.lr_g < 0 else args.lr_g
        lr_d = 2e-4 if args.lr_d < 0 else args.lr_d
        n_critic = 1 if args.n_critic < 0 else args.n_critic
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print(f"Training GAN Variant: {args.objective.upper()}")
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs} | Batch Size: {args.batch_size}")
    print(f"Generator LR: {lr_g} | Discriminator/Critic LR: {lr_d}")
    print(f"Critic Updates per G Step: {n_critic}")
    if args.objective == "wgan":
        print(f"WGAN Weight Clipping: [-{args.clip_value}, {args.clip_value}]")
    print("=" * 60)
    
    # Create output folders
    checkpoint_folder = os.path.join(args.checkpoint_dir, args.objective)
    result_folder = os.path.join(args.result_dir, args.objective)
    os.makedirs(checkpoint_folder, exist_ok=True)
    os.makedirs(result_folder, exist_ok=True)
    
    # 2. Data Pipeline
    dataset = CelebADataset(root_dir=args.celeba_dir, split="train")
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, 
                            num_workers=args.num_workers, drop_last=True)
    
    # 3. Initialize Networks
    netG = Generator(latent_dim=args.latent_dim).to(device)
    netD = Discriminator().to(device)
    
    netG.apply(weights_init)
    netD.apply(weights_init)
    
    # 4. Set up Loss criteria and Optimizers
    if args.objective == "vanilla":
        criterion = nn.BCEWithLogitsLoss()
        optimizer_g = torch.optim.Adam(netG.parameters(), lr=lr_g, betas=(0.5, 0.999))
        optimizer_d = torch.optim.Adam(netD.parameters(), lr=lr_d, betas=(0.5, 0.999))
    elif args.objective == "lsgan":
        criterion = nn.MSELoss()
        optimizer_g = torch.optim.Adam(netG.parameters(), lr=lr_g, betas=(0.5, 0.999))
        optimizer_d = torch.optim.Adam(netD.parameters(), lr=lr_d, betas=(0.5, 0.999))
    elif args.objective == "wgan":
        # WGAN does not use standard nn.Module loss functions; loss is computed directly on raw scores
        optimizer_g = torch.optim.RMSprop(netG.parameters(), lr=lr_g)
        optimizer_d = torch.optim.RMSprop(netD.parameters(), lr=lr_d)
        
    start_epoch = 0
    g_losses = []
    d_losses = []
    
    # Resume training if specified
    if args.resume:
        print(f"Resuming training from checkpoint: {args.resume}")
        start_epoch, loaded_g_losses, loaded_d_losses = utils.load_checkpoint(
            args.resume, netG, netD, optimizer_g, optimizer_d
        )
        g_losses.extend(loaded_g_losses)
        d_losses.extend(loaded_d_losses)
        print(f"Resumed at Epoch {start_epoch + 1}")
        
    # Generate fixed noise vector for progress visualization
    fixed_noise = torch.randn(64, args.latent_dim, device=device)
    
    # 5. Training Loop
    total_steps = len(dataloader)
    print("Starting training process...")
    
    for epoch in range(start_epoch, args.epochs):
        epoch_g_losses = []
        epoch_d_losses = []
        start_time = time.time()
        
        for i, real_images in enumerate(dataloader):
            if args.max_steps > 0 and i >= args.max_steps:
                break
            batch_size = real_images.size(0)
            real_images = real_images.to(device)
            
            # -----------------------------------------------------------
            # A. Optimize Discriminator / Critic
            # -----------------------------------------------------------
            optimizer_d.zero_grad()
            
            # Generate fake images
            noise = torch.randn(batch_size, args.latent_dim, device=device)
            fake_images = netG(noise)
            
            # Forward pass real and fake through D
            d_real_out = netD(real_images)
            d_fake_out = netD(fake_images.detach())
            
            # Loss calculations per objective
            if args.objective == "vanilla":
                # Real labels = 1, Fake labels = 0
                loss_d_real = criterion(d_real_out, torch.ones(batch_size, 1, device=device))
                loss_d_fake = criterion(d_fake_out, torch.zeros(batch_size, 1, device=device))
                loss_d = loss_d_real + loss_d_fake
            elif args.objective == "lsgan":
                # LSGAN: Real label = 1, Fake label = 0, with MSE loss
                loss_d_real = 0.5 * criterion(d_real_out, torch.ones(batch_size, 1, device=device))
                loss_d_fake = 0.5 * criterion(d_fake_out, torch.zeros(batch_size, 1, device=device))
                loss_d = loss_d_real + loss_d_fake
            elif args.objective == "wgan":
                # WGAN: maximize E[D(x)] - E[D(G(z))] -> minimize -E[D(x)] + E[D(G(z))]
                loss_d = -torch.mean(d_real_out) + torch.mean(d_fake_out)
                
            loss_d.backward()
            optimizer_d.step()
            
            # Apply weight clipping for WGAN Critic
            if args.objective == "wgan":
                for p in netD.parameters():
                    p.data.clamp_(-args.clip_value, args.clip_value)
            
            epoch_d_losses.append(loss_d.item())
            
            # -----------------------------------------------------------
            # B. Optimize Generator
            # -----------------------------------------------------------
            # Only update generator every n_critic steps
            if (i + 1) % n_critic == 0:
                optimizer_g.zero_grad()
                
                # Generate new fake images (or reuse current ones, new noise is cleaner)
                noise = torch.randn(batch_size, args.latent_dim, device=device)
                fake_images = netG(noise)
                d_fake_g_out = netD(fake_images)
                
                # Loss calculations per objective
                if args.objective == "vanilla":
                    # Generator wants D to classify fake as 1 (real)
                    loss_g = criterion(d_fake_g_out, torch.ones(batch_size, 1, device=device))
                elif args.objective == "lsgan":
                    # LSGAN: Generator wants D to output 1 (real) with MSE loss
                    loss_g = 0.5 * criterion(d_fake_g_out, torch.ones(batch_size, 1, device=device))
                elif args.objective == "wgan":
                    # WGAN: maximize E[D(G(z))] -> minimize -E[D(G(z))]
                    loss_g = -torch.mean(d_fake_g_out)
                    
                loss_g.backward()
                optimizer_g.step()
                
                epoch_g_losses.append(loss_g.item())
            
            # Log batches progress
            if (i + 1) % args.log_interval == 0:
                cur_loss_g = epoch_g_losses[-1] if epoch_g_losses else 0.0
                print(f"Epoch [{epoch+1}/{args.epochs}] Batch [{i+1}/{total_steps}] | "
                      f"Loss_D: {loss_d.item():.4f} | Loss_G: {cur_loss_g:.4f}")
                      
        # Record average epoch losses (aggregate over the epoch)
        avg_loss_d = sum(epoch_d_losses) / len(epoch_d_losses)
        avg_loss_g = sum(epoch_g_losses) / len(epoch_g_losses) if epoch_g_losses else 0.0
        d_losses.append(avg_loss_d)
        g_losses.append(avg_loss_g)
        
        epoch_time = time.time() - start_time
        print(f"--> Epoch {epoch+1} Complete | Time: {epoch_time:.1f}s | "
              f"Avg Loss_D: {avg_loss_d:.4f} | Avg Loss_G: {avg_loss_g:.4f}")
              
        # 6. Save Sample Image Grids at checkpoints (Early: Epoch 1, Mid: Halfway, Final: End)
        if epoch == 0:
            grid_path = os.path.join(result_folder, "samples_epoch_001_early.png")
            with torch.no_grad():
                fake_samples = netG(fixed_noise)
            utils.save_image_grid(fake_samples, grid_path)
            print(f"Saved early training samples to {grid_path}")
        elif epoch == (args.epochs // 2) - 1:
            grid_path = os.path.join(result_folder, f"samples_epoch_{epoch+1:03d}_mid.png")
            with torch.no_grad():
                fake_samples = netG(fixed_noise)
            utils.save_image_grid(fake_samples, grid_path)
            print(f"Saved mid training samples to {grid_path}")
        elif epoch == args.epochs - 1:
            grid_path = os.path.join(result_folder, f"samples_epoch_{epoch+1:03d}_final.png")
            with torch.no_grad():
                fake_samples = netG(fixed_noise)
            utils.save_image_grid(fake_samples, grid_path)
            print(f"Saved final training samples to {grid_path}")
            
        # 7. Periodic Checkpoints and Loss Curve plotting
        if (epoch + 1) % args.save_interval == 0 or (epoch + 1) == args.epochs:
            checkpoint_path = os.path.join(checkpoint_folder, f"checkpoint_epoch_{epoch+1}.pth")
            utils.save_checkpoint({
                'epoch': epoch + 1,
                'generator': netG.state_dict(),
                'discriminator': netD.state_dict(),
                'optimizer_g': optimizer_g.state_dict(),
                'optimizer_d': optimizer_d.state_dict(),
                'g_losses': g_losses,
                'd_losses': d_losses
            }, checkpoint_path)
            print(f"Saved model checkpoint to {checkpoint_path}")
            
            # Save loss history and plot training curves
            loss_history_path = os.path.join(result_folder, "loss_history.json")
            utils.save_loss_history(g_losses, d_losses, loss_history_path)
            
            curve_path = os.path.join(result_folder, "loss_curves.png")
            utils.plot_loss_curves(g_losses, d_losses, curve_path, 
                                   title=f"{args.objective.upper()} GAN Training Loss Curves")
            print(f"Updated loss curve plot at {curve_path}")
            
    print(f"Training for {args.objective.upper()} finished successfully!")

if __name__ == "__main__":
    main()
