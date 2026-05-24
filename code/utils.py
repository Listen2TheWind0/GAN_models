import os
import json
import matplotlib.pyplot as plt
import torch
import torchvision.utils as vutils

def denormalize(tensor):
    """
    Denormalizes image tensors from [-1, 1] back to [0, 1] range.
    """
    return tensor * 0.5 + 0.5

def save_image_grid(tensor, filepath, nrow=8, padding=2):
    """
    Saves a batch of image tensors to a grid file.
    Args:
        tensor (Tensor): Image batch of shape (B, C, H, W).
        filepath (str): Destination file path.
        nrow (int): Number of images displayed in each row.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    # Denormalize first to map image range to [0, 1]
    denorm_tensor = denormalize(tensor.detach().cpu())
    vutils.save_image(denorm_tensor, filepath, nrow=nrow, padding=padding, normalize=False)

def plot_loss_curves(g_losses, d_losses, filepath, title="Training Loss Curves"):
    """
    Plots Generator and Discriminator/Critic loss history and saves to filepath.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(g_losses, label="Generator Loss", color="royalblue", alpha=0.8)
    plt.plot(d_losses, label="Discriminator/Critic Loss", color="darkorange", alpha=0.8)
    plt.xlabel("Iterations (or Batches)")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close()

def save_loss_history(g_losses, d_losses, filepath):
    """
    Saves the loss list to a JSON file.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    data = {
        "g_losses": g_losses,
        "d_losses": d_losses
    }
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)

def load_loss_history(filepath):
    """
    Loads loss lists from a JSON file.
    """
    if not os.path.exists(filepath):
        return [], []
    with open(filepath, "r") as f:
        data = json.load(f)
    return data.get("g_losses", []), data.get("d_losses", [])

def save_checkpoint(state, filepath):
    """
    Saves PyTorch training state/checkpoint.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(state, filepath)

def load_checkpoint(filepath, model_g=None, model_d=None, optimizer_g=None, optimizer_d=None):
    """
    Loads PyTorch training state/checkpoint.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint not found at: {filepath}")
        
    state = torch.load(filepath, map_location=lambda storage, loc: storage)
    
    if model_g is not None and 'generator' in state:
        model_g.load_state_dict(state['generator'])
    if model_d is not None and 'discriminator' in state:
        model_d.load_state_dict(state['discriminator'])
    if optimizer_g is not None and 'optimizer_g' in state:
        optimizer_g.load_state_dict(state['optimizer_g'])
    if optimizer_d is not None and 'optimizer_d' in state:
        optimizer_d.load_state_dict(state['optimizer_d'])
        
    return state.get('epoch', 0), state.get('g_losses', []), state.get('d_losses', [])
