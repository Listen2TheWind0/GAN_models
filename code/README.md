# DTS410TC Generative AI - Coursework 2: Training and Analysis of GAN Variants for Face Image Generation

This repository contains the complete implementation and evaluation code for **Coursework 2: Face Image Generation using Generative Adversarial Networks (GANs)** on the CelebA dataset ($64 \times 64$ resolution). It compares three major GAN objectives:
1. **Vanilla GAN** (Standard Minimax Loss with Non-Saturating Heuristic)
2. **LSGAN** (Least-Squares Loss)
3. **WGAN** (Wasserstein GAN with Weight Clipping)

---

## Project Structure

```text
code/
├── dataset.py        # Custom Dataset class that handles CelebA train/val/test splits
├── networks.py       # PyTorch classes for the Generator and Discriminator/Critic
├── utils.py          # Logging, plotting curves, saving image grids, and checkpoints
├── train.py          # Unified script to train Vanilla, LSGAN, or WGAN models
├── interpolate.py    # Latent space linear interpolation analysis
├── evaluate.py       # Frechet Inception Distance (FID) computation script
└── README.md         # Documentation and execution instructions
```

---

## Theoretical Explanations

### 1. Vanilla GAN
The minimax objective with Binary Cross-Entropy (BCE) loss:
$$\min_G \max_D \mathbb{E}_{x \sim p_{data}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]$$
* **Non-Saturating Heuristic:** To prevent gradient vanishing in the early stage of training $G$, we minimize $-\mathbb{E}_{z}[\log D(G(z))]$ instead of minimizing $\mathbb{E}_{z}[\log(1 - D(G(z)))]$.

### 2. Least-Squares GAN (LSGAN)
Uses mean-squared error (L2 loss) which provides stable, continuous gradients even for samples far from the decision boundary:
$$\mathcal{L}_D = \frac{1}{2} \mathbb{E}_{x \sim p_{data}}[(D(x) - 1)^2] + \frac{1}{2} \mathbb{E}_{z \sim p_z}[D(G(z))^2]$$
$$\mathcal{L}_G = \frac{1}{2} \mathbb{E}_{z \sim p_z}[(D(G(z)) - 1)^2]$$

### 3. Wasserstein GAN (WGAN)
Uses Earth Mover's (Wasserstein) Distance:
$$\min_G \max_{D \in \mathcal{D}} \mathbb{E}_{x \sim p_{data}}[D(x)] - \mathbb{E}_{z \sim p_z}[D(G(z))]$$

* **Why is the Sigmoid activation removed in WGAN?**
  Sigmoid squashes outputs to $[0, 1]$, representing a probability. WGAN calculates the Wasserstein distance using the Kantorovich-Rubinstein duality, where the critic $D(x)$ must output unconstrained real scores (Wasserstein potentials) $\mathbb{R}$. Keeping Sigmoid would restrict the score range, saturate gradients, and prevent $D$ from properly estimating the Wasserstein distance.
* **What is the role of the Lipschitz constraint?**
  The Kantorovich-Rubinstein duality requires the critic $D$ to be a 1-Lipschitz continuous function ($|D(x_1) - D(x_2)| \le |x_1 - x_2|$). This bounds the gradients of the critic, preventing them from exploding during backpropagation. In this implementation, the Lipschitz constraint is enforced by **weight clipping**, restricting the critic's parameters to a compact metric space $[-c, c]$ (typically $c = 0.01$) after each parameter update.

---

## Installation & Environment Setup

We recommend using Anaconda or Miniconda for environment management.

```bash
# Create and activate conda environment
conda create -n dts410_cw2 python=3.10 -y
conda activate dts410_cw2

# Install PyTorch with CUDA support (e.g., CUDA 12.4)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install additional requirements
pip install numpy matplotlib pillow tqdm pytorch-fid
```

---

## Execution Instructions

Ensure that the CelebA dataset is extracted under the folder `d:/GAN_models/celeba` with the following structure:
```text
celeba/
├── Eval/
│   └── list_eval_partition.txt
├── Img/
│   └── img_align_celeba/
│       ├── 000001.jpg
│       └── ...
```

### 1. Training a Model
To launch training, run the `train.py` script and specify the `--objective` parameter:

```bash
# Train Vanilla GAN
python train.py --objective vanilla --epochs 50 --batch_size 128

# Train LSGAN
python train.py --objective lsgan --epochs 50 --batch_size 128

# Train WGAN (automatically uses RMSprop, weight clipping c=0.01, and n_critic=5)
python train.py --objective wgan --epochs 50 --batch_size 128
```

*Note: All checkpoints will be saved in `d:/GAN_models/checkpoints/<objective>/` and results (training loss curves, checkpoint grids) in `d:/GAN_models/results/<objective>/`.*

### 2. Latent Space Interpolation
To perform linear interpolation between randomly sampled latent codes using a trained generator:

```bash
python interpolate.py \
    --checkpoint d:/GAN_models/checkpoints/wgan/checkpoint_epoch_50.pth \
    --output_path d:/GAN_models/results/wgan/interpolation.png \
    --num_lines 5 \
    --num_steps 8
```

### 3. Quantitative Evaluation (FID Score)
To calculate the Frechet Inception Distance (FID) using 5,000 generated images compared to 5,000 real images from the official test split:

```bash
python evaluate.py \
    --checkpoint d:/GAN_models/checkpoints/wgan/checkpoint_epoch_50.pth \
    --celeba_dir d:/GAN_models/celeba \
    --num_samples 5000 \
    --batch_size 50 \
    --clean
```
*(The real test images will be automatically cropped to $128 \times 128$ and resized to $64 \times 64$ to match the Generator output format perfectly, ensuring a mathematically fair FID comparison).*
