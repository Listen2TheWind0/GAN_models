import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class CelebADataset(Dataset):
    """
    Custom Dataset class for loading Align & Cropped CelebA images.
    Filters images based on train/val/test splits using list_eval_partition.txt.
    """
    def __init__(self, root_dir, split='train', crop_size=128, img_size=64, transform=None):
        """
        Args:
            root_dir (str): Root directory of the dataset (should contain 'Img/img_align_celeba' and 'Eval/list_eval_partition.txt').
            split (str): One of 'train', 'val', or 'test'.
            crop_size (int): Size to center-crop the 178x218 original image. Default is 128.
            img_size (int): Resolution to resize the cropped image to. Default is 64.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.img_dir = os.path.join(root_dir, 'Img', 'img_align_celeba')
        self.partition_file = os.path.join(root_dir, 'Eval', 'list_eval_partition.txt')
        
        if not os.path.exists(self.img_dir):
            raise FileNotFoundError(f"Image directory not found at {self.img_dir}")
        if not os.path.exists(self.partition_file):
            raise FileNotFoundError(f"Partition file not found at {self.partition_file}")
            
        # Map split string to partition integer
        # 0: train, 1: val, 2: test
        split_map = {'train': 0, 'val': 1, 'test': 2}
        if split not in split_map:
            raise ValueError("split must be one of 'train', 'val', or 'test'")
        target_partition = split_map[split]
        
        # Read the partition file and gather matching image filenames
        self.image_filenames = []
        with open(self.partition_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    filename, part_id = parts
                    if int(part_id) == target_partition:
                        self.image_filenames.append(filename)
                        
        print(f"Loaded CelebA [{split}] split: {len(self.image_filenames)} images found.")
        
        # Set up default transform if none is provided
        if transform is not None:
            self.transform = transform
        else:
            transform_list = []
            if crop_size is not None and crop_size > 0:
                transform_list.append(transforms.CenterCrop(crop_size))
            transform_list.extend([
                transforms.Resize(img_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # normalizes from [0, 1] to [-1, 1]
            ])
            self.transform = transforms.Compose(transform_list)

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        filename = self.image_filenames[idx]
        img_path = os.path.join(self.img_dir, filename)
        
        # Open image and convert to RGB
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image

if __name__ == "__main__":
    # Quick sanity check
    root_path = "../celeba"
    try:
        train_dataset = CelebADataset(root_dir=root_path, split='train')
        test_dataset = CelebADataset(root_dir=root_path, split='test')
        
        print(f"Sample image shape: {train_dataset[0].shape} (Expected: torch.Size([3, 64, 64]))")
        print(f"Sample value range: [{train_dataset[0].min():.4f}, {train_dataset[0].max():.4f}] (Expected close to [-1, 1])")
        assert train_dataset[0].shape == (3, 64, 64), "Incorrect dataset image shape!"
        print("Dataset sanity check passed successfully!")
    except Exception as e:
        print(f"Error checking dataset: {e}")
