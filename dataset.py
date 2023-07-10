import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


classes = [l for l in os.listdir("Dataset") if not l.startswith(".")]
class_to_idx = {classes[i]: i for i in range(len(classes))}

def load_dataset(dir):
    train_labels, train_fpaths = [], []
    val_labels, val_fpaths = [], []
    for fname in os.listdir(dir):
        if fname.startswith("."):
            continue
        with open(os.path.join(dir, fname, f"{fname}.csv")) as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            if not line.strip():
                continue
            label = fname
            _, _, fpath = line.strip().split(",")
            if i < 10:
                val_fpaths.append(os.path.join(dir, fname, f"{fpath}.png"))
                val_labels.append(label)
            else:
                train_fpaths.append(os.path.join(dir, fname, f"{fpath}.png"))
                train_labels.append(label)

    return train_fpaths, train_labels, val_fpaths, val_labels

class GeoDatset(Dataset):
    """Face Landmarks dataset."""
    def __init__(self, X, y):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.X = X
        self.y = y
        self.convert_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = Image.open(self.X[idx])
        sample = (self.convert_tensor(image), 
                  torch.tensor(class_to_idx[self.y[idx]]))

        return sample

def make_x_y(batch):
    X, y = zip(*batch)

    X = torch.stack(X)
    y = torch.stack(y)
    
    trans = torch.nn.Sequential(
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    )
    scripted_transforms = torch.jit.script(trans)

    X = scripted_transforms(X)
    return X, y
