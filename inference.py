import torch
from model import *
from dataset import *
from torch.utils.data import DataLoader
import config
from tqdm import tqdm



model = dense(num_classes=15)
model.load_state_dict(torch.load("best_model.pth"))
model.eval()

device = torch.device('cuda' if torch.cuda.is_available(
    ) else 'mps' if torch.backends.mps.is_available() else 'cpu')

model.to(device)

_, _, val_X, val_y = load_dataset(config.data_path)

input_size = 448
collator_test = Collator(train=False, input_size=input_size)

test_dataset = GeoDatset(val_X, val_y)

trans = torch.nn.Sequential(
        transforms.Resize(input_size),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    )

with torch.no_grad():

    total_loss = 0

    all_predicted = []
    all_labels = []
    correct = 0
    total = 0
    cur_label = 0
    Xs = []
    ys = []
    idx = 0

    while idx < len(test_dataset):
        X, y, label = test_dataset[idx]
        if label == cur_label:
            Xs.append(X)
            ys.append(y)
        else:
            images = torch.stack(Xs).to(device)
            labels = torch.stack(ys).to(device)
            images = trans(images)
            outputs = model(images)
            _, predicted = torch.max(torch.sum(outputs.data, dim=0), dim=0)
            total += 1
            correct += (predicted == labels[0]).sum().item()
            Xs = [X]
            ys = [y]
            cur_label = label

        idx += 1

    print(correct / total)