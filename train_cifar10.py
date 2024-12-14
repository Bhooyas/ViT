import torch
from torch import nn
from torch import optim
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from model import ViT
from tqdm import tqdm
from safetensors.torch import save_model, load_model
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
import math

image_size = (3, 64, 64)
patch_size = 8
d_model = 256
heads = 8
d_head = 32
depth = 4
dropout = 0
factor = 4
num_classes = 10
classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

batch_size = 512
lr = 0.003
weight_decay = 0.003
warmup_percent = 0.2
epochs = 100
betas = (0.85, 0.98)
model_weights = "vit_cifar10.safetensors"

device = "cuda" if torch.cuda.is_available() else "cpu"

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomResizedCrop(64),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transform = transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def target_transform(x):
    return F.one_hot(torch.tensor(x), num_classes=num_classes).float()

train_ds = CIFAR10("./data/cifar", transform=train_transform, download=True, target_transform=target_transform)
train_dl = DataLoader(train_ds, shuffle=True, batch_size=batch_size)
test_ds = CIFAR10("./data/cifar", transform=transform, download=True, train=False)
test_dl = DataLoader(test_ds, shuffle=True, batch_size=batch_size)

total_steps = len(train_dl) * epochs
print(f"{total_steps = }")
warmup_steps = math.ceil(total_steps * warmup_percent)

model = ViT(image_size=image_size[1], patch_size=patch_size, num_classes=num_classes, depth=depth, d_model=d_model, heads=heads, d_head=d_head, dropout=dropout, factor=factor).to(device)
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas)
loss_fn = nn.CrossEntropyLoss()
# scheduler = optim.lr_scheduler.LinearLR(optimizer)
# scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
# scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=total_steps//2)
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, epochs=epochs, steps_per_epoch=len(train_dl))

# load_model(model, model_weights)

model_size = sum(p.numel() for p in model.parameters())
print(f"{model_size = :,}")

for epoch in range(epochs):
    pbar = tqdm(train_dl)
    pbar.set_description(f"Epoch {epoch+1}/{epochs}")

    for images, labels in pbar:
        optimizer.zero_grad()

        images, labels = images.to(device), labels.to(device)
        y_hat = model(images)
        loss = loss_fn(y_hat, labels)

        loss.backward()
        optimizer.step()

        pbar.set_postfix({"Loss": loss.item()})

        scheduler.step()
    1/0

y_true = []
y_pred = []

with torch.no_grad():
    model.eval()
    pbar = tqdm(test_dl)
    pbar.set_description(f"Test")
    for images, labels in pbar:
        images = images.to(device)
        outputs = model(images)
        _, predictions = torch.max(outputs, dim=1)

        y_true.extend(labels.numpy())
        y_pred.extend(predictions.cpu().numpy())


cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=classes)
disp.plot()
# plt.show()
plt.savefig('cm.png')

cr = classification_report(y_true, y_pred, target_names=classes)
print(cr)

save_model(model, model_weights)
