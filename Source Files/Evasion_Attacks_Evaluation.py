#Necessary Imports

import torchattacks
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import AdamW
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.signal import stft
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from itertools import combinations
import h5py
import warnings
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from tqdm import tqdm
from collections import defaultdict
import cv2

##
warnings.filterwarnings("ignore")

drive.mount('/content/drive') #Replace with your actual path

# Load and preprocess data for generating adversarial examples (PGD-1, PGD-2, PGD-3, FSGM)
# Spectrogram 
def iq_to_spectrogram(iq_data, nperseg=256, noverlap=128):
    I = iq_data[::2]
    Q = iq_data[1::2]
    sig = I + 1j * Q
    _, _, Zxx = stft(sig, nperseg=nperseg, noverlap=noverlap)
    return np.abs(Zxx)

# Dataset
class CustomH5Dataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data.astype(np.float32)
        self.labels = labels.flatten()
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample, label

# Load & Process Data 
dataset_path = '/content/drive/MyDrive/dataset_training_no_aug.h5' #Replace with your actual path
with h5py.File(dataset_path, 'r') as f:
    data = np.array(f['data'])
    labels = np.array(f['label']).squeeze(0)

spectrograms = np.array([iq_to_spectrogram(iq) for iq in data])  # (15000, 256, 65)
spectrograms = spectrograms.reshape(-1, 256, 65, 1)
labels = labels - 1  # Convert to 0-indexed

mean = np.mean(spectrograms)
std = np.std(spectrograms)

train_data, test_data, train_labels, test_labels = train_test_split(
    spectrograms, labels, test_size=0.1, random_state=42
)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[mean], std=[std])
])

train_dataset = CustomH5Dataset(train_data, train_labels, transform)
test_dataset = CustomH5Dataset(test_data, test_labels, transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Vision Transformer
class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # [B, E, H/P1, W/P2]
        x = x.flatten(2).transpose(1, 2)  # [B, N, E]
        return x

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_dim, embed_dim)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim, depth, num_heads, mlp_dim, num_classes):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
        self.encoder = nn.ModuleList([TransformerEncoder(embed_dim, num_heads, mlp_dim) for _ in range(depth)])
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) + self.pos_embed
        for layer in self.encoder:
            x = layer(x)
        return self.classifier(x[:, 0])

# Model Setup  
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

img_size = (256, 65)
patch_size = (16, 8)
model = VisionTransformer(img_size, patch_size, 1, 128, depth=6, num_heads=8, mlp_dim=256, num_classes=30).to(device)

state = torch.load('/content/drive/MyDrive/vit_model_speechcommands.pth', map_location=device) # Load the model after cross-domain training (from main_audio_dataset.py or from main.py)
if 'pos_embed' in state:
    print("Interpolating positional embeddings...")
    pre_pos = state['pos_embed']
    cls_tok, pre_patches = pre_pos[:, :1, :], pre_pos[:, 1:, :]
    old_size = int((pre_patches.shape[1]) ** 0.5)
    new_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
    pre_patches = pre_patches.permute(0, 2, 1).reshape(1, 128, old_size, old_size)
    new_patches = nn.functional.interpolate(pre_patches, size=new_size, mode='bicubic', align_corners=False)
    new_patches = new_patches.reshape(1, 128, -1).permute(0, 2, 1)
    state['pos_embed'] = torch.cat((cls_tok, new_patches), dim=1)
model.load_state_dict(state, strict=False)

# Adversarial Training 

optimizer = optim.Adam(model.parameters(), lr=5e-3)#, weight_decay=1e-5)
criterion = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=155, eta_min=1e-5)
# Initialize PGD attack for adversarial examples
pgd = torchattacks.PGD(model, eps=0.01, alpha=0.01, steps=1) # Change the value of eps to increase or decrease the perturbation ratio. Increase steps for PGD-2 and PGD-3

# Training loop with adversarial examples
for epoch in range(155):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for x, y in train_loader:
        x, y = x.to(device).float(), y.to(device).long()

        # Generate adversarial examples
        x_adv = pgd(x, y)

        # Forward pass with adversarial examples
        optimizer.zero_grad()
        logits = model(x_adv)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    scheduler.step()

    acc = 100 * correct / total
    print(f"Epoch {epoch+1} | Loss: {total_loss/len(train_loader):.4f} | Acc: {acc:.2f}%")

# Save the adversarially fine-tuned model
torch.save(model.state_dict(), '/content/drive/MyDrive/vit_adversarial_finetuned.pth') # Replace with your actual path
print("Adversarially fine-tuned model saved.")

# Load the finetuned model to evaluate its robustness against evasion attacks

# Load Fine-Tuned Vision Transformer Model 
class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # [B, E, H/P1, W/P2]
        x = x.flatten(2).transpose(1, 2)  # [B, N, E]
        return x


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_dim, embed_dim)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim, depth, num_heads, mlp_dim, num_classes):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
        self.encoder = nn.ModuleList([TransformerEncoder(embed_dim, num_heads, mlp_dim) for _ in range(depth)])
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) + self.pos_embed
        for layer in self.encoder:
            x = layer(x)
        return self.classifier(x[:, 0])


# Load the fine-tuned model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

img_size = (256, 65)
patch_size = (16, 8)
model = VisionTransformer(img_size, patch_size, 1, 128, depth=6, num_heads=8, mlp_dim=256, num_classes=30).to(device)

# Load the fine-tuned model state dictionary
model.load_state_dict(torch.load('/content/drive/MyDrive/vit_adversarial_finetuned.pth', map_location=device)) # Replace with your actual path
model.eval()  # Set model to evaluation mode


# Apply Evasion Attack 
# Create PGD attack object
pgd = torchattacks.PGD(model, eps=0.01, alpha=0.01, steps=1) # Change the value of eps to increase or decrease the perturbation ratio. Increase steps for PGD-2 and PGD-3

def evaluate_model_on_adversarial_examples(model, test_loader, attack):
    correct, total = 0, 0
    total_confidence = 0
    total_success = 0
    for x, y in test_loader:
        x, y = x.to(device).float(), y.to(device).long()

        # Apply PGD attack
        x_adv = attack(x, y)

        # Forward pass with adversarial examples
        logits = model(x_adv)
        preds = torch.argmax(logits, dim=1)

        # Attack success
        attack_success = (preds != y).sum().item()
        total_success += attack_success

        correct += (preds == y).sum().item()
        total += y.size(0)

        # Confidence
        confidence = torch.softmax(logits, dim=1)
        total_confidence += confidence.max(dim=1)[0].sum().item()

    accuracy = 100 * correct / total
    avg_confidence = total_confidence / total
    asr = total_success / total
    return accuracy, avg_confidence, asr

# Evaluate on PGD adversarial examples only
pgd_accuracy, pgd_confidence, pgd_asr = evaluate_model_on_adversarial_examples(model, test_loader, pgd)

print(f"Accuracy on PGD adversarial examples: {pgd_accuracy:.2f}% | Average Confidence: {pgd_confidence:.4f} | Attack Success Rate: {pgd_asr:.4f}")
