#Necessary Imports

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
from sklearn.metrics import accuracy_score, pairwise_distances
from scipy.signal import stft
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import torchaudio
from itertools import combinations
import h5py
import warnings
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from tqdm import tqdm
from collections import defaultdict

warnings.filterwarnings("ignore")
# Mount Google Drive (if needed for data)
drive.mount('/content/drive') #Replace with your actual path

#  Spectrogram Function 
def iq_to_spectrogram(iq_data, nperseg=256, noverlap=128):
    I = iq_data[::2]
    Q = iq_data[1::2]
    sig = I + 1j * Q
    _, _, Zxx = stft(sig, nperseg=nperseg, noverlap=noverlap)
    return np.abs(Zxx)

#  Dataset Class 
class SpectrogramDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

#  Load and Process Data 
dataset_path = '/content/drive/MyDrive/dataset_training_no_aug.h5' #Dataset for GAN training to generate spoofed samples: replace with your actual path/dataset
with h5py.File(dataset_path, 'r') as f:
    iq_data = np.array(f['data'])
    labels = np.array(f['label']).squeeze(0)

spectrograms = np.array([iq_to_spectrogram(iq) for iq in iq_data])  # shape (N, 256, 65)
spectrograms = spectrograms.reshape(-1, 1, 256, 65)  
labels = labels - 1  # Ensure 0-indexed

# Normalize dataset
mean = np.mean(spectrograms)
std = np.std(spectrograms)
spectrograms = (spectrograms - mean) / std

# Split dataset
train_data, _, train_labels, _ = train_test_split(spectrograms, labels, test_size=0.1, random_state=42)
dataset = SpectrogramDataset(train_data, train_labels)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

#  Conditional GAN Architecture 
class Generator(nn.Module):
    def __init__(self, noise_dim, num_classes, img_shape):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, noise_dim)
        self.model = nn.Sequential(
            nn.Linear(noise_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, int(np.prod(img_shape))),
            nn.Tanh()
        )
        self.img_shape = img_shape

    def forward(self, noise, labels):
        gen_input = noise * self.label_emb(labels)
        out = self.model(gen_input)
        return out.view(out.size(0), *self.img_shape)

class Discriminator(nn.Module):
    def __init__(self, num_classes, img_shape):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, int(np.prod(img_shape)))
        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1)
        )

    def forward(self, imgs, labels):
        flat_imgs = imgs.view(imgs.size(0), -1)
        flat_labels = self.label_emb(labels)
        d_in = flat_imgs * flat_labels
        return self.model(d_in)

#  Training Setup 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
img_shape = (1, 256, 65)
noise_dim = 100
num_classes = len(np.unique(labels))

G = Generator(noise_dim, num_classes, img_shape).to(device)
D = Discriminator(num_classes, img_shape).to(device)

optimizer_G = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))
criterion = nn.BCEWithLogitsLoss()

#  cGAN Training Loop 
epochs = 50
for epoch in range(epochs):
    for imgs, labels in dataloader:
        batch_size = imgs.size(0)
        real_imgs = imgs.to(device)
        labels = labels.to(device)

        # Label smoothing and noise
        real_labels = torch.full((batch_size, 1), 0.9, device=device)
        fake_labels = torch.full((batch_size, 1), 0.1, device=device)

        # Add small noise to real images (optional but stabilizing)
        real_imgs += 0.05 * torch.randn_like(real_imgs)

        #  Train Generator 
        optimizer_G.zero_grad()
        noise = torch.randn(batch_size, noise_dim).to(device)
        gen_imgs = G(noise, labels)
        g_loss = criterion(D(gen_imgs, labels), real_labels)
        g_loss.backward()
        optimizer_G.step()

        #  Train Discriminator 
        optimizer_D.zero_grad()
        real_loss = criterion(D(real_imgs, labels), real_labels)
        fake_loss = criterion(D(gen_imgs.detach(), labels), fake_labels)
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()

    print(f"[Epoch {epoch+1}] D loss: {d_loss.item():.4f} | G loss: {g_loss.item():.4f}")

#  Save Models 
torch.save(G.state_dict(), '/content/drive/MyDrive/generator_rfspoof.pth') #Replace with your actual path to save the Generator
torch.save(D.state_dict(), '/content/drive/MyDrive/discriminator_rfspoof.pth') #Replace with your actual path to save the Discriminator

# Robustness against RF spoofing and adversarial examples

class Generator(nn.Module):
    def __init__(self, noise_dim, num_classes, img_shape):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, noise_dim)
        self.model = nn.Sequential(
            nn.Linear(noise_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, int(np.prod(img_shape))),
            nn.Tanh()
        )
        self.img_shape = img_shape

    def forward(self, noise, labels):
        gen_input = noise * self.label_emb(labels)
        out = self.model(gen_input)
        return out.view(out.size(0), *self.img_shape)


# Vision Transformer Definition

class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels=1, embed_dim=128):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(out)

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_dim, dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim, depth, num_heads, mlp_dim, num_classes):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(0.1)
        self.blocks = nn.Sequential(*[Block(embed_dim, num_heads, mlp_dim) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        B = x.size(0)
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x[:, 0])
        return self.head(x)


# Setup and load pretrained models

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
img_shape = (1, 256, 65)
noise_dim = 100
num_classes = 30
n_samples_per_class = 100

# Load Generator
G = Generator(noise_dim, num_classes, img_shape).to(device)
G.load_state_dict(torch.load('/content/drive/MyDrive/generator_rfspoof.pth', map_location=device)) #Replace with your actual path to load the pretrained generator
G.eval()

# Load Vision Transformer
vit = VisionTransformer(
    img_size=(256, 65),
    patch_size=(16, 8),
    in_channels=1,
    embed_dim=128,
    depth=6,
    num_heads=8,
    mlp_dim=256,
    num_classes=num_classes
).to(device)

# Load ViT weights with positional embedding interpolation
vit_ckpt = torch.load('/content/drive/MyDrive/vit_supervised_finetuned.pth', map_location=device) #Relace with your actual path to load the pretrained ViT+classifier (from main.py) 

if 'pos_embed' in vit_ckpt:
    pos_embed_pretrained = vit_ckpt['pos_embed']
    embedding_dim = pos_embed_pretrained.shape[-1]
    cls_token = pos_embed_pretrained[:, 0:1, :]
    patch_pos_embed = pos_embed_pretrained[:, 1:, :]
    num_patches_old = patch_pos_embed.shape[1]
    new_grid_size = vit.patch_embed.grid_size
    num_patches_new = new_grid_size[0] * new_grid_size[1]

    if num_patches_old != num_patches_new:
        h_old = int(np.sqrt(num_patches_old))
        w_old = num_patches_old // h_old
        patch_pos_embed = patch_pos_embed.reshape(1, h_old, w_old, embedding_dim).permute(0, 3, 1, 2)
        patch_pos_embed = F.interpolate(patch_pos_embed, size=new_grid_size, mode='bicubic', align_corners=False)
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, -1, embedding_dim)
        vit_ckpt['pos_embed'] = torch.cat((cls_token, patch_pos_embed), dim=1)

vit.load_state_dict(vit_ckpt, strict=False)
vit.eval()


# Compute Mean/Std from Real Data

dataset_path = '/content/drive/MyDrive/dataset_training_no_aug.h5' #Replace with your actual path
with h5py.File(dataset_path, 'r') as f:
    data = np.array(f['data'])
    labels = np.array(f['label']).squeeze(0)

def iq_to_spectrogram(iq_data, nperseg=256, noverlap=128):
    I = iq_data[::2]
    Q = iq_data[1::2]
    sig = I + 1j * Q
    sig = torch.tensor(sig, dtype=torch.cfloat)
    Zxx = torch.stft(sig, n_fft=nperseg, hop_length=nperseg - noverlap, return_complex=True)
    return torch.abs(Zxx).numpy()


spectrograms = np.array([iq_to_spectrogram(iq) for iq in data])  # shape: (N, 256, 65)
spectrograms = spectrograms[..., np.newaxis]  # shape: (N, 256, 65, 1)
mean = np.mean(spectrograms)
std = np.std(spectrograms)


# Define Transform

transform = transforms.Compose([
    transforms.Lambda(lambda x: torch.tensor(x, dtype=torch.float32)),
    transforms.Normalize(mean=[mean], std=[std])
])


# Evaluation

all_preds = []
all_true = []

with torch.no_grad():
    for label in range(num_classes):
        for _ in range(n_samples_per_class):
            noise = torch.randn(1, noise_dim).to(device)
            label_tensor = torch.tensor([label]).to(device)

            fake_img = G(noise, label_tensor)  # shape: (1, 1, 256, 65)
            fake_img_np = fake_img.squeeze(0).cpu().numpy()         # (1, 256, 65)
            fake_img_tensor = torch.tensor(fake_img_np, dtype=torch.float32)
            fake_img_transformed = transform(fake_img_tensor)       # (1, 256, 65)
            fake_img_transformed = fake_img_transformed.unsqueeze(0).to(device)  # (1, 1, 256, 65)

            logits = vit(fake_img_transformed)
            pred = torch.argmax(logits, dim=1).item()

            all_preds.append(pred)
            all_true.append(label)


# Evaluation Results

acc = accuracy_score(all_true, all_preds)
print(f"Accuracy on spoofed samples: {acc * 100:.2f}%")
