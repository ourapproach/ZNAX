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
from torchaudio.datasets import SPEECHCOMMANDS
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from tqdm import tqdm
from collections import defaultdict
import cv2

warnings.filterwarnings("ignore")

drive.mount('/content/drive') #Replace with your actual path

# Patch Embedding

class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super().__init__()
        self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.projection(x)
        x = x.flatten(2).transpose(1, 2)
        return x


# Multi-Head Self-Attention

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


# Transformer Encoder Block

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

  
# Learnable Margins

class LearnableMargins(nn.Module):
    def __init__(self, initial_m1=0.0, initial_m2=0.6):
        super().__init__()
        self.m1 = nn.Parameter(torch.tensor(initial_m1, dtype=torch.float32))
        self.m2 = nn.Parameter(torch.tensor(initial_m2, dtype=torch.float32))

    def forward(self):
        return torch.clamp(self.m1, min=0), torch.clamp(self.m2, min=self.m1 + 0.1)


# Learnable Threshold

class LearnableThreshold(nn.Module):
    def __init__(self, initial_threshold=0.2):
        super().__init__()
        self.threshold = nn.Parameter(torch.tensor(initial_threshold, dtype=torch.float32))

    def forward(self):
        return torch.clamp(self.threshold, min=0.0, max=1.0)


# Dual Margin Contrastive Loss

def dual_margin_contrastive_loss(emb1, emb2, label, m1, m2, threshold):
    emb1 = F.normalize(emb1, p=2, dim=1)
    emb2 = F.normalize(emb2, p=2, dim=1)
    D = F.pairwise_distance(emb1, emb2)

    threshold = torch.clamp(threshold, min=0.0, max=1.0)
    label = torch.sigmoid(D - threshold)

    loss_pos = label * torch.pow(torch.clamp(D - m1, min=0), 2)
    loss_neg = (1 - label) * torch.pow(torch.clamp(m2 - D, min=0), 2)

    loss = 30.0 * torch.mean(loss_pos + loss_neg)
    return loss


# Vision Transformer

class VisionTransformer(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim, depth, num_heads, mlp_dim):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, (img_size // patch_size) ** 2 + 1, embed_dim))
        self.encoder = nn.ModuleList([
            TransformerEncoder(embed_dim, num_heads, mlp_dim) for _ in range(depth)
        ])
        self.learnable_margins = LearnableMargins()

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) + self.pos_embed
        for layer in self.encoder:
            x = layer(x)
        return x[:, 0]

    def get_margins(self):
        return self.learnable_margins()


# Unlabeled Dataset

class UnlabeledSpectrogramDataset(Dataset):
    def __init__(self, dataset, cache_dir, max_length=128, use_gpu=True):
        self.dataset = dataset
        self.cache_dir = cache_dir
        self.max_length = max_length
        os.makedirs(self.cache_dir, exist_ok=True)

        self.device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')

        self.mel_spec = MelSpectrogram(
            sample_rate=16000, n_fft=1024, hop_length=512, n_mels=128
        ).to(self.device)
        self.db_transform = AmplitudeToDB().to(self.device)

        print("Dataset initialized.")
        self._cache_spectrograms()
        self._generate_pairs()

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        i1, i2 = self.pairs[idx]
        spec1, _ = torch.load(os.path.join(self.cache_dir, f"{i1}.pt"))
        spec2, _ = torch.load(os.path.join(self.cache_dir, f"{i2}.pt"))
        return spec1, spec2

    def _cache_spectrograms(self):
        print("Caching spectrograms to disk...")
        for i in tqdm(range(len(self.dataset))):
            cache_path = os.path.join(self.cache_dir, f"{i}.pt")
            if not os.path.exists(cache_path):
                waveform, _, _ = self.dataset[i][:3]
                waveform = waveform.to(self.device)
                mel = self.mel_spec(waveform)
                db = self.db_transform(mel).squeeze(0)

                T = db.shape[1]
                if T < self.max_length:
                    db = F.pad(db, (0, self.max_length - T))
                else:
                    db = db[:, :self.max_length]

                spec = db.unsqueeze(0).cpu()
                torch.save((spec, None), cache_path)

    def _generate_pairs(self):
        indices = list(range(len(self.dataset)))
        random.shuffle(indices)
        self.pairs = [(indices[i], indices[i + 1]) for i in range(0, len(indices) - 1, 2)]


# Prepare Data

raw_data = SPEECHCOMMANDS(root='./', download=True)
subset_data = torch.utils.data.Subset(raw_data, list(range(64000)))

spectrogram_dataset = UnlabeledSpectrogramDataset(
    dataset=subset_data,
    cache_dir="/content/speech_cache"
)

train_loader = DataLoader(spectrogram_dataset, batch_size=64, shuffle=False)


# Model Initialization

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = VisionTransformer(
    img_size=128,
    patch_size=16,
    in_channels=1,
    embed_dim=128,
    depth=6,
    num_heads=8,
    mlp_dim=256
).to(device)

learnable_threshold = LearnableThreshold(initial_threshold=0.2).to(device)

# Set up optimizer
base_params = [p for name, p in model.named_parameters() if "learnable_margins" not in name]

optimizer = AdamW(
    [{'params': base_params},
     {'params': model.learnable_margins.parameters(), 'lr': 5e-5},
     {'params': learnable_threshold.parameters(), 'lr': 4e-5}],
    lr=3.5e-4, weight_decay=1e-1
)
scheduler = CosineAnnealingLR(optimizer, T_max=20)


# Training Loop 

model.train()
num_epochs = 20
total_steps = 0

for epoch in range(num_epochs):
    total_loss = 0.0
    for spec1, spec2 in train_loader:
        spec1 = spec1.to(device)
        spec2 = spec2.to(device)

        emb1 = model(spec1)
        emb2 = model(spec2)

        threshold = learnable_threshold()
        D = F.pairwise_distance(emb1, emb2)
        label = torch.sigmoid(D - threshold)

        m1, m2 = model.get_margins()

        loss = dual_margin_contrastive_loss(emb1, emb2, label, m1, m2, threshold)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_steps += 1

    scheduler.step()
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, m1: {m1.item():.4f}, m2: {m2.item():.4f}, threshold: {threshold.item():.4f}")

# Save model
save_path = '/content/drive/MyDrive/vit_model_speechcommands.pth' # Replace with your actual path
torch.save(model.state_dict(), save_path)
print(f"Model saved to {save_path}")


# Save Embeddings

model.eval()
save_path_embeddings = '/content/drive/MyDrive/embeddings_speechcommands.npy' # Replace with your actual path

all_embeddings1 = []
all_embeddings2 = []

with torch.no_grad():
    for spec1, spec2 in train_loader:
        spec1 = spec1.to(device)
        spec2 = spec2.to(device)

        emb1 = model(spec1)
        emb2 = model(spec2)

        all_embeddings1.append(emb1.cpu().numpy())
        all_embeddings2.append(emb2.cpu().numpy())

all_embeddings1 = np.concatenate(all_embeddings1, axis=0)
all_embeddings2 = np.concatenate(all_embeddings2, axis=0)

combined_embeddings = np.stack((all_embeddings1, all_embeddings2), axis=1)
np.save(save_path_embeddings, combined_embeddings)

print(f"Embeddings saved at: {save_path_embeddings}")

# print("spec1 shape:", spec1.shape)
# with h5py.File('/content/drive/MyDrive/dataset_training_no_aug.h5', 'r') as f: # Replace with your actual path
#     print(f['data'].shape)   # Should print something like (N, ...)
#     print(f['label'].shape)  # If it prints (1,), then that's wrong

# Model Definitions 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super().__init__()
        self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
    def forward(self, x):
        x = self.projection(x)
        x = x.flatten(2).transpose(1, 2)
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
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2,0,3,1,4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scores = (q @ k.transpose(-2,-1)) / (self.head_dim ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        x = (attn @ v).transpose(1,2).reshape(B, N, C)
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

class LearnableMargins(nn.Module):
    def __init__(self, initial_m1=0.0, initial_m2=0.6):
        super().__init__()
        self.m1 = nn.Parameter(torch.tensor(initial_m1))
        self.m2 = nn.Parameter(torch.tensor(initial_m2))
    def forward(self):
        return torch.clamp(self.m1, min=0), torch.clamp(self.m2, min=self.m1+0.1)

class LearnableThreshold(nn.Module):
    def __init__(self, initial_threshold=0.2):
        super().__init__()
        self.threshold = nn.Parameter(torch.tensor(initial_threshold, dtype=torch.float32))
    def forward(self):
        return torch.clamp(self.threshold, min=0.0, max=1.0)

class VisionTransformer(nn.Module):
    def __init__(self, img_size=128, patch_size=16, in_channels=1, embed_dim=128, depth=6, num_heads=8, mlp_dim=256):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1,1,embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, (img_size//patch_size)**2 + 1, embed_dim))
        self.encoder = nn.ModuleList([TransformerEncoder(embed_dim, num_heads, mlp_dim) for _ in range(depth)])
        self.learnable_margins = LearnableMargins()
        self.learnable_threshold = LearnableThreshold()
    def forward(self, x):
        B = x.size(0)
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls, x), dim=1) + self.pos_embed
        for layer in self.encoder:
            x = layer(x)
        return x[:,0]
    def get_margins(self):
        return self.learnable_margins()
    def get_threshold(self):
        return self.learnable_threshold()


# instantiate & load
model = VisionTransformer().to(device)
state = torch.load('/content/drive/MyDrive/vit_model_speechcommands.pth', map_location=device) # Replace with your actual path
model.load_state_dict(state, strict=False)
projection_layer = nn.Linear(128, 64).to(device)

#  Load Source Embeddings 

src_emb = np.load('/content/drive/MyDrive/embeddings_speechcommands.npy') # Replace with your actual path
source_embeddings = torch.tensor(src_emb, dtype=torch.float32, device=device)
# source_embeddings shape [N, 2, 128]

print(f"Source embeddings loaded: {source_embeddings.shape}")

# Dataset Preparation 

def iq_to_spectrogram(iq_data, nperseg=256, noverlap=128):
    I, Q = iq_data[::2], iq_data[1::2]
    sig = I + 1j*Q
    _, _, Zxx = stft(sig, nperseg=nperseg, noverlap=noverlap)
    spec = np.abs(Zxx)
    return spec #cv2.resize(spec, (128,128)) ###spec

class LazyH5Dataset(Dataset):
    def __init__(self, h5_path, indices, mean, std):
        self.h5, self.idx, self.tr = h5_path, indices, transforms.Normalize([mean],[std])
        with h5py.File(self.h5,'r') as f:
            labels = np.array(f['label']).squeeze(0)
        self.labels = labels[indices]
    def __len__(self):
        return len(self.idx)
    def __getitem__(self,i):
        with h5py.File(self.h5,'r') as f:
            iq = f['data'][self.idx[i]]
            spec = iq_to_spectrogram(iq)
        x1 = self.tr(torch.tensor(spec, dtype=torch.float32).unsqueeze(0))
        # negative
        j = i
        while j==i:
            j = random.randrange(len(self.idx))
        with h5py.File(self.h5,'r') as f:
            iq2 = f['data'][self.idx[j]]
            spec2 = iq_to_spectrogram(iq2)
        x2 = self.tr(torch.tensor(spec2, dtype=torch.float32).unsqueeze(0))
        lbl = 1.0 if self.labels[i]==self.labels[j] else 0.0
        return x1, x2, torch.tensor(lbl, dtype=torch.float32)

# compute mean/std on small sample
ds_path = '/content/drive/MyDrive/dataset_training_no_aug.h5' # Replace with your actual path
with h5py.File(ds_path,'r') as f:
    samp = np.array(f['data'][:500])
mean = np.mean([iq_to_spectrogram(s) for s in samp])
std  = np.std ([iq_to_spectrogram(s) for s in samp])

# splits
with h5py.File(ds_path,'r') as f: total = len(f['data'])
inds = list(range(total)); random.shuffle(inds)
t = int(0.8*total)
train_ds = LazyH5Dataset(ds_path, inds[:t], mean, std)
val_ds   = LazyH5Dataset(ds_path, inds[t:], mean, std)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

# Loss Functions

def dual_margin_contrastive_loss(z1, z2, lbl, m1, m2, threshold):
    D = F.pairwise_distance(z1, z2)
    threshold = torch.clamp(threshold, min=0.0, max=1.0)

    # Soft label based on distance and threshold
    soft_label = torch.sigmoid(D - threshold)

    loss_pos = soft_label * (D - m1).clamp(min=0).pow(2)
    loss_neg = (1 - soft_label) * (m2 - D + 1e-4).clamp(min=0).pow(2)

    return 30 * (loss_pos + loss_neg).mean()


def mmd_loss(source, target, bandwidth=1.0):
    # both [B,64]
    source = F.normalize(source,p=2,dim=1)
    target = F.normalize(target,p=2,dim=1)
    def pdist_sq(a,b):
        an = a.pow(2).sum(1,keepdim=True)
        bn = b.pow(2).sum(1,keepdim=True)
        return (an + bn.t() - 2*a.mm(b.t())).clamp(min=0)
    gamma = 1.0/(2*bandwidth**2)
    Kss = torch.exp(-gamma * pdist_sq(source, source))
    Ktt = torch.exp(-gamma * pdist_sq(target, target))
    Kst = torch.exp(-gamma * pdist_sq(source, target))
    return Kss.mean() + Ktt.mean() - 2*Kst.mean()

# Optimizer 

# Separate parameter groups
base_params = [p for name, p in model.named_parameters() if "learnable_margins" not in name and "learnable_threshold" not in name]
margins_params = model.learnable_margins.parameters()
threshold_params = model.learnable_threshold.parameters()
projection_params = projection_layer.parameters()

# Combine base parameters with projection layer
all_base_params = list(base_params) + list(projection_params)

# Create optimizer
opt = AdamW([
    { 'params': all_base_params,       'lr': 3.5e-4 },   # backbone
    { 'params': margins_params,        'lr': 5e-6 },   # margins
    { 'params': threshold_params,      'lr': 4e-6 },   # threshold
], weight_decay=1e-1)



sched = CosineAnnealingLR(opt, T_max=20)#, eta_min=1e-10)

#  Training Loop (Initial Training and Domain Alignment)

for epoch in range(20):
    model.train()
    t_loss = 0
    for x1,x2,pl in train_loader:
        x1,x2,pl = x1.to(device), x2.to(device), pl.to(device)
        opt.zero_grad()

        z1 = model(x1)
        z2 = model(x2)
        e1 = F.normalize(projection_layer(z1),p=2,dim=1)
        e2 = F.normalize(projection_layer(z2),p=2,dim=1)

        # Get margins and threshold
        m1, m2 = model.get_margins()
        threshold = model.get_threshold()

        # Use thresholded contrastive loss
        cL = dual_margin_contrastive_loss(e1, e2, pl, m1, m2, threshold)

        B = e1.size(0)
        idx = torch.randperm(source_embeddings.size(0))[:B]
        src = source_embeddings[idx].mean(1)
        se = F.normalize(projection_layer(src), p=2, dim=1)

        mL = mmd_loss(se, e1)
        loss = cL + mL

        loss.backward()
        opt.step()
        t_loss += loss.item()

    model.eval(); v_loss=0
    with torch.no_grad():
        for x1,x2,pl in val_loader:
            x1,x2,pl = x1.to(device), x2.to(device), pl.to(device)
            z1 = model(x1)
            z2 = model(x2)
            e1 = F.normalize(projection_layer(z1),p=2,dim=1)
            e2 = F.normalize(projection_layer(z2),p=2,dim=1)

            m1, m2 = model.get_margins()
            threshold = model.get_threshold()

            cL = dual_margin_contrastive_loss(e1, e2, pl, m1, m2, threshold)

            B = e1.size(0)
            idx = torch.randperm(source_embeddings.size(0))[:B]
            src = source_embeddings[idx].mean(1)
            se = F.normalize(projection_layer(src), p=2, dim=1)

            mL = mmd_loss(se, e1)
            v_loss += (cL + mL).item()

    sched.step()
    print(f"Epoch {epoch+1}/20 | Train Loss: {t_loss/len(train_loader):.4f} | Val Loss: {v_loss/len(val_loader):.4f} | m1={m1:.4f}, m2={m2:.4f}, threshold={threshold:.4f}")

# Save model and projection layer
save_path = '/content/drive/MyDrive/vit_pretrained_speechcommands.pth' # Replace with your actual path
torch.save({
    'vit_model': model.state_dict(),
    'projection_layer': projection_layer.state_dict()
}, save_path)
print(f"Model saved to {save_path}")

# print(f"Max label: {max(train_ds.labels)}, Min label: {min(train_ds.labels)}")
# print(f"Unique labels: {np.unique(train_ds.labels)}")

#  Spectrogram
def iq_to_spectrogram(iq_data, nperseg=256, noverlap=128):
    I = iq_data[::2]
    Q = iq_data[1::2]
    sig = I + 1j * Q
    _, _, Zxx = stft(sig, nperseg=nperseg, noverlap=noverlap)
    return np.abs(Zxx)

#  Dataset 
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

#  Load & Process Data 
dataset_path = '/content/drive/MyDrive/dataset_training_no_aug.h5' # Replace with your actual path
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

#  Vision Transformer
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

#  Model Setup 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

img_size = (256, 65)
patch_size = (16, 8)
model = VisionTransformer(img_size, patch_size, 1, 128, depth=6, num_heads=8, mlp_dim=256, num_classes=30).to(device)

state = torch.load('/content/drive/MyDrive/vit_model_speechcommands.pth', map_location=device) # Replace with your actual path
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

#  Training (Supervised Finetuning)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=3.5e-3, weight_decay=1e-5)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=55, eta_min=1e-5)

for epoch in range(55):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for x, y in train_loader:
        x, y = x.to(device).float(), y.to(device).long()
        optimizer.zero_grad()
        logits = model(x)
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

#  Save 
torch.save(model.state_dict(), '/content/drive/MyDrive/vit_supervised_finetuned.pth') # Replace with your actual path
print("Fine-tuned model saved.")

# Test Evaluation
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        preds = torch.argmax(model(x), dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
test_acc = 100.0 * correct / total
print(f"Test Accuracy: {test_acc:.2f}%")
