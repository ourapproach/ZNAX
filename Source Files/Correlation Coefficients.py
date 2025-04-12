#Necessary imports

import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import AdamW
import os
import h5py
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
from torch.utils.data import Dataset
from torch.utils.data import random_split, DataLoader
import random
from itertools import combinations
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
from scipy.signal import stft
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import pairwise_distances

import warnings
warnings.filterwarnings("ignore")

# Mount Google Drive / --- Replace with your actual path
drive.mount('/content/drive')

# Patch Embedding
class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super().__init__()
        self.patch_size = patch_size
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
        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv_proj(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        x = torch.matmul(attn, v).transpose(1, 2).reshape(B, N, C)
        return self.out_proj(x)


# Transformer Encoder
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


# Vision Transformer
class VisionTransformer(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim, depth, num_heads, mlp_dim):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, (img_size // patch_size) ** 2 + 1, embed_dim))
        self.encoder = nn.ModuleList([TransformerEncoder(embed_dim, num_heads, mlp_dim) for _ in range(depth)])
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

# Dataset Preparation
train_transform = transforms.Compose([
    transforms.Resize((32, 32)),

    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261])
])



val_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261])
])


full_train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)


train_size = int(0.8 * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size
train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])


def create_balanced_pairs(dataset):
    """
    Create balanced pairs of positive and negative samples from a dataset.
    Args:
        dataset: PyTorch dataset object (e.g., validation dataset).
    Returns:
        list of (img1, img2, label) pairs
    """
    class_to_images = {}
    for idx, (image, label) in enumerate(dataset):
        if label not in class_to_images:
            class_to_images[label] = []
        class_to_images[label].append((image, idx))

    positive_pairs = []
    negative_pairs = []

    # Create positive pairs (within the same class)
    for class_label, images in class_to_images.items():
        if len(images) > 1:
            for (img1, _), (img2, _) in combinations(images, 2):
                positive_pairs.append((img1, img2, 1))

    # Create negative pairs (from different classes)
    all_classes = list(class_to_images.keys())
    for class1, class2 in combinations(all_classes, 2):
        for (img1, _), (img2, _) in zip(class_to_images[class1], class_to_images[class2]):
            negative_pairs.append((img1, img2, 0))

    # Balance the number of pairs
    num_pairs = min(len(positive_pairs), len(negative_pairs))
    positive_pairs = random.sample(positive_pairs, num_pairs)
    negative_pairs = random.sample(negative_pairs, num_pairs)

    all_pairs = positive_pairs + negative_pairs
    random.shuffle(all_pairs)

    return all_pairs

# Create balanced pairs for validation
validation_pairs = create_balanced_pairs(val_dataset)

# Wrap the pairs into a DataLoader
class PairDataset(torch.utils.data.Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img1, img2, label = self.pairs[idx]
        return img1, img2, label

# Create validation and test DataLoader
val_loader = DataLoader(PairDataset(validation_pairs), batch_size=64, shuffle=False)

# Train and test loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=val_transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Dual margin contrastive loss

class LearnableMargins(nn.Module):
    def __init__(self, initial_m1=0.6, initial_m2=1.6):
        super().__init__()
        self.m1 = nn.Parameter(torch.tensor(initial_m1, dtype=torch.float32))
        self.m2 = nn.Parameter(torch.tensor(initial_m2, dtype=torch.float32))




    def forward(self):
        # Ensure m1 < m2 by enforcing constraints
        return torch.clamp(self.m1, min=0), torch.clamp(self.m2, min=self.m1 + 0.1)


def dual_margin_contrastive_loss(emb1, emb2, label, m1, m2):
    emb1 = F.normalize(emb1, p=2, dim=1)
    emb2 = F.normalize(emb2, p=2, dim=1)
    D = F.pairwise_distance(emb1, emb2)
    loss_pos = label * torch.pow(torch.clamp(D - m1, min=0), 2)
    loss_neg = (1 - label) * torch.pow(torch.clamp(m2 - D, min=0), 2)
    return 30.0 * torch.mean(loss_pos + loss_neg)

#### Remove '#' and execute each model (section) one by one

# #Model Rank-1

# # Initialize device, model, and optimizer
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model = VisionTransformer(32, 4, 3, 32, 1, 32, 224).to(device)
# #(self, img_size, patch_size, in_channels, embed_dim, depth, num_heads, mlp_dim)


# # Exclude learnable_margins parameters from the main parameter group
# base_params = [
#     p for name, p in model.named_parameters() if "learnable_margins" not in name
# ]

# optimizer = AdamW(
#     [{'params': base_params},  # Regular model parameters
#      {'params': model.learnable_margins.parameters(), 'lr': 1e-3}],  # Higher learning rate for margins
#     lr=4e-4, weight_decay=1e-1
# )

# scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

# # # Store embeddings during the training loop
# # cifar_embeddings = []

# # Training Loop
# for epoch in range(30):
#     model.train()
#     total_train_loss = 0
#     for images, labels in train_loader:
#         images, labels = images.to(device), labels.to(device)
#         batch_size = images.size(0)

#         # Create pairs within the batch
#         idx = torch.randperm(batch_size)
#         img1, img2 = images, images[idx]
#         pair_labels = (labels == labels[idx]).float().to(device)

#         # Forward pass
#         emb1 = model(img1)
#         emb2 = model(img2)

#         # # Save embeddings for CIFAR10
#         # cifar_embeddings.append(emb1.cpu().detach().numpy())  # Save embeddings of img1
#         # cifar_embeddings.append(emb2.cpu().detach().numpy())  # Save embeddings of img2

#         # Get learnable margins
#         m1, m2 = model.get_margins()

#         # Compute contrastive loss
#         loss = dual_margin_contrastive_loss(emb1, emb2, pair_labels, m1, m2)

#         # Backpropagation and optimization
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         total_train_loss += loss.item()

#     scheduler.step()

#     # Validation
#     model.eval()
#     total_val_loss = 0
#     with torch.no_grad():
#         for val_img1, val_img2, val_labels in val_loader:
#             val_img1, val_img2, val_labels = (
#                 val_img1.to(device),
#                 val_img2.to(device),
#                 val_labels.to(device),
#             )

#             # Forward pass for validation
#             val_emb1 = model(val_img1)
#             val_emb2 = model(val_img2)

#             # Get learnable margins
#             val_m1, val_m2 = model.get_margins()

#             # Compute validation contrastive loss
#             val_loss = dual_margin_contrastive_loss(
#                 val_emb1, val_emb2, val_labels, val_m1, val_m2
#             )
#             total_val_loss += val_loss.item()

#     # Logging losses
#     print(
#         f"Epoch [{epoch+1}/30], Train Loss: {total_train_loss / len(train_loader):.4f}, "
#         f"Val Loss: {total_val_loss / len(val_loader):.4f}, "
#         f"m1: {m1.item():.4f}, m2: {m2.item():.4f}"
#     )

# # Model 1
# #Evaluation on Test Data
# model.eval()  # Set the model to evaluation mode
# total_test_loss = 0

# with torch.no_grad():
#     for test_images, test_labels in test_loader:
#         test_images, test_labels = test_images.to(device), test_labels.to(device)
#         batch_size = test_images.size(0)

#         # Create pairs within the test batch
#         idx = torch.randperm(batch_size)
#         test_img1, test_img2 = test_images, test_images[idx]
#         test_pair_labels = (test_labels == test_labels[idx]).float().to(device)

#         # Forward pass for test data
#         test_emb1 = model(test_img1)
#         test_emb2 = model(test_img2)

#         # Get learnable margins
#         test_m1, test_m2 = model.get_margins()

#         # Compute test contrastive loss
#         test_loss = dual_margin_contrastive_loss(test_emb1, test_emb2, test_pair_labels, test_m1, test_m2)
#         total_test_loss += test_loss.item()

# # Average test loss
# avg_test_loss = total_test_loss / len(test_loader)
# print(f"Test Loss: {avg_test_loss:.4f}, Final m1: {test_m1.item():.4f}, Final m2: {test_m2.item():.4f}")

# #Model Rank-2

# # Initialize device, model, and optimizer
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model = VisionTransformer(32, 32, 3, 32, 1, 2, 128).to(device)
# #(self, img_size, patch_size, in_channels, embed_dim, depth, num_heads, mlp_dim)


# # Exclude learnable_margins parameters from the main parameter group
# base_params = [
#     p for name, p in model.named_parameters() if "learnable_margins" not in name
# ]

# optimizer = AdamW(
#     [{'params': base_params},  # Regular model parameters
#      {'params': model.learnable_margins.parameters(), 'lr': 1e-3}],  # Higher learning rate for margins
#     lr=2e-4, weight_decay=1e-1
# )

# scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

# # # Store embeddings during the training loop
# # cifar_embeddings = []

# # Training Loop
# for epoch in range(30):
#     model.train()
#     total_train_loss = 0
#     for images, labels in train_loader:
#         images, labels = images.to(device), labels.to(device)
#         batch_size = images.size(0)

#         # Create pairs within the batch
#         idx = torch.randperm(batch_size)
#         img1, img2 = images, images[idx]
#         pair_labels = (labels == labels[idx]).float().to(device)

#         # Forward pass
#         emb1 = model(img1)
#         emb2 = model(img2)

#         # # Save embeddings for CIFAR10
#         # cifar_embeddings.append(emb1.cpu().detach().numpy())  # Save embeddings of img1
#         # cifar_embeddings.append(emb2.cpu().detach().numpy())  # Save embeddings of img2

#         # Get learnable margins
#         m1, m2 = model.get_margins()

#         # Compute contrastive loss
#         loss = dual_margin_contrastive_loss(emb1, emb2, pair_labels, m1, m2)

#         # Backpropagation and optimization
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         total_train_loss += loss.item()

#     scheduler.step()

#     # Validation
#     model.eval()
#     total_val_loss = 0
#     with torch.no_grad():
#         for val_img1, val_img2, val_labels in val_loader:
#             val_img1, val_img2, val_labels = (
#                 val_img1.to(device),
#                 val_img2.to(device),
#                 val_labels.to(device),
#             )

#             # Forward pass for validation
#             val_emb1 = model(val_img1)
#             val_emb2 = model(val_img2)

#             # Get learnable margins
#             val_m1, val_m2 = model.get_margins()

#             # Compute validation contrastive loss
#             val_loss = dual_margin_contrastive_loss(
#                 val_emb1, val_emb2, val_labels, val_m1, val_m2
#             )
#             total_val_loss += val_loss.item()

#     # Logging losses
#     print(
#         f"Epoch [{epoch+1}/30], Train Loss: {total_train_loss / len(train_loader):.4f}, "
#         f"Val Loss: {total_val_loss / len(val_loader):.4f}, "
#         f"m1: {m1.item():.4f}, m2: {m2.item():.4f}"
#     )

# # Model 2
# #Evaluation on Test Data
# model.eval()  # Set the model to evaluation mode
# total_test_loss = 0

# with torch.no_grad():
#     for test_images, test_labels in test_loader:
#         test_images, test_labels = test_images.to(device), test_labels.to(device)
#         batch_size = test_images.size(0)

#         # Create pairs within the test batch
#         idx = torch.randperm(batch_size)
#         test_img1, test_img2 = test_images, test_images[idx]
#         test_pair_labels = (test_labels == test_labels[idx]).float().to(device)

#         # Forward pass for test data
#         test_emb1 = model(test_img1)
#         test_emb2 = model(test_img2)

#         # Get learnable margins
#         test_m1, test_m2 = model.get_margins()

#         # Compute test contrastive loss
#         test_loss = dual_margin_contrastive_loss(test_emb1, test_emb2, test_pair_labels, test_m1, test_m2)
#         total_test_loss += test_loss.item()

# # Average test loss
# avg_test_loss = total_test_loss / len(test_loader)
# print(f"Test Loss: {avg_test_loss:.4f}, Final m1: {test_m1.item():.4f}, Final m2: {test_m2.item():.4f}")

# #Model Rank-3

# # Initialize device, model, and optimizer
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model = VisionTransformer(32, 16, 3, 32, 2, 32, 256).to(device)
# #(self, img_size, patch_size, in_channels, embed_dim, depth, num_heads, mlp_dim)


# # Exclude learnable_margins parameters from the main parameter group
# base_params = [
#     p for name, p in model.named_parameters() if "learnable_margins" not in name
# ]

# optimizer = AdamW(
#     [{'params': base_params},  # Regular model parameters
#      {'params': model.learnable_margins.parameters(), 'lr': 1e-3}],  # Higher learning rate for margins
#     lr=2e-4, weight_decay=1e-1
# )

# scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

# # # Store embeddings during the training loop
# # cifar_embeddings = []

# # Training Loop
# for epoch in range(30):
#     model.train()
#     total_train_loss = 0
#     for images, labels in train_loader:
#         images, labels = images.to(device), labels.to(device)
#         batch_size = images.size(0)

#         # Create pairs within the batch
#         idx = torch.randperm(batch_size)
#         img1, img2 = images, images[idx]
#         pair_labels = (labels == labels[idx]).float().to(device)

#         # Forward pass
#         emb1 = model(img1)
#         emb2 = model(img2)

#         # # Save embeddings for CIFAR10
#         # cifar_embeddings.append(emb1.cpu().detach().numpy())  # Save embeddings of img1
#         # cifar_embeddings.append(emb2.cpu().detach().numpy())  # Save embeddings of img2

#         # Get learnable margins
#         m1, m2 = model.get_margins()

#         # Compute contrastive loss
#         loss = dual_margin_contrastive_loss(emb1, emb2, pair_labels, m1, m2)

#         # Backpropagation and optimization
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         total_train_loss += loss.item()

#     scheduler.step()

#     # Validation
#     model.eval()
#     total_val_loss = 0
#     with torch.no_grad():
#         for val_img1, val_img2, val_labels in val_loader:
#             val_img1, val_img2, val_labels = (
#                 val_img1.to(device),
#                 val_img2.to(device),
#                 val_labels.to(device),
#             )

#             # Forward pass for validation
#             val_emb1 = model(val_img1)
#             val_emb2 = model(val_img2)

#             # Get learnable margins
#             val_m1, val_m2 = model.get_margins()

#             # Compute validation contrastive loss
#             val_loss = dual_margin_contrastive_loss(
#                 val_emb1, val_emb2, val_labels, val_m1, val_m2
#             )
#             total_val_loss += val_loss.item()

#     # Logging losses
#     print(
#         f"Epoch [{epoch+1}/30], Train Loss: {total_train_loss / len(train_loader):.4f}, "
#         f"Val Loss: {total_val_loss / len(val_loader):.4f}, "
#         f"m1: {m1.item():.4f}, m2: {m2.item():.4f}"
#     )

# # Model 3
# #Evaluation on Test Data
# model.eval()  # Set the model to evaluation mode
# total_test_loss = 0

# with torch.no_grad():
#     for test_images, test_labels in test_loader:
#         test_images, test_labels = test_images.to(device), test_labels.to(device)
#         batch_size = test_images.size(0)

#         # Create pairs within the test batch
#         idx = torch.randperm(batch_size)
#         test_img1, test_img2 = test_images, test_images[idx]
#         test_pair_labels = (test_labels == test_labels[idx]).float().to(device)

#         # Forward pass for test data
#         test_emb1 = model(test_img1)
#         test_emb2 = model(test_img2)

#         # Get learnable margins
#         test_m1, test_m2 = model.get_margins()

#         # Compute test contrastive loss
#         test_loss = dual_margin_contrastive_loss(test_emb1, test_emb2, test_pair_labels, test_m1, test_m2)
#         total_test_loss += test_loss.item()

# # Average test loss
# avg_test_loss = total_test_loss / len(test_loader)
# print(f"Test Loss: {avg_test_loss:.4f}, Final m1: {test_m1.item():.4f}, Final m2: {test_m2.item():.4f}")

# #Model Rank-4

# # Initialize device, model, and optimizer
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model = VisionTransformer(32, 8, 3, 192, 1, 8, 576).to(device)
# #(self, img_size, patch_size, in_channels, embed_dim, depth, num_heads, mlp_dim)


# # Exclude learnable_margins parameters from the main parameter group
# base_params = [
#     p for name, p in model.named_parameters() if "learnable_margins" not in name
# ]

# optimizer = AdamW(
#     [{'params': base_params},  # Regular model parameters
#      {'params': model.learnable_margins.parameters(), 'lr': 1e-3}],
#     lr=6e-4, weight_decay=1e-1
# )

# scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=35)

# # # Store embeddings during the training loop
# # cifar_embeddings = []

# # Training Loop
# for epoch in range(35):
#     model.train()
#     total_train_loss = 0
#     for images, labels in train_loader:
#         images, labels = images.to(device), labels.to(device)
#         batch_size = images.size(0)

#         # Create pairs within the batch
#         idx = torch.randperm(batch_size)
#         img1, img2 = images, images[idx]
#         pair_labels = (labels == labels[idx]).float().to(device)

#         # Forward pass
#         emb1 = model(img1)
#         emb2 = model(img2)

#         # # Save embeddings for CIFAR10
#         # cifar_embeddings.append(emb1.cpu().detach().numpy())  # Save embeddings of img1
#         # cifar_embeddings.append(emb2.cpu().detach().numpy())  # Save embeddings of img2

#         # Get learnable margins
#         m1, m2 = model.get_margins()

#         # Compute contrastive loss
#         loss = dual_margin_contrastive_loss(emb1, emb2, pair_labels, m1, m2)

#         # Backpropagation and optimization
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         total_train_loss += loss.item()

#     scheduler.step()

#     # Validation
#     model.eval()
#     total_val_loss = 0
#     with torch.no_grad():
#         for val_img1, val_img2, val_labels in val_loader:
#             val_img1, val_img2, val_labels = (
#                 val_img1.to(device),
#                 val_img2.to(device),
#                 val_labels.to(device),
#             )

#             # Forward pass for validation
#             val_emb1 = model(val_img1)
#             val_emb2 = model(val_img2)

#             # Get learnable margins
#             val_m1, val_m2 = model.get_margins()

#             # Compute validation contrastive loss
#             val_loss = dual_margin_contrastive_loss(
#                 val_emb1, val_emb2, val_labels, val_m1, val_m2
#             )
#             total_val_loss += val_loss.item()

#     # Logging losses
#     print(
#         f"Epoch [{epoch+1}/35], Train Loss: {total_train_loss / len(train_loader):.4f}, "
#         f"Val Loss: {total_val_loss / len(val_loader):.4f}, "
#         f"m1: {m1.item():.4f}, m2: {m2.item():.4f}"
#     )

# # Model 4
# #Evaluation on Test Data
# model.eval()  # Set the model to evaluation mode
# total_test_loss = 0

# with torch.no_grad():
#     for test_images, test_labels in test_loader:
#         test_images, test_labels = test_images.to(device), test_labels.to(device)
#         batch_size = test_images.size(0)

#         # Create pairs within the test batch
#         idx = torch.randperm(batch_size)
#         test_img1, test_img2 = test_images, test_images[idx]
#         test_pair_labels = (test_labels == test_labels[idx]).float().to(device)

#         # Forward pass for test data
#         test_emb1 = model(test_img1)
#         test_emb2 = model(test_img2)

#         # Get learnable margins
#         test_m1, test_m2 = model.get_margins()

#         # Compute test contrastive loss
#         test_loss = dual_margin_contrastive_loss(test_emb1, test_emb2, test_pair_labels, test_m1, test_m2)
#         total_test_loss += test_loss.item()

# # Average test loss
# avg_test_loss = total_test_loss / len(test_loader)
# print(f"Test Loss: {avg_test_loss:.4f}, Final m1: {test_m1.item():.4f}, Final m2: {test_m2.item():.4f}")

# #Model Rank-5

# # Initialize device, model, and optimizer
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model = VisionTransformer(32, 4, 3, 32, 1, 2, 128).to(device)
# #(self, img_size, patch_size, in_channels, embed_dim, depth, num_heads, mlp_dim)


# # Exclude learnable_margins parameters from the main parameter group
# base_params = [
#     p for name, p in model.named_parameters() if "learnable_margins" not in name
# ]

# optimizer = AdamW(
#     [{'params': base_params},  # Regular model parameters
#      {'params': model.learnable_margins.parameters(), 'lr': 1e-3}],  # Higher learning rate for margins
#     lr=1e-4, weight_decay=1e-1
# )

# scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

# # # Store embeddings during the training loop
# # cifar_embeddings = []

# # Training Loop
# for epoch in range(30):
#     model.train()
#     total_train_loss = 0
#     for images, labels in train_loader:
#         images, labels = images.to(device), labels.to(device)
#         batch_size = images.size(0)

#         # Create pairs within the batch
#         idx = torch.randperm(batch_size)
#         img1, img2 = images, images[idx]
#         pair_labels = (labels == labels[idx]).float().to(device)

#         # Forward pass
#         emb1 = model(img1)
#         emb2 = model(img2)

#         # # Save embeddings for CIFAR10
#         # cifar_embeddings.append(emb1.cpu().detach().numpy())  # Save embeddings of img1
#         # cifar_embeddings.append(emb2.cpu().detach().numpy())  # Save embeddings of img2

#         # Get learnable margins
#         m1, m2 = model.get_margins()

#         # Compute contrastive loss
#         loss = dual_margin_contrastive_loss(emb1, emb2, pair_labels, m1, m2)

#         # Backpropagation and optimization
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         total_train_loss += loss.item()

#     scheduler.step()

#     # Validation
#     model.eval()
#     total_val_loss = 0
#     with torch.no_grad():
#         for val_img1, val_img2, val_labels in val_loader:
#             val_img1, val_img2, val_labels = (
#                 val_img1.to(device),
#                 val_img2.to(device),
#                 val_labels.to(device),
#             )

#             # Forward pass for validation
#             val_emb1 = model(val_img1)
#             val_emb2 = model(val_img2)

#             # Get learnable margins
#             val_m1, val_m2 = model.get_margins()

#             # Compute validation contrastive loss
#             val_loss = dual_margin_contrastive_loss(
#                 val_emb1, val_emb2, val_labels, val_m1, val_m2
#             )
#             total_val_loss += val_loss.item()

#     # Logging losses
#     print(
#         f"Epoch [{epoch+1}/30], Train Loss: {total_train_loss / len(train_loader):.4f}, "
#         f"Val Loss: {total_val_loss / len(val_loader):.4f}, "
#         f"m1: {m1.item():.4f}, m2: {m2.item():.4f}"
#     )

# # Model 5
# #Evaluation on Test Data
# model.eval()  # Set the model to evaluation mode
# total_test_loss = 0

# with torch.no_grad():
#     for test_images, test_labels in test_loader:
#         test_images, test_labels = test_images.to(device), test_labels.to(device)
#         batch_size = test_images.size(0)

#         # Create pairs within the test batch
#         idx = torch.randperm(batch_size)
#         test_img1, test_img2 = test_images, test_images[idx]
#         test_pair_labels = (test_labels == test_labels[idx]).float().to(device)

#         # Forward pass for test data
#         test_emb1 = model(test_img1)
#         test_emb2 = model(test_img2)

#         # Get learnable margins
#         test_m1, test_m2 = model.get_margins()

#         # Compute test contrastive loss
#         test_loss = dual_margin_contrastive_loss(test_emb1, test_emb2, test_pair_labels, test_m1, test_m2)
#         total_test_loss += test_loss.item()

# # Average test loss
# avg_test_loss = total_test_loss / len(test_loader)
# print(f"Test Loss: {avg_test_loss:.4f}, Final m1: {test_m1.item():.4f}, Final m2: {test_m2.item():.4f}")

# #Model Rank-6

# # Initialize device, model, and optimizer
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model = VisionTransformer(32, 16, 3, 32, 1, 16, 64).to(device)
# #(self, img_size, patch_size, in_channels, embed_dim, depth, num_heads, mlp_dim)


# # Exclude learnable_margins parameters from the main parameter group
# base_params = [
#     p for name, p in model.named_parameters() if "learnable_margins" not in name
# ]

# optimizer = AdamW(
#     [{'params': base_params},  # Regular model parameters
#      {'params': model.learnable_margins.parameters(), 'lr': 1e-3}],
#     lr=1e-3, weight_decay=1e-1
# )

# scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25)

# # # Store embeddings during the training loop
# # cifar_embeddings = []

# # Training Loop
# for epoch in range(25):
#     model.train()
#     total_train_loss = 0
#     for images, labels in train_loader:
#         images, labels = images.to(device), labels.to(device)
#         batch_size = images.size(0)

#         # Create pairs within the batch
#         idx = torch.randperm(batch_size)
#         img1, img2 = images, images[idx]
#         pair_labels = (labels == labels[idx]).float().to(device)

#         # Forward pass
#         emb1 = model(img1)
#         emb2 = model(img2)

#         # # Save embeddings for CIFAR10
#         # cifar_embeddings.append(emb1.cpu().detach().numpy())  # Save embeddings of img1
#         # cifar_embeddings.append(emb2.cpu().detach().numpy())  # Save embeddings of img2

#         # Get learnable margins
#         m1, m2 = model.get_margins()

#         # Compute contrastive loss
#         loss = dual_margin_contrastive_loss(emb1, emb2, pair_labels, m1, m2)

#         # Backpropagation and optimization
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         total_train_loss += loss.item()

#     scheduler.step()

#     # Validation
#     model.eval()
#     total_val_loss = 0
#     with torch.no_grad():
#         for val_img1, val_img2, val_labels in val_loader:
#             val_img1, val_img2, val_labels = (
#                 val_img1.to(device),
#                 val_img2.to(device),
#                 val_labels.to(device),
#             )

#             # Forward pass for validation
#             val_emb1 = model(val_img1)
#             val_emb2 = model(val_img2)

#             # Get learnable margins
#             val_m1, val_m2 = model.get_margins()

#             # Compute validation contrastive loss
#             val_loss = dual_margin_contrastive_loss(
#                 val_emb1, val_emb2, val_labels, val_m1, val_m2
#             )
#             total_val_loss += val_loss.item()

#     # Logging losses
#     print(
#         f"Epoch [{epoch+1}/30], Train Loss: {total_train_loss / len(train_loader):.4f}, "
#         f"Val Loss: {total_val_loss / len(val_loader):.4f}, "
#         f"m1: {m1.item():.4f}, m2: {m2.item():.4f}"
#     )

# # Model 6
# #Evaluation on Test Data
# model.eval()  # Set the model to evaluation mode
# total_test_loss = 0

# with torch.no_grad():
#     for test_images, test_labels in test_loader:
#         test_images, test_labels = test_images.to(device), test_labels.to(device)
#         batch_size = test_images.size(0)

#         # Create pairs within the test batch
#         idx = torch.randperm(batch_size)
#         test_img1, test_img2 = test_images, test_images[idx]
#         test_pair_labels = (test_labels == test_labels[idx]).float().to(device)

#         # Forward pass for test data
#         test_emb1 = model(test_img1)
#         test_emb2 = model(test_img2)

#         # Get learnable margins
#         test_m1, test_m2 = model.get_margins()

#         # Compute test contrastive loss
#         test_loss = dual_margin_contrastive_loss(test_emb1, test_emb2, test_pair_labels, test_m1, test_m2)
#         total_test_loss += test_loss.item()

# # Average test loss
# avg_test_loss = total_test_loss / len(test_loader)
# print(f"Test Loss: {avg_test_loss:.4f}, Final m1: {test_m1.item():.4f}, Final m2: {test_m2.item():.4f}")

# #Model Rank-7

# # Initialize device, model, and optimizer
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model = VisionTransformer(32, 16, 3, 32, 1, 16, 96).to(device)
# #(self, img_size, patch_size, in_channels, embed_dim, depth, num_heads, mlp_dim)


# # Exclude learnable_margins parameters from the main parameter group
# base_params = [
#     p for name, p in model.named_parameters() if "learnable_margins" not in name
# ]

# optimizer = AdamW(
#     [{'params': base_params},  # Regular model parameters
#      {'params': model.learnable_margins.parameters(), 'lr': 1e-3}],  # Higher learning rate for margins
#     lr=1e-3, weight_decay=1e-1
# )

# scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25)

# # # Store embeddings during the training loop
# # cifar_embeddings = []

# # Training Loop
# for epoch in range(25):
#     model.train()
#     total_train_loss = 0
#     for images, labels in train_loader:
#         images, labels = images.to(device), labels.to(device)
#         batch_size = images.size(0)

#         # Create pairs within the batch
#         idx = torch.randperm(batch_size)
#         img1, img2 = images, images[idx]
#         pair_labels = (labels == labels[idx]).float().to(device)

#         # Forward pass
#         emb1 = model(img1)
#         emb2 = model(img2)

#         # # Save embeddings for CIFAR10
#         # cifar_embeddings.append(emb1.cpu().detach().numpy())  # Save embeddings of img1
#         # cifar_embeddings.append(emb2.cpu().detach().numpy())  # Save embeddings of img2

#         # Get learnable margins
#         m1, m2 = model.get_margins()

#         # Compute contrastive loss
#         loss = dual_margin_contrastive_loss(emb1, emb2, pair_labels, m1, m2)

#         # Backpropagation and optimization
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         total_train_loss += loss.item()

#     scheduler.step()

#     # Validation
#     model.eval()
#     total_val_loss = 0
#     with torch.no_grad():
#         for val_img1, val_img2, val_labels in val_loader:
#             val_img1, val_img2, val_labels = (
#                 val_img1.to(device),
#                 val_img2.to(device),
#                 val_labels.to(device),
#             )

#             # Forward pass for validation
#             val_emb1 = model(val_img1)
#             val_emb2 = model(val_img2)

#             # Get learnable margins
#             val_m1, val_m2 = model.get_margins()

#             # Compute validation contrastive loss
#             val_loss = dual_margin_contrastive_loss(
#                 val_emb1, val_emb2, val_labels, val_m1, val_m2
#             )
#             total_val_loss += val_loss.item()

#     # Logging losses
#     print(
#         f"Epoch [{epoch+1}/25], Train Loss: {total_train_loss / len(train_loader):.4f}, "
#         f"Val Loss: {total_val_loss / len(val_loader):.4f}, "
#         f"m1: {m1.item():.4f}, m2: {m2.item():.4f}"
#     )

# # Model 7
# #Evaluation on Test Data
# model.eval()  # Set the model to evaluation mode
# total_test_loss = 0

# with torch.no_grad():
#     for test_images, test_labels in test_loader:
#         test_images, test_labels = test_images.to(device), test_labels.to(device)
#         batch_size = test_images.size(0)

#         # Create pairs within the test batch
#         idx = torch.randperm(batch_size)
#         test_img1, test_img2 = test_images, test_images[idx]
#         test_pair_labels = (test_labels == test_labels[idx]).float().to(device)

#         # Forward pass for test data
#         test_emb1 = model(test_img1)
#         test_emb2 = model(test_img2)

#         # Get learnable margins
#         test_m1, test_m2 = model.get_margins()

#         # Compute test contrastive loss
#         test_loss = dual_margin_contrastive_loss(test_emb1, test_emb2, test_pair_labels, test_m1, test_m2)
#         total_test_loss += test_loss.item()

# # Average test loss
# avg_test_loss = total_test_loss / len(test_loader)
# print(f"Test Loss: {avg_test_loss:.4f}, Final m1: {test_m1.item():.4f}, Final m2: {test_m2.item():.4f}")

# #Model Rank-8

# # Initialize device, model, and optimizer
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model = VisionTransformer(32, 16, 3, 32, 1, 8, 160).to(device)
# #(self, img_size, patch_size, in_channels, embed_dim, depth, num_heads, mlp_dim)


# # Exclude learnable_margins parameters from the main parameter group
# base_params = [
#     p for name, p in model.named_parameters() if "learnable_margins" not in name
# ]

# optimizer = AdamW(
#     [{'params': base_params},  # Regular model parameters
#      {'params': model.learnable_margins.parameters(), 'lr': 1e-3}],
#     lr=8e-4, weight_decay=1e-1
# )

# scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25)

# # # Store embeddings during the training loop
# # cifar_embeddings = []

# # Training Loop
# for epoch in range(25):
#     model.train()
#     total_train_loss = 0
#     for images, labels in train_loader:
#         images, labels = images.to(device), labels.to(device)
#         batch_size = images.size(0)

#         # Create pairs within the batch
#         idx = torch.randperm(batch_size)
#         img1, img2 = images, images[idx]
#         pair_labels = (labels == labels[idx]).float().to(device)

#         # Forward pass
#         emb1 = model(img1)
#         emb2 = model(img2)

#         # # Save embeddings for CIFAR10
#         # cifar_embeddings.append(emb1.cpu().detach().numpy())  # Save embeddings of img1
#         # cifar_embeddings.append(emb2.cpu().detach().numpy())  # Save embeddings of img2

#         # Get learnable margins
#         m1, m2 = model.get_margins()

#         # Compute contrastive loss
#         loss = dual_margin_contrastive_loss(emb1, emb2, pair_labels, m1, m2)

#         # Backpropagation and optimization
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         total_train_loss += loss.item()

#     scheduler.step()

#     # Validation
#     model.eval()
#     total_val_loss = 0
#     with torch.no_grad():
#         for val_img1, val_img2, val_labels in val_loader:
#             val_img1, val_img2, val_labels = (
#                 val_img1.to(device),
#                 val_img2.to(device),
#                 val_labels.to(device),
#             )

#             # Forward pass for validation
#             val_emb1 = model(val_img1)
#             val_emb2 = model(val_img2)

#             # Get learnable margins
#             val_m1, val_m2 = model.get_margins()

#             # Compute validation contrastive loss
#             val_loss = dual_margin_contrastive_loss(
#                 val_emb1, val_emb2, val_labels, val_m1, val_m2
#             )
#             total_val_loss += val_loss.item()

#     # Logging losses
#     print(
#         f"Epoch [{epoch+1}/25], Train Loss: {total_train_loss / len(train_loader):.4f}, "
#         f"Val Loss: {total_val_loss / len(val_loader):.4f}, "
#         f"m1: {m1.item():.4f}, m2: {m2.item():.4f}"
#     )

# # Model 8
# #Evaluation on Test Data
# model.eval()  # Set the model to evaluation mode
# total_test_loss = 0

# with torch.no_grad():
#     for test_images, test_labels in test_loader:
#         test_images, test_labels = test_images.to(device), test_labels.to(device)
#         batch_size = test_images.size(0)

#         # Create pairs within the test batch
#         idx = torch.randperm(batch_size)
#         test_img1, test_img2 = test_images, test_images[idx]
#         test_pair_labels = (test_labels == test_labels[idx]).float().to(device)

#         # Forward pass for test data
#         test_emb1 = model(test_img1)
#         test_emb2 = model(test_img2)

#         # Get learnable margins
#         test_m1, test_m2 = model.get_margins()

#         # Compute test contrastive loss
#         test_loss = dual_margin_contrastive_loss(test_emb1, test_emb2, test_pair_labels, test_m1, test_m2)
#         total_test_loss += test_loss.item()

# # Average test loss
# avg_test_loss = total_test_loss / len(test_loader)
# print(f"Test Loss: {avg_test_loss:.4f}, Final m1: {test_m1.item():.4f}, Final m2: {test_m2.item():.4f}")

# #Model Rank-9

# # Initialize device, model, and optimizer
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model = VisionTransformer(32, 16, 3, 32, 2, 4, 192).to(device)
# #(self, img_size, patch_size, in_channels, embed_dim, depth, num_heads, mlp_dim)


# # Exclude learnable_margins parameters from the main parameter group
# base_params = [
#     p for name, p in model.named_parameters() if "learnable_margins" not in name
# ]

# optimizer = AdamW(
#     [{'params': base_params},  # Regular model parameters
#      {'params': model.learnable_margins.parameters(), 'lr': 1e-3}],  # Higher learning rate for margins
#     lr=4e-4, weight_decay=1e-1
# )

# scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25)

# # # Store embeddings during the training loop
# # cifar_embeddings = []

# # Training Loop
# for epoch in range(25):
#     model.train()
#     total_train_loss = 0
#     for images, labels in train_loader:
#         images, labels = images.to(device), labels.to(device)
#         batch_size = images.size(0)

#         # Create pairs within the batch
#         idx = torch.randperm(batch_size)
#         img1, img2 = images, images[idx]
#         pair_labels = (labels == labels[idx]).float().to(device)

#         # Forward pass
#         emb1 = model(img1)
#         emb2 = model(img2)

#         # # Save embeddings for CIFAR10
#         # cifar_embeddings.append(emb1.cpu().detach().numpy())  # Save embeddings of img1
#         # cifar_embeddings.append(emb2.cpu().detach().numpy())  # Save embeddings of img2

#         # Get learnable margins
#         m1, m2 = model.get_margins()

#         # Compute contrastive loss
#         loss = dual_margin_contrastive_loss(emb1, emb2, pair_labels, m1, m2)

#         # Backpropagation and optimization
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         total_train_loss += loss.item()

#     scheduler.step()

#     # Validation
#     model.eval()
#     total_val_loss = 0
#     with torch.no_grad():
#         for val_img1, val_img2, val_labels in val_loader:
#             val_img1, val_img2, val_labels = (
#                 val_img1.to(device),
#                 val_img2.to(device),
#                 val_labels.to(device),
#             )

#             # Forward pass for validation
#             val_emb1 = model(val_img1)
#             val_emb2 = model(val_img2)

#             # Get learnable margins
#             val_m1, val_m2 = model.get_margins()

#             # Compute validation contrastive loss
#             val_loss = dual_margin_contrastive_loss(
#                 val_emb1, val_emb2, val_labels, val_m1, val_m2
#             )
#             total_val_loss += val_loss.item()

#     # Logging losses
#     print(
#         f"Epoch [{epoch+1}/25], Train Loss: {total_train_loss / len(train_loader):.4f}, "
#         f"Val Loss: {total_val_loss / len(val_loader):.4f}, "
#         f"m1: {m1.item():.4f}, m2: {m2.item():.4f}"
#     )

# # Model 9
# #Evaluation on Test Data
# model.eval()  # Set the model to evaluation mode
# total_test_loss = 0

# with torch.no_grad():
#     for test_images, test_labels in test_loader:
#         test_images, test_labels = test_images.to(device), test_labels.to(device)
#         batch_size = test_images.size(0)

#         # Create pairs within the test batch
#         idx = torch.randperm(batch_size)
#         test_img1, test_img2 = test_images, test_images[idx]
#         test_pair_labels = (test_labels == test_labels[idx]).float().to(device)

#         # Forward pass for test data
#         test_emb1 = model(test_img1)
#         test_emb2 = model(test_img2)

#         # Get learnable margins
#         test_m1, test_m2 = model.get_margins()

#         # Compute test contrastive loss
#         test_loss = dual_margin_contrastive_loss(test_emb1, test_emb2, test_pair_labels, test_m1, test_m2)
#         total_test_loss += test_loss.item()

# # Average test loss
# avg_test_loss = total_test_loss / len(test_loader)
# print(f"Test Loss: {avg_test_loss:.4f}, Final m1: {test_m1.item():.4f}, Final m2: {test_m2.item():.4f}")

# #Model Rank-10

# # Initialize device, model, and optimizer
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model = VisionTransformer(32, 8, 3, 32, 4, 2, 256).to(device)
# #(self, img_size, patch_size, in_channels, embed_dim, depth, num_heads, mlp_dim)


# # Exclude learnable_margins parameters from the main parameter group
# base_params = [
#     p for name, p in model.named_parameters() if "learnable_margins" not in name
# ]

# optimizer = AdamW(
#     [{'params': base_params},  # Regular model parameters
#      {'params': model.learnable_margins.parameters(), 'lr': 1e-3}],  # Higher learning rate for margins
#     lr=3e-4, weight_decay=1e-1
# )

# scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25)

# # # Store embeddings during the training loop
# # cifar_embeddings = []

# # Training Loop
# for epoch in range(25):
#     model.train()
#     total_train_loss = 0
#     for images, labels in train_loader:
#         images, labels = images.to(device), labels.to(device)
#         batch_size = images.size(0)

#         # Create pairs within the batch
#         idx = torch.randperm(batch_size)
#         img1, img2 = images, images[idx]
#         pair_labels = (labels == labels[idx]).float().to(device)

#         # Forward pass
#         emb1 = model(img1)
#         emb2 = model(img2)

#         # # Save embeddings for CIFAR10
#         # cifar_embeddings.append(emb1.cpu().detach().numpy())  # Save embeddings of img1
#         # cifar_embeddings.append(emb2.cpu().detach().numpy())  # Save embeddings of img2

#         # Get learnable margins
#         m1, m2 = model.get_margins()

#         # Compute contrastive loss
#         loss = dual_margin_contrastive_loss(emb1, emb2, pair_labels, m1, m2)

#         # Backpropagation and optimization
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         total_train_loss += loss.item()

#     scheduler.step()

#     # Validation
#     model.eval()
#     total_val_loss = 0
#     with torch.no_grad():
#         for val_img1, val_img2, val_labels in val_loader:
#             val_img1, val_img2, val_labels = (
#                 val_img1.to(device),
#                 val_img2.to(device),
#                 val_labels.to(device),
#             )

#             # Forward pass for validation
#             val_emb1 = model(val_img1)
#             val_emb2 = model(val_img2)

#             # Get learnable margins
#             val_m1, val_m2 = model.get_margins()

#             # Compute validation contrastive loss
#             val_loss = dual_margin_contrastive_loss(
#                 val_emb1, val_emb2, val_labels, val_m1, val_m2
#             )
#             total_val_loss += val_loss.item()

#     # Logging losses
#     print(
#         f"Epoch [{epoch+1}/25], Train Loss: {total_train_loss / len(train_loader):.4f}, "
#         f"Val Loss: {total_val_loss / len(val_loader):.4f}, "
#         f"m1: {m1.item():.4f}, m2: {m2.item():.4f}"
#     )

# # Model 10
# #Evaluation on Test Data
# model.eval()  # Set the model to evaluation mode
# total_test_loss = 0

# with torch.no_grad():
#     for test_images, test_labels in test_loader:
#         test_images, test_labels = test_images.to(device), test_labels.to(device)
#         batch_size = test_images.size(0)

#         # Create pairs within the test batch
#         idx = torch.randperm(batch_size)
#         test_img1, test_img2 = test_images, test_images[idx]
#         test_pair_labels = (test_labels == test_labels[idx]).float().to(device)

#         # Forward pass for test data
#         test_emb1 = model(test_img1)
#         test_emb2 = model(test_img2)

#         # Get learnable margins
#         test_m1, test_m2 = model.get_margins()

#         # Compute test contrastive loss
#         test_loss = dual_margin_contrastive_loss(test_emb1, test_emb2, test_pair_labels, test_m1, test_m2)
#         total_test_loss += test_loss.item()

# # Average test loss
# avg_test_loss = total_test_loss / len(test_loader)
# print(f"Test Loss: {avg_test_loss:.4f}, Final m1: {test_m1.item():.4f}, Final m2: {test_m2.item():.4f}")

#SPR correlation coefficient

# Define the proxy ranks and test losses
proxy_ranks = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=torch.float32)  # Given
test_losses = torch.tensor([0.0001, 0.0004, 0.0012, 0.0265, 0.0213, 0.0326, 0.0322, 0.0306, 0.0336, 0.0404], dtype=torch.float32)

# Step 1: Rank the test losses
test_loss_ranks = torch.argsort(test_losses).argsort() + 1  # Adding 1 to start ranks from 1

# Step 2: Calculate the difference between ranks
differences = proxy_ranks - test_loss_ranks.float()

# Step 3: Compute the squared differences
squared_differences = differences ** 2

# Step 4: Compute Spearman's rank correlation coefficient
n = len(proxy_ranks)  # Total number of data points
spearman_coefficient = 1 - (6 * torch.sum(squared_differences) / (n * (n**2 - 1)))

# Display the ranks and the coefficient
print("Proxy Ranks:", proxy_ranks.tolist())
print("Test Loss Ranks:", test_loss_ranks.tolist())
print()
print()
print(f"Spearman's Rank Correlation Coefficient: {spearman_coefficient.item():.4f}")


## Kendall's Tau correlation coefficient

from itertools import combinations

# Define the proxy ranks and test losses
proxy_ranks = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=torch.float32)  # Given
test_losses = torch.tensor([0.0001, 0.0004, 0.0012, 0.0265, 0.0213, 0.0326, 0.0322, 0.0306, 0.0336, 0.0404], dtype=torch.float32)

# Step 1: Rank the test losses
test_loss_ranks = torch.argsort(test_losses).argsort() + 1  # Adding 1 to start ranks from 1

# Step 2: Create all pair combinations
n = len(proxy_ranks)
pairs = list(combinations(range(n), 2))  # Generate all unique index pairs

# Step 3: Calculate concordant and discordant pairs
concordant = 0
discordant = 0

for i, j in pairs:
    # Proxy rank difference
    proxy_diff = proxy_ranks[i] - proxy_ranks[j]
    # Test loss rank difference
    test_diff = test_loss_ranks[i] - test_loss_ranks[j]
    # Concordant if both differences have the same sign
    if proxy_diff * test_diff > 0:
        concordant += 1
    # Discordant if differences have opposite signs
    elif proxy_diff * test_diff < 0:
        discordant += 1

# Step 4: Compute Kendall's Tau
tau = (concordant - discordant) / (0.5 * n * (n - 1))

# Display the results
print("Proxy Ranks:", proxy_ranks.tolist())
print("Test Loss Ranks:", test_loss_ranks.tolist())
print()
print()
print(f"Kendall's Tau Correlation Coefficient: {tau:.4f}")
