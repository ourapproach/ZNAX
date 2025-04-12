#Install

pip install deepspeed
pip install torch torchvision thop
#Imports

import torch
import torch.nn as nn
from einops import rearrange
from torch.autograd import grad
from sklearn.metrics import silhouette_score
import numpy as np
import random
import torch.nn.functional as F
from torchvision import datasets, transforms
#from deepspeed.profiling.flops_profiler import get_model_profile
from thop import profile
import logging

logging.basicConfig(level=logging.ERROR)  # Show only errors

# Vision Transformer 
class ViT(nn.Module):
    def __init__(self, image_size, patch_size, embed_dim, num_layers, num_heads, mlp_ratio, dropout_rate):
        super(ViT, self).__init__()
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.embed_dim = embed_dim

        # Patch Embedding
        self.patch_embedding = nn.Linear(patch_size * patch_size * 3, embed_dim)

        # Transformer Encoder with custom MLP ratio and dropout
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=mlp_ratio * embed_dim,  
                dropout=dropout_rate  
            )
            for _ in range(num_layers)
        ])

        # CLS Token and Positional Embeddings
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.position_embeddings = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim))

        
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # Divide input image into patches
        B, C, H, W = x.shape
        x = rearrange(x, 'b c (h ph) (w pw) -> b (h w) (ph pw c)', ph=self.patch_size, pw=self.patch_size)

        
        x = self.patch_embedding(x)

        # Add CLS token and position embeddings
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.position_embeddings

        
        x = self.dropout(x)

        
        for layer in self.layers:
            x = layer(x)

        
        cls_output = x[:, 0]
        return cls_output

#Search Space
def get_vit_architecture():
    search_space = {
        "patch_size": [4, 8, 16],# 32],

        "embed_dim": [32, 64, 128, 192, 256],#, 320],# 384], #, 448, 512],



        "num_layers": list(range(3, 9)),

        "num_heads": [2, 4, 8, 16], #, 32], #embed_dim/num_heads=0

        "mlp_ratio": [2, 3, 4, 5, 6],

        "dropout_rate": [0.0, 0.1, 0.2, 0.3, 0.4]
    }

    # Select random values from the search space
    architecture = {key: random.choice(values) for key, values in search_space.items()}

    # Ensure embed_dim is divisible by num_heads
    while architecture["embed_dim"] % architecture["num_heads"] != 0:
        # Re-select num_heads if it doesn't divide evenly
        architecture["num_heads"] = random.choice(search_space["num_heads"])

    #print(f"Selected Architecture: {architecture}")  # Debugging step

    return architecture
  class DMCLLoss(torch.nn.Module):
    def __init__(self, m1=0.0, m2=0.6, scale_factor=30.0):
        super(DMCLLoss, self).__init__()
        self.m1 = m1  
        self.m2 = m2  
        self.scale_factor = scale_factor  #Optional

    def forward(self, features, labels):
        # Features: (batch_size, embed_dim)
        # Labels: (batch_size)

        # Normalize the feature vectors to unit length
        features = F.normalize(features, p=2, dim=1)

        # Compute pairwise Euclidean distance
        dist_matrix = torch.cdist(features, features, p=2)  

        # Calculate mask for positive and negative pairs
        mask_pos = (labels.unsqueeze(1) == labels.unsqueeze(0))  
        mask_neg = ~mask_pos  # True for different class

        # Positive pairs: minimize the distance to be smaller than m1
        loss_pos = mask_pos * torch.pow(torch.clamp(dist_matrix - self.m1, min=0), 2)

        # Negative pairs: maximize the distance to be larger than m2
        loss_neg = mask_neg * torch.pow(torch.clamp(self.m2 - dist_matrix, min=0), 2)

        
        loss = self.scale_factor * torch.mean(loss_pos + loss_neg)

        return loss



# Zero-cost proxy calculation 
def compute_proxies(model, data_loader, device, m1=0.0, m2=0.6):
    model.to(device)
    model.train()
    dmcl_loss = 0.0
    avg_grad_norm = 0.0
    attention_entropies = []
    embeddings = []
    labels = []

    
    dmcl_loss_fn = DMCLLoss(m1=m1, m2=m2)

    
    for i, (images, targets) in enumerate(data_loader):
        images, targets = images.to(device), targets.to(device)

        
        random_targets = torch.randint(0, 10, size=targets.shape, device=device)

        
        outputs = model(images)

        # Proxy-1: Discriminability (DMCL Loss)
        loss = dmcl_loss_fn(outputs, random_targets)
        dmcl_loss += loss.item()

        # Proxy-2: Trainability (Backward pass to compute gradients)
        gradients = grad(loss, model.parameters(), retain_graph=True, create_graph=False)
        avg_grad_norm += sum(g.norm().item() for g in gradients if g is not None)

        # Proxy-3: Cohesiveness (Silhouette Score -- clusters embeddings to assess separation)
        embeddings.append(outputs.detach().cpu().numpy())
        labels.append(random_targets.cpu().numpy())

        # Proxy-4: Diversity (Attention Entropy -- diversity of attention output)
        probs = torch.softmax(outputs, dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1).mean().item()
        attention_entropies.append(entropy)

        break  # Evaluate only on one batch for proxy calculations

    # Final proxy scores
    avg_grad_norm /= len(gradients)  # Normalize average gradient norm
    silhouette = silhouette_score(np.vstack(embeddings), np.hstack(labels))
    attention_entropy = np.mean(attention_entropies)

    return {
        "DMCL_Loss": dmcl_loss,
        "Avg_Grad_Norm": avg_grad_norm,
        "Silhouette_Score": silhouette,
        "Attention_Entropy": attention_entropy
    }
  
# Non-linear Aggregation Function
def non_linear_aggregation(metrics, m):
    """
    metrics: List of proxy values (DMCL_Loss, Avg_Grad_Norm, Silhouette_Score, Attention_Entropy)
    m: Total number of architectures
    Returns: Non-linear aggregated score
    """
    # Rank the metrics (lower is better for DMCL_Loss and Avg_Grad_Norm)
    ranked_metrics = {k: sorted(metrics, key=lambda x: x[k]) for k in metrics[0].keys()}

    scores = []
    for idx, metric in enumerate(metrics):
        rank_sum = 0
        for k, ranks in ranked_metrics.items():
            rank = ranks.index(metric) + 1  # Rank starts from 1
            rank_sum += np.log(rank / m)  # Logarithmic rank normalization
        scores.append(rank_sum)
    return scores

### Remove '#' and execute each section

######### FLOPs Constraints


# # Function to count trainable parameters
# def count_parameters(model):
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)

# # Function to compute FLOPs using thop
# def compute_flops(model, input_res=32):
#     device = next(model.parameters()).device
#     dummy_input = torch.randn(1, 3, input_res, input_res).to(device)
#     macs, _ = profile(model, inputs=(dummy_input,), verbose=False)
#     return 2 * macs  # Convert MACs to FLOPs

# # Mutation function with FLOPs constraint
# def mutate_architecture(parent_arch, flops_budget=None, max_retries=5):
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     mutation = parent_arch.copy()

#     for _ in range(max_retries):
#         mutation_choice = random.choice(["patch_size", "embed_dim", "num_layers", "num_heads", "mlp_ratio", "dropout_rate"])

#         if mutation_choice == "patch_size":
#             mutation["patch_size"] = random.choice([4, 8, 16])
#         elif mutation_choice == "embed_dim":
#             mutation["embed_dim"] = random.choice([32, 64, 128, 192, 256])
#         elif mutation_choice == "num_layers":
#             mutation["num_layers"] = random.choice(list(range(3, 9)))
#         elif mutation_choice == "num_heads":
#             embed_dim = mutation["embed_dim"]
#             valid_heads = [i for i in [2, 4, 8, 16] if embed_dim % i == 0]
#             mutation["num_heads"] = random.choice(valid_heads) if valid_heads else random.choice([2, 4, 8, 16])
#         elif mutation_choice == "mlp_ratio":
#             mutation["mlp_ratio"] = random.choice(list(range(2, 6)))
#         elif mutation_choice == "dropout_rate":
#             mutation["dropout_rate"] = random.choice([0.0, 0.1, 0.2, 0.3, 0.4])

#         # Compute FLOPs and check if within budget
#         model = ViT(
#             image_size=32,
#             patch_size=mutation["patch_size"],
#             embed_dim=mutation["embed_dim"],
#             num_layers=mutation["num_layers"],
#             num_heads=mutation["num_heads"],
#             mlp_ratio=mutation["mlp_ratio"],
#             dropout_rate=mutation["dropout_rate"],
#         ).to(device)

#         mutated_flops = compute_flops(model)
#         del model
#         torch.cuda.empty_cache()

#         if not flops_budget or mutated_flops <= flops_budget:
#             return mutation  # Valid mutation found

#     return parent_arch  # Return parent if no valid mutation found

# # Evolutionary search with FLOPs constraint and multi-objective sorting
# def evolutionary_search(data_loader, generations=50, population_size=200, flops_budget=1e9):
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     population = [get_vit_architecture() for _ in range(population_size)]

#     for generation in range(generations):
#         metrics, param_counts, flops_counts = [], [], []

#         for arch in population:
#             model = ViT(
#                 image_size=32,
#                 patch_size=arch["patch_size"],
#                 embed_dim=arch["embed_dim"],
#                 num_layers=arch["num_layers"],
#                 num_heads=arch["num_heads"],
#                 mlp_ratio=arch["mlp_ratio"],
#                 dropout_rate=arch["dropout_rate"],
#             ).to(device)

#             proxies = compute_proxies(model, data_loader, device)
#             metrics.append(proxies)
#             param_counts.append(count_parameters(model))
#             flops_counts.append(compute_flops(model))

#             del model
#             torch.cuda.empty_cache()

#         # Compute scores and sort architectures
#         scores = non_linear_aggregation(metrics, population_size)
#         scored_population = list(zip(scores, population, param_counts, flops_counts))

#         # Step 1: Sort by performance score (ascending, lower is better)
#         scored_population.sort(key=lambda x: x[0])

#         # Step 2: Sort by FLOPs for architectures with similar scores
#         sorted_population = sorted(scored_population, key=lambda x: (x[0], x[3]))  # (score, FLOPs)

#         # Enforce FLOPs budget
#         if flops_budget:
#             sorted_population = [(s, a, p, f) for s, a, p, f in sorted_population if f <= flops_budget]
#             if not sorted_population:
#                 print(f" No architectures found within FLOPs budget of {flops_budget}! Consider increasing the limit.")
#                 break

#         # Print best architecture
#         best_score, best_arch, best_params, best_flops = sorted_population[0]
#         print(f"Generation {generation + 1}: Best Architecture - {best_arch}")
#         print(f"   Score: {best_score}, Parameters: {best_params}, FLOPs: {best_flops}\n")

#         # Select top architectures for the next generation
#         top_archs = sorted_population[:population_size // 2]
#         new_population = [arch for _, arch, _, _ in top_archs]

#         # Apply mutation while keeping FLOPs budget
#         while len(new_population) < population_size:
#             mutated_arch = mutate_architecture(random.choice(top_archs)[1], flops_budget)
#             new_population.append(mutated_arch)

#         population = new_population

#     # Get final top architectures
#     top_architectures = sorted_population[:20]
#     print("\nTop Architectures from Last Generation:")
#     for i, (score, arch, params, flops) in enumerate(top_architectures, 1):
#         print(f"Architecture {i}:")
#         print(f"   Score: {score}")
#         print(f"   Parameters: {params}")
#         print(f"   FLOPs: {flops}")
#         for key, value in arch.items():
#             print(f"   {key}: {value}")
#         print()

#     return top_architectures

# ## Without FLOPs constraint

# # Function to count trainable parameters
# def count_parameters(model):
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)

# # Function to compute FLOPs using thop
# def compute_flops(model, input_res=32):
#     device = next(model.parameters()).device
#     dummy_input = torch.randn(1, 3, input_res, input_res).to(device)
#     macs, _ = profile(model, inputs=(dummy_input,), verbose=False)
#     return 2 * macs  # Convert MACs to FLOPs

# # Mutation function (No FLOPs constraint)
# def mutate_architecture(parent_arch, max_retries=5):
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     mutation = parent_arch.copy()

#     for _ in range(max_retries):
#         mutation_choice = random.choice(["patch_size", "embed_dim", "num_layers", "num_heads", "mlp_ratio", "dropout_rate"])

#         if mutation_choice == "patch_size":
#             mutation["patch_size"] = random.choice([4, 8, 16])
#         elif mutation_choice == "embed_dim":
#             mutation["embed_dim"] = random.choice([32, 64, 128, 192, 256])
#         elif mutation_choice == "num_layers":
#             mutation["num_layers"] = random.choice(list(range(3, 9)))
#         elif mutation_choice == "num_heads":
#             embed_dim = mutation["embed_dim"]
#             valid_heads = [i for i in [2, 4, 8, 16] if embed_dim % i == 0]
#             mutation["num_heads"] = random.choice(valid_heads) if valid_heads else random.choice([2, 4, 8, 16])
#         elif mutation_choice == "mlp_ratio":
#             mutation["mlp_ratio"] = random.choice(list(range(2, 6)))
#         elif mutation_choice == "dropout_rate":
#             mutation["dropout_rate"] = random.choice([0.0, 0.1, 0.2, 0.3, 0.4])

#         return mutation  # Accept any mutation

#     return parent_arch  # Return parent if no valid mutation found

# # Evolutionary search (No FLOPs constraint)
# def evolutionary_search(data_loader, generations=20, population_size=30):
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     population = [get_vit_architecture() for _ in range(population_size)]

#     for generation in range(generations):
#         metrics, param_counts, flops_counts = [], [], []

#         for arch in population:
#             model = ViT(
#                 image_size=32,
#                 patch_size=arch["patch_size"],
#                 embed_dim=arch["embed_dim"],
#                 num_layers=arch["num_layers"],
#                 num_heads=arch["num_heads"],
#                 mlp_ratio=arch["mlp_ratio"],
#                 dropout_rate=arch["dropout_rate"],
#             ).to(device)

#             proxies = compute_proxies(model, data_loader, device)
#             metrics.append(proxies)
#             param_counts.append(count_parameters(model))
#             flops_counts.append(compute_flops(model))

#             del model
#             torch.cuda.empty_cache()

#         # Compute scores and sort architectures (No FLOPs filtering)
#         scores = non_linear_aggregation(metrics, population_size)
#         scored_population = list(zip(scores, population, param_counts, flops_counts))
#         scored_population.sort(key=lambda x: x[0])

#         # Print best architecture
#         best_score, best_arch, best_params, best_flops = scored_population[0]
#         print(f"Generation {generation + 1}: Best Architecture - {best_arch}")
#         print(f"   Score: {best_score}, Parameters: {best_params}, FLOPs: {best_flops}\n")

#         # Select top architectures for the next generation
#         top_archs = scored_population[:population_size // 2]
#         new_population = [arch for _, arch, _, _ in top_archs]
#         while len(new_population) < population_size:
#             mutated_arch = mutate_architecture(random.choice(top_archs)[1])
#             new_population.append(mutated_arch)

#         population = new_population

#     # Get final top architectures
#     top_architectures = scored_population[:20]
#     print("\nTop Architectures from Last Generation:")
#     for i, (score, arch, params, flops) in enumerate(top_architectures, 1):
#         print(f"Architecture {i}:")
#         print(f"   Score: {score}")
#         print(f"   Parameters: {params}")
#         print(f"   FLOPs: {flops}")
#         for key, value in arch.items():
#             print(f"   {key}: {value}")
#         print()

#     return top_architectures

### Pareto Optimality

# Function to count trainable parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Function to compute FLOPs using thop
def compute_flops(model, input_res=32):
    device = next(model.parameters()).device
    dummy_input = torch.randn(1, 3, input_res, input_res).to(device)
    macs, _ = profile(model, inputs=(dummy_input,), verbose=False)
    return 2 * macs  # Convert MACs to FLOPs

# Mutation function with FLOPs constraint
def mutate_architecture(parent_arch, flops_budget=None, max_retries=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mutation = parent_arch.copy()

    for _ in range(max_retries):
        mutation_choice = random.choice(["patch_size", "embed_dim", "num_layers", "num_heads", "mlp_ratio", "dropout_rate"])

        if mutation_choice == "patch_size":
            mutation["patch_size"] = random.choice([4, 8, 16])
        elif mutation_choice == "embed_dim":
            mutation["embed_dim"] = random.choice([32, 64, 128, 192, 256])
        elif mutation_choice == "num_layers":
            mutation["num_layers"] = random.choice(list(range(3, 9)))
        elif mutation_choice == "num_heads":
            embed_dim = mutation["embed_dim"]
            valid_heads = [i for i in [2, 4, 8, 16] if embed_dim % i == 0]
            mutation["num_heads"] = random.choice(valid_heads) if valid_heads else random.choice([2, 4, 8, 16])
        elif mutation_choice == "mlp_ratio":
            mutation["mlp_ratio"] = random.choice(list(range(2, 6)))
        elif mutation_choice == "dropout_rate":
            mutation["dropout_rate"] = random.choice([0.0, 0.1, 0.2, 0.3, 0.4])

        # Compute FLOPs and check if within budget
        model = ViT(
            image_size=32,
            patch_size=mutation["patch_size"],
            embed_dim=mutation["embed_dim"],
            num_layers=mutation["num_layers"],
            num_heads=mutation["num_heads"],
            mlp_ratio=mutation["mlp_ratio"],
            dropout_rate=mutation["dropout_rate"],
        ).to(device)

        mutated_flops = compute_flops(model)
        del model
        torch.cuda.empty_cache()

        if not flops_budget or mutated_flops <= flops_budget:
            return mutation  # Valid mutation found

    return parent_arch  # Return parent if no valid mutation found

# Pareto selection with near-optimal solutions
def select_pareto_front(scored_population, epsilon=0.2, min_archs=20):
    pareto_front = []
    for candidate in scored_population:
        score_c, arch_c, params_c, flops_c = candidate
        is_dominated = False

        for other in scored_population:
            score_o, arch_o, params_o, flops_o = other
            if (score_o < score_c and flops_o <= flops_c * (1 + epsilon)) or (flops_o < flops_c and score_o <= score_c * (1 + epsilon)):
                is_dominated = True
                break

        if not is_dominated:
            pareto_front.append(candidate)

    # Ensure at least min_archs architectures are selected
    if len(pareto_front) < min_archs:
        pareto_front.extend(scored_population[:min_archs - len(pareto_front)])

    return pareto_front

# Evolutionary search with FLOPs constraint
def evolutionary_search(data_loader, generations=50, population_size=2000, flops_budget=1e9):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    population = [get_vit_architecture() for _ in range(population_size)]

    for generation in range(generations):
        metrics, param_counts, flops_counts = [], [], []

        for arch in population:
            model = ViT(
                image_size=32,
                patch_size=arch["patch_size"],
                embed_dim=arch["embed_dim"],
                num_layers=arch["num_layers"],
                num_heads=arch["num_heads"],
                mlp_ratio=arch["mlp_ratio"],
                dropout_rate=arch["dropout_rate"],
            ).to(device)

            proxies = compute_proxies(model, data_loader, device)
            metrics.append(proxies)
            param_counts.append(count_parameters(model))
            flops_counts.append(compute_flops(model))

            del model
            torch.cuda.empty_cache()

        # Compute scores and sort architectures
        scores = non_linear_aggregation(metrics, population_size)
        scored_population = list(zip(scores, population, param_counts, flops_counts))
        scored_population.sort(key=lambda x: x[0])

        # Select near-optimal Pareto front
        selected_population = select_pareto_front(scored_population, epsilon=0.2, min_archs=20)

        # Print best architecture
        best_score, best_arch, best_params, best_flops = selected_population[0]
        print(f"Generation {generation + 1}: Best Architecture - {best_arch}")
        print(f"   Score: {best_score}, Parameters: {best_params}, FLOPs: {best_flops}\n")

        # Select top architectures for the next generation
        new_population = [arch for _, arch, _, _ in selected_population]
        while len(new_population) < population_size:
            mutated_arch = mutate_architecture(random.choice(new_population), flops_budget)
            new_population.append(mutated_arch)

        population = new_population

    # Get final top architectures
    top_architectures = selected_population[:20]
    print("\nTop Architectures from Last Generation:")
    for i, (score, arch, params, flops) in enumerate(top_architectures, 1):
        print(f"Architecture {i}:")
        print(f"   Score: {score}")
        print(f"   Parameters: {params}")
        print(f"   FLOPs: {flops}")
        for key, value in arch.items():
            print(f"   {key}: {value}")
        print()

    return top_architectures
# Load CIFAR-10 Dataset
data_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10(
        root=".",
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261])
        ])
    ),
    batch_size=64,
    shuffle=True
)

# Perform Evolutionary Search
best_architecture = evolutionary_search(data_loader)
#print("Best Architecture Found:", best_architecture)
