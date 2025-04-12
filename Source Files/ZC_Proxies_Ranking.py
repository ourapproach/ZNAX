#Necessary imports

import torch
import torch.nn as nn
from einops import rearrange
from torch.autograd import grad
from sklearn.metrics import silhouette_score
import numpy as np
import random
import torch.nn.functional as F
from torchvision import datasets, transforms

# Vision Transformer 
class ViT(nn.Module):
    def __init__(self, image_size, patch_size, embed_dim, num_layers, num_heads, mlp_ratio, dropout_rate):
        super(ViT, self).__init__()
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.embed_dim = embed_dim

        
        self.patch_embedding = nn.Linear(patch_size * patch_size * 3, embed_dim)

        
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=mlp_ratio * embed_dim,  
                dropout=dropout_rate  
            )
            for _ in range(num_layers)
        ])

        
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.position_embeddings = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim))

        
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        
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

#Defining search space
def get_vit_architecture():
    search_space = {
        "patch_size": [4, 8, 16], #, 32],

        "embed_dim": [32, 64, 128, 192, 256], #, 320, 384, 448, 512],



        "num_layers": list(range(3, 9)),

        "num_heads": [2, 4, 8, 16], #, 32], 

        "mlp_ratio": list(range(2, 3, 4, 5, 6)),

        "dropout_rate": [0.0, 0.1, 0.2, 0.3, 0.4]
    }

    
    architecture = {key: random.choice(values) for key, values in search_space.items()}

    # Ensure embed_dim is divisible by num_heads
    while architecture["embed_dim"] % architecture["num_heads"] != 0:
        # Re-select num_heads if it doesn't divide evenly
        architecture["num_heads"] = random.choice(search_space["num_heads"])

    #print(f"Selected Architecture: {architecture}")  # Debugging step

    return architecture

#Dual margin contrastive loss
class DMCLLoss(torch.nn.Module):
    def __init__(self, m1=0.00, m2=0.6, scale_factor=30.0):
        super(DMCLLoss, self).__init__()
        self.m1 = m1  
        self.m2 = m2  
        self.scale_factor = scale_factor  

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

        # Positive pairs --- minimize the distance to be smaller than m1
        loss_pos = mask_pos * torch.pow(torch.clamp(dist_matrix - self.m1, min=0), 2)

        # Negative pairs --- maximize the distance to be larger than m2
        loss_neg = mask_neg * torch.pow(torch.clamp(self.m2 - dist_matrix, min=0), 2)

        # Final DMCL loss (including optional scaling factor)
        loss = self.scale_factor * torch.mean(loss_pos + loss_neg)

        return loss



# Computing zero-cost proxies
def compute_proxies(model, data_loader, device, m1=0.2, m2=0.4):
    model.to(device)
    model.train()
    dmcl_loss = 0.0
    avg_grad_norm = 0.0
    attention_entropies = []
    embeddings = []
    labels = []

    # Create the DMCL loss function instance
    dmcl_loss_fn = DMCLLoss(m1=m1, m2=m2)

    # Fetch a single batch for proxy calculation
    for i, (images, targets) in enumerate(data_loader):
        images, targets = images.to(device), targets.to(device)

        # Create random targets for DMCL calculation (using targets from the dataset to keep it coherent)
        random_targets = torch.randint(0, 10, size=targets.shape, device=device)

        # Forward pass through the model
        outputs = model(images)

        # Proxy-1: Discriminability
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

# Evolutionary Search 
def evolutionary_search(data_loader, generations=50, population_size=2000):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    population = [get_vit_architecture() for _ in range(population_size)]  # Generate initial population

    for generation in range(generations):
        metrics = []
        for arch in population:
            # Build the model using the current architecture
            model = ViT(
                image_size=32,
                patch_size=arch["patch_size"],
                embed_dim=arch["embed_dim"],
                num_layers=arch["num_layers"],
                num_heads=arch["num_heads"],
                mlp_ratio=arch["mlp_ratio"],  
                dropout_rate=arch["dropout_rate"],  
            )
            proxies = compute_proxies(model, data_loader, device)
            metrics.append(proxies)

        # Non-linear aggregation to compute scores
        scores = non_linear_aggregation(metrics, population_size)

        # Combine architectures with their scores
        scored_population = list(zip(scores, population))
        scored_population.sort(key=lambda x: x[0])  # Lower score is better

        # Print best architecture at each generation
        print(f"Generation {generation + 1}: Best Architecture - {scored_population[0][1]}")

        # Select top architectures for the next generation
        top_archs = scored_population[:population_size // 2]

        # Generate new population by mutation
        new_population = [arch for _, arch in top_archs]

        
        while len(new_population) < population_size:
            
            mutated_arch = mutate_architecture(random.choice(top_archs)[1], get_vit_architecture)
            new_population.append(mutated_arch)

        population = new_population

    # Get the top 10 architectures from the final generation
    top_10_architectures = scored_population[:10]

    
    print("\nTop 10 Architectures from Last Generation:")
    for i, (score, arch) in enumerate(top_10_architectures, 1):
        print(f"Architecture {i}:")
        print(f"Score: {score}")
        for key, value in arch.items():
            print(f"{key}: {value}")
        print()  

    return top_10_architectures


def mutate_architecture(parent_arch, arch_space_func):
    
    mutation = parent_arch.copy()

    
    mutation_choice = random.choice(["patch_size", "embed_dim", "num_layers", "num_heads", "mlp_ratio", "dropout_rate"])

    if mutation_choice == "patch_size":
        mutation["patch_size"] = random.choice([4, 8, 16, 32])
    elif mutation_choice == "embed_dim":
        mutation["embed_dim"] = random.choice([32, 64, 128, 192, 256, 320, 384, 448, 512])
    elif mutation_choice == "num_layers":

        #mutation["num_layers"] = random.choice([4, 6, 8, 10, 12])
        mutation["num_layers"] = random.choice(list(range(1, 9)))

    elif mutation_choice == "num_heads":
        # Ensure that embed_dim is divisible by num_heads
        embed_dim = mutation["embed_dim"]
        #valid_heads = [i for i in [2, 4, 6, 8, 10, 12] if embed_dim % i == 0]
        valid_heads = [i for i in [2, 4, 8, 16, 32] if embed_dim % i == 0]
        mutation["num_heads"] = random.choice(valid_heads) if valid_heads else random.choice([2, 4, 8, 16, 32])
    elif mutation_choice == "mlp_ratio":
        mutation["mlp_ratio"] = random.choice(list(range(1, 9)))
    elif mutation_choice == "dropout_rate":
        mutation["dropout_rate"] = random.choice([0.0, 0.1, 0.2, 0.3, 0.4])

    # # Debugging: Ensure embed_dim % num_heads == 0
    # if mutation["embed_dim"] % mutation["num_heads"] != 0:
    #     print(f"Mutation Error: embed_dim {mutation['embed_dim']} is not divisible by num_heads {mutation['num_heads']}")

    return mutation

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

#Evolutionary search --- call
best_architecture = evolutionary_search(data_loader)
#print("Best Architecture Found:", best_architecture)
