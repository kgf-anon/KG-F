## Training Hyperparameters


### Table 1. Hyperparameter configurations for multi-agent GNN (top) and single-agent GNN baseline (bottom).  

| **Component** | **Hyperparameter Value** |
|---------------|---------------------------|
| **Multi-Agent GNN** | |
| *GATv2 Agent (each)* | |
| Hidden dimension (per head) | 32 |
| Number of heads | 4 |
| Intermediate GAT layer | 1 |
| Output embedding dimension | 64 |
| Dropout rate | 0.2 |
| Activation function | LeakyReLU |
| Normalization layers | BatchNorm + LayerNorm |
| *Fusion MLP (meta-agent)* | |
| Input dimension | 192 (3 agents × 64 each) |
| Hidden layers | [256, 128, 64] |
| Output dimension | 8 (target nutrients) |
| Dropout rate | 0.2 |
| *Optimization* | |
| Optimizer | AdamW |
| Learning rate | 5 × 10⁻⁴ |
| Weight decay | 1 × 10⁻⁴ |
| Loss function | Smooth L1 loss (δ = 1.0) |
| Scheduler | ReduceLROnPlateau |
| Gradient clipping | max_norm = 0.5 |
| Epochs | 500 |
| Early stopping patience | 40 epochs |
| **Single-Agent GNN** | |
| *GATv2 Single Model* | |
| Input dimension | 12+ (soil + weather + crop) |
| Hidden dimension (per head) | 32 |
| Number of heads | 4 |
| Output embedding dimension | 64 |
| Dropout rate | 0.2 |
| Activation function | LeakyReLU |
| Normalization | BatchNorm only |
| *Optimization (same)* | Shared with above |

---

**Note:** The single-agent baseline processes all heterogeneous features jointly in one GATv2 model, leading to feature interference and limited generalization due to domain entanglement (e.g., soil vs. weather). The multi-agent architecture assigns specialized agents to each domain (soil, crop, weather), enabling targeted representation learning and robust fusion through an MLP. Both models use the same per-layer capacity (32×4 = 128 parameters) to ensure fair comparison.
