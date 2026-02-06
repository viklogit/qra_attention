"""
Configuration for QRA Attention experiments.
"""

from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ExperimentConfig:
    # Model
    model_name: str = "distilbert-base-uncased"
    max_length: int = 256
    num_labels: int = 2
    
    # Training
    batch_size: int = 16  # Reduced for T4 stability with RFF
    learning_rate: float = 5e-6  # Lowered specifically for RFF stability
    num_epochs: int = 3
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    seed: int = 42
    
    # Freezing
    freeze_embeddings: bool = True
    freeze_layers: tuple = (0, 1, 2, 3)  # Layers to freeze
    
    # Paths
    output_dir: str = "results/baseline"
    logging_dir: str = "logs/baseline"
