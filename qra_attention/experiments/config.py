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
    batch_size: int = 32  # RESTORED (was 16)
    learning_rate: float = 2e-5  # RESTORED (was 5e-6)
    num_epochs: int = 3
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    seed: int = 42
    
    # Freezing
    freeze_embeddings: bool = True
    freeze_layers: tuple = (0, 1, 2, 3)
    
    # Paths
    output_dir: str = "results/baseline"
    logging_dir: str = "logs/baseline"