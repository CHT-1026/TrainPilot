"""Configuration boundary for the project-owned pretraining implementation."""
from dataclasses import dataclass


@dataclass
class TrainPilotconfig:
    vocab_size: int = 6400
    hidden_size: int = 128
    num_layers: int = 2
    num_heads: int = 4
    num_kv_heads: int = 2
    intermediate_size: int = 512
    max_seq_len: int = 128
    dropout: float = 0.0
    flash_attn: bool = False
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1e6
    tie_word_embeddings: bool = True

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    def __post_init__(self):
        positive = (self.vocab_size, self.hidden_size, self.num_layers, self.num_heads,
                    self.num_kv_heads, self.intermediate_size, self.max_seq_len)
        if any(value <= 0 for value in positive):
            raise ValueError("Model dimensions must be positive")
        if self.hidden_size % self.num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        if self.num_heads % self.num_kv_heads:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        if self.head_dim % 2:
            raise ValueError("RoPE requires even head_dim")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.rms_norm_eps <= 0 or self.rope_theta <= 0:
            raise ValueError("rms_norm_eps and rope_theta must be positive")
