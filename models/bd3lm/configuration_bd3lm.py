"""BD3LM config for Hugging Face.

Pure Transformer for Block Diffusion (LLaMA-style).
Ported from WindowDiffusion implementation.
"""
from transformers import AutoConfig, PretrainedConfig


class BD3LMConfig(PretrainedConfig):
    """Hugging Face configuration class for BD3LM."""
    model_type = "bd3lm"

    def __init__(
        self,
        block_size: int = 64,
        vocab_size: int = 126464,
        model_length: int = 2048,
        hidden_dim: int = 1024,
        n_heads: int = 16,
        head_dim: int = 64,
        intermediate_size: int = 4864,
        n_blocks: int = 24,
        dropout: float = 0.0,
        rope_theta: float = 10000.0,
        tie_word_embeddings: bool = False,
        attn_backend: str = 'flex',
        cross_attn: bool = True,
        var_min: bool = True,
        sampling_eps_min: float = 1e-3,
        sampling_eps_max: float = 0.999,
        **kwargs):
        super().__init__(**kwargs)
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.model_length = model_length
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.intermediate_size = intermediate_size
        self.n_blocks = n_blocks
        self.dropout = dropout
        self.rope_theta = rope_theta
        self.tie_word_embeddings = tie_word_embeddings
        self.attn_backend = attn_backend
        self.cross_attn = cross_attn
        self.var_min = var_min
        self.sampling_eps_min = sampling_eps_min
        self.sampling_eps_max = sampling_eps_max


AutoConfig.register("bd3lm", BD3LMConfig)
