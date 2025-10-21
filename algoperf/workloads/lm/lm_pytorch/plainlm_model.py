"""
Originally based on the plainLM codebase:
https://github.com/Niccolo-Ajroldi/plainLM
under the MIT license https://github.com/Niccolo-Ajroldi/plainLM/blob/main/LICENSE.
"""

import math
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class ModelConfig:
  vocab_size: int
  seq_len: int
  dim: int
  expand: float
  n_layers: int
  n_heads: int
  rmsnorm_eps: float = 1e-6
  tie_embeddings: bool = True


class MLP(nn.Module):
  def __init__(self, dim: int, hidden_dim: int, multiple_of: int = 256):
    super().__init__()
    hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
    self.fc1 = nn.Linear(dim, 2 * hidden_dim, bias=False)
    self.fc2 = nn.Linear(hidden_dim, dim, bias=False)
    self.glu = nn.GLU(dim=2)

    # Initialize with Xavier uniform
    nn.init.xavier_uniform_(self.fc1.weight)
    nn.init.xavier_uniform_(self.fc2.weight)

  def forward(self, x):
    # x: (bsz, T, dim)
    return self.fc2(self.glu(self.fc1(x)))


def precompute_freqs_cis(
  dim: int, end: int, theta: float = 10000.0, condense_ratio: int = 1
):
  inv_freqs = 1.0 / (
    theta
    ** (
      torch.arange(0, dim, 2, dtype=torch.float32, device=torch.device('cpu'))
      / dim
    )
  )
  t = (
    torch.arange(end, dtype=torch.float32, device=inv_freqs.device)
    / condense_ratio
  )
  freqs = torch.outer(t, inv_freqs).float()
  return torch.stack(
    [torch.cos(freqs)[None, :, None, :], torch.sin(freqs)[None, :, None, :]],
    dim=4,
  )


def apply_rotary_emb_complex_like(
  q: torch.Tensor, k: torch.Tensor, freqs_cis: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
  # Rotate query and key vectors using RoPE
  qk_r2 = torch.cat([q, k], dim=2).unflatten(dim=-1, sizes=(-1, 2)).float()
  rotated_qk_r2 = torch.stack(
    [
      qk_r2[..., 0] * freqs_cis[..., 0] - qk_r2[..., 1] * freqs_cis[..., 1],
      qk_r2[..., 1] * freqs_cis[..., 0] + qk_r2[..., 0] * freqs_cis[..., 1],
    ],
    -1,
  ).flatten(3)
  rotated_qk = rotated_qk_r2
  return torch.split(rotated_qk.type_as(q), q.shape[2], dim=2)


class Attention(nn.Module):
  def __init__(self, cfg: ModelConfig):
    super().__init__()
    assert cfg.dim % cfg.n_heads == 0
    self.dim = cfg.dim
    self.n_heads = cfg.n_heads
    self.head_dim = cfg.dim // cfg.n_heads

    self.w_qkv = nn.Linear(cfg.dim, 3 * cfg.dim, bias=False)
    self.w_out = nn.Linear(cfg.dim, cfg.dim, bias=False)

  def forward(self, x, freqs_cis):
    bsz, seqlen, d = x.shape  # (bsz, seqlen, d)

    q, k, v = self.w_qkv(x).split(d, dim=2)  # (bsz, seqlen, d)
    q = q.view(
      bsz, seqlen, self.n_heads, self.head_dim
    )  # (bsz, seqlen, nh, h_dim)
    k = k.view(
      bsz, seqlen, self.n_heads, self.head_dim
    )  # (bsz, seqlen, nh, h_dim)
    v = v.view(
      bsz, seqlen, self.n_heads, self.head_dim
    )  # (bsz, seqlen, nh, h_dim)

    q, k = apply_rotary_emb_complex_like(
      q, k, freqs_cis=freqs_cis
    )  # (bsz, seqlen, nh, h_dim)

    q = q.transpose(1, 2)  # (bsz, nh, seqlen, h_dim)
    k = k.transpose(1, 2)  # (bsz, nh, seqlen, h_dim)
    v = v.transpose(1, 2)  # (bsz, nh, seqlen, h_dim)

    out = F.scaled_dot_product_attention(
      q, k, v, is_causal=True
    )  # (bsz, nh, seqlen, h_dim)

    out = (
      out.transpose(1, 2).contiguous().view(bsz, seqlen, d)
    )  # (bsz, seqlen, d)

    return self.w_out(out)


class Block(nn.Module):
  def __init__(self, layer_id: int, cfg: ModelConfig):
    super().__init__()
    self.attn = Attention(cfg)
    self.attn_norm = nn.RMSNorm(cfg.dim, eps=cfg.rmsnorm_eps)
    self.mlp = MLP(dim=cfg.dim, hidden_dim=int(cfg.expand * cfg.dim))
    self.mlp_norm = nn.RMSNorm(cfg.dim, eps=cfg.rmsnorm_eps)
    self.layer_id = layer_id

  def forward(self, x, freqs_cis):
    # x: (bsz, seqlen, dim)
    x = x + self.attn(self.attn_norm(x), freqs_cis)
    x = x + self.mlp(self.mlp_norm(x))
    return x


class Transformer(nn.Module):
  def __init__(self, cfg):
    super().__init__()
    self.n_layers = cfg.n_layers
    self.cfg = cfg
    head_dim = cfg.dim // cfg.n_heads
    assert cfg.dim % cfg.n_heads == 0

    self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.dim)
    self.layers = nn.ModuleList(
      [Block(idx, cfg) for idx in range(cfg.n_layers)]
    )
    self.out_norm = nn.RMSNorm(cfg.dim, eps=cfg.rmsnorm_eps)
    self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)

    # Initialize freqs_cis on CPU first (more memory efficient)
    self.register_buffer(
      'freqs_cis',
      precompute_freqs_cis(head_dim, cfg.seq_len, 500000)[0 : cfg.seq_len],
      persistent=False,
    )

    # init all weights, scale residual branches
    self.apply(self._init_weights)
    self._scale_residual_branches()

    # Move model to device (which will also move freqs_cis)
    if torch.cuda.is_available():
      self.cuda()

    if cfg.tie_embeddings:
      self.tie_weights()

  def forward(self, x, targets=None):
    # x: (bsz, seqlen)
    x = self.embed_tokens(x)  # (bsz, seqlen, dim)
    L = x.shape[1]

    # Make sure we have enough precomputed frequencies
    if L > self.freqs_cis.shape[1]:
      # Need to recompute for longer sequence
      head_dim = self.cfg.dim // self.cfg.n_heads
      new_freqs = precompute_freqs_cis(
        head_dim, max(L, self.cfg.seq_len), 500000
      )
      self.register_buffer(
        'freqs_cis', new_freqs[0 : max(L, self.cfg.seq_len)], persistent=False
      )
      if torch.cuda.is_available():
        self.freqs_cis = self.freqs_cis.cuda()

    # Select the frequencies for current sequence length and ensure correct device
    freqs_cis = self.freqs_cis[:, :L, :].to(x.device)

    for layer in self.layers:
      x = layer(x, freqs_cis)  # (bsz, seqlen, dim)
    out = self.lm_head(self.out_norm(x))  # (bsz, seqlen, vocab_size)
    if targets is not None:
      loss = F.cross_entropy(
        out.view(-1, out.size(-1)), targets.view(-1), ignore_index=-100
      )
      return out, loss
    return out

  def predict(self, x, k=1):
    """Generate k tokens autoregressively.

    Args:
        x: Input token sequence of shape (batch_size, seq_len)
        k: Number of tokens to predict

    Returns:
        Tuple of (input_ids, predicted_ids)
    """

    # Store original input
    original_input = x.clone()
    generated_input = x.clone()

    # Generate k tokens autoregressively
    for i in range(k):
      # Get logits for the entire sequence
      logits = self(generated_input)

      # Get the logits for the last token in each sequence
      next_token_logits = logits[:, -1, :]

      # Zero out the last token ID to prevent repetition
      # This is a common issue - the model gets stuck repeating the last token
      last_token_id = generated_input[:, -1]
      next_token_logits.scatter_(1, last_token_id.unsqueeze(1), float('-inf'))

      # Get the most likely token
      next_token = torch.argmax(next_token_logits, dim=-1)

      # Append the predicted token to the sequence
      next_token = next_token.unsqueeze(1)  # Add sequence dimension
      generated_input = torch.cat([generated_input, next_token], dim=1)

    # For debugging, print predictions for the first item in the batch
    print('\nPyTorch detailed prediction (first item in batch):')
    predicted_sequence = generated_input[0, -k:].tolist()
    print(f'  Predicted token IDs: {predicted_sequence}')
    for i, token_id in enumerate(predicted_sequence):
      print(f'  Step {i + 1}: Predicted token {token_id}')

    # Return all tokens, not just the last k
    return original_input, generated_input[:, -k:]

  def _init_weights(self, module):
    if isinstance(module, nn.Linear):
      torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
      if module.bias is not None:
        torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
      torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

  def _scale_residual_branches(self):
    for n, p in self.named_parameters():
      if n.endswith('fc2.weight'):  # mlp/glu output layer
        torch.nn.init.normal_(
          p, mean=0.0, std=0.02 / math.sqrt(2 * self.n_layers)
        )
      if n.endswith('w_out.weight'):  # attn output layer
        torch.nn.init.normal_(
          p, mean=0.0, std=0.02 / math.sqrt(2 * self.n_layers)
        )

  def tie_weights(self):
    self.lm_head.weight = self.embed_tokens.weight

  def count_params(self, non_embedding=True):
    n_params = sum(p.numel() for p in self.parameters())
    if non_embedding:
      n_params -= self.embed_tokens.weight.numel()
      if (
        self.lm_head.weight is not self.embed_tokens.weight
      ):  # if no weight tying
        n_params -= self.lm_head.weight.numel()
    return n_params


def main():
  print('Initializing transformer model and running forward pass...')

  seq_length = 1024

  # Define model configuration
  config = ModelConfig(
    vocab_size=50257,  # Common vocab size for tokenizers like BPE or SentencePiece
    seq_len=seq_length,  # Maximum sequence length
    dim=1024,  # Embedding dimension
    expand=4.0,  # MLP expansion factor
    n_layers=12,  # Number of transformer layers
    n_heads=8,  # Number of attention heads
    rmsnorm_eps=1e-6,  # RMSNorm epsilon
    tie_embeddings=True,  # Tie embedding and output weights
  )

  # Instantiate the model
  model = Transformer(config)
  print(f'Model has {model.count_params():,} parameters.')

  # Create some random input data
  batch_size = 2
  input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))

  # Move data to the same device as the model
  if torch.cuda.is_available():
    input_ids = input_ids.cuda()

  # Run a forward pass
  print(f'Running forward pass with input shape: {input_ids.shape}')
  logits = model(input_ids)
  print(f'Output logits shape: {logits.shape}')

  # Run prediction
  print('Running prediction...')
  original_input, predicted_ids = model.predict(input_ids[:, :10], k=5)
  print(f'Original input shape for prediction: {original_input.shape}')
  print(f'Predicted IDs shape: {predicted_ids.shape}')
  print(f'Predicted IDs: {predicted_ids}')


if __name__ == '__main__':
  main()
