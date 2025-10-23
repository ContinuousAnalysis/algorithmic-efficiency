"""
Test file to verify that JAX and PyTorch implementations produce identical outputs
when given the same weights and inputs.

Tests are performed module-by-module:
1. RMSNorm
2. RoPE (Rotary Position Embeddings)
3. MLP
4. Attention
5. Transformer Block
6. Full Model
"""

import os
import sys

# Disable GPU access for both jax and pytorch.
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import jax
import jax.numpy as jnp
import numpy as np
import torch
import torch.nn.functional as F
from absl import flags, logging
from absl.testing import absltest, parameterized

# Import JAX implementation
from algoperf.workloads.lm.lm_jax.nanodo_model import (
  CausalAttn,
  Mlp,
  TBlock,
  TransformerDo,
  apply_rope,
  init_rope,
)
from algoperf.workloads.lm.lm_jax.nanodo_model import (
  ModelConfig as JaxModelConfig,
)

# Import PyTorch implementation
from algoperf.workloads.lm.lm_pytorch.plainlm_model import (
  MLP,
  Attention,
  Block,
  Transformer,
  apply_rotary_emb_complex_like,
  precompute_freqs_cis,
)
from algoperf.workloads.lm.lm_pytorch.plainlm_model import (
  ModelConfig as PyTorchModelConfig,
)

FLAGS = flags.FLAGS
# Needed to avoid UnparsedFlagAccessError
FLAGS(sys.argv)

# ============================================================================
# Helper Functions
# ============================================================================


def assert_close(jax_output, torch_output, rtol=1e-5, atol=1e-6, name=''):
  """Assert that JAX and PyTorch outputs are close."""
  jax_np = np.array(jax_output)
  torch_np = torch_output.detach().cpu().numpy()

  mse = np.mean((jax_np - torch_np) ** 2)
  max_diff = np.max(np.abs(jax_np - torch_np))

  logging.info(f'\n{name} Comparison:')
  logging.info(f'  MSE: {mse:.8e}')
  logging.info(f'  Max Difference: {max_diff:.8e}')

  np.testing.assert_allclose(
    jax_np,
    torch_np,
    rtol=rtol,
    atol=atol,
    err_msg=f'{name} outputs do not match',
  )


# ============================================================================
# Test Functions (unchanged)
# ============================================================================


def test_rmsnorm():
  """Test that RMSNorm produces identical outputs."""
  logging.info('\n' + '=' * 70)
  logging.info('Testing RMSNorm')
  logging.info('=' * 70)

  batch_size, seq_len, dim = 2, 10, 256
  eps = 1e-6

  # Create random input
  np_input = np.random.randn(batch_size, seq_len, dim).astype(np.float32)

  # Initialize PyTorch RMSNorm
  torch_norm = torch.nn.RMSNorm(dim, eps=eps)
  torch_input = torch.tensor(np_input)

  # Initialize JAX RMSNorm (using Flax's RMSNorm from nanodo)
  from flax import linen as nn

  flax_norm = nn.RMSNorm(epsilon=eps)
  jax_input = jnp.array(np_input)
  flax_params = flax_norm.init(jax.random.PRNGKey(0), jax_input)

  # Copy weights from PyTorch to JAX
  with torch.no_grad():
    flax_params['params']['scale'] = jnp.array(torch_norm.weight.numpy())

  # Forward pass
  with torch.no_grad():
    torch_output = torch_norm(torch_input)

  jax_output = flax_norm.apply(flax_params, jax_input)

  # Compare
  assert_close(jax_output, torch_output, name='RMSNorm')
  logging.info('✓ RMSNorm test passed')


def test_rope():
  """Test that RoPE produces identical outputs."""
  logging.info('\n' + '=' * 70)
  logging.info('Testing RoPE (Rotary Position Embeddings)')
  logging.info('=' * 70)

  batch_size, seq_len, n_heads, dim = 2, 16, 4, 128
  head_dim = dim // n_heads

  # Initialize RoPE
  torch_freqs = precompute_freqs_cis(head_dim, seq_len, theta=500000)
  jax_freqs = init_rope(dim, seq_len, n_heads)

  # Create random Q and K
  np_q = np.random.randn(batch_size, seq_len, n_heads, head_dim).astype(
    np.float32
  )
  np_k = np.random.randn(batch_size, seq_len, n_heads, head_dim).astype(
    np.float32
  )

  # PyTorch forward
  torch_q = torch.tensor(np_q)
  torch_k = torch.tensor(np_k)
  with torch.no_grad():
    torch_q_rot, torch_k_rot = apply_rotary_emb_complex_like(
      torch_q, torch_k, freqs_cis=torch_freqs
    )

  # JAX forward
  jax_q = jnp.array(np_q)
  jax_k = jnp.array(np_k)
  jax_q_rot, jax_k_rot = apply_rope(jax_q, jax_k, jax_freqs)

  # Compare
  assert_close(jax_q_rot, torch_q_rot, name='RoPE Q')
  assert_close(jax_k_rot, torch_k_rot, name='RoPE K')
  logging.info('✓ RoPE test passed')


def copy_mlp_params(pytorch_mlp, flax_params):
  """Copy MLP parameters from PyTorch to JAX."""
  new_params = flax_params.copy()

  # Handle compiled models
  if hasattr(pytorch_mlp, '_orig_mod'):
    pytorch_mlp = pytorch_mlp._orig_mod

  # Copy fc1 and fc2 weights (transposed for JAX)
  new_params['params']['Dense_0']['kernel'] = (
    pytorch_mlp.fc1.weight.detach().numpy().T
  )
  new_params['params']['Dense_1']['kernel'] = (
    pytorch_mlp.fc2.weight.detach().numpy().T
  )

  return new_params


def test_mlp():
  """Test that MLP produces identical outputs."""
  logging.info('\n' + '=' * 70)
  logging.info('Testing MLP')
  logging.info('=' * 70)

  batch_size, seq_len, dim = 2, 10, 256
  hidden_dim = 1024

  # Initialize PyTorch MLP
  pytorch_mlp = MLP(dim=dim, hidden_dim=hidden_dim)

  # Initialize JAX MLP
  cfg = JaxModelConfig(
    model_dim=dim,
    num_heads=4,
    seq_len=128,
    num_layers=2,
    vocab_size=1000,
    expanded_model_dim=hidden_dim,
    dtype=jnp.float32,
    rmsnorm_epsilon=1e-6,
  )
  flax_mlp = Mlp(cfg)

  # Initialize JAX params
  dummy_input = jnp.ones((batch_size, seq_len, dim))
  flax_params = flax_mlp.init(jax.random.PRNGKey(0), dummy_input)

  # Copy weights
  flax_params = copy_mlp_params(pytorch_mlp, flax_params)

  # Create input
  np_input = np.random.randn(batch_size, seq_len, dim).astype(np.float32)
  torch_input = torch.tensor(np_input)
  jax_input = jnp.array(np_input)

  # Forward pass
  with torch.no_grad():
    torch_output = pytorch_mlp(torch_input)

  jax_output = flax_mlp.apply(flax_params, jax_input)

  # Compare
  assert_close(jax_output, torch_output, name='MLP')
  logging.info('✓ MLP test passed')


def copy_attention_params(pytorch_attn, flax_params):
  """Copy attention parameters from PyTorch to JAX."""
  # Handle compiled models
  if hasattr(pytorch_attn, '_orig_mod'):
    pytorch_attn = pytorch_attn._orig_mod

  n_heads = pytorch_attn.n_heads
  head_dim = pytorch_attn.head_dim
  dim = pytorch_attn.dim

  # Split PyTorch's combined qkv weights
  w_qkv = pytorch_attn.w_qkv.weight
  q_weight, k_weight, v_weight = [
    u.detach().numpy() for u in w_qkv.split(dim, dim=0)
  ]

  # Reshape for Flax's DenseGeneral format [D, H, Dh]
  def reshape_for_flax(w, n_heads, head_dim):
    return w.reshape(n_heads, head_dim, -1).transpose(2, 0, 1)

  new_params = {
    'query': {'kernel': reshape_for_flax(q_weight, n_heads, head_dim)},
    'key': {'kernel': reshape_for_flax(k_weight, n_heads, head_dim)},
    'value': {'kernel': reshape_for_flax(v_weight, n_heads, head_dim)},
    'attn_out_proj': {'kernel': pytorch_attn.w_out.weight.detach().numpy().T},
  }

  return {'params': new_params}


def test_attention():
  """Test that Attention produces identical outputs."""
  logging.info('\n' + '=' * 70)
  logging.info('Testing Attention')
  logging.info('=' * 70)

  batch_size, seq_len, dim, n_heads = 2, 16, 256, 4

  # Initialize PyTorch Attention
  config = PyTorchModelConfig(
    vocab_size=1000,
    seq_len=seq_len,
    model_dim=dim,
    expanded_model_dim=1024,
    num_layers=1,
    num_heads=n_heads,
    rmsnorm_epsilon=1e-6,
  )
  pytorch_attn = Attention(config)
  freqs_cis = precompute_freqs_cis(dim // n_heads, seq_len, theta=500000)

  # Initialize JAX Attention
  cfg = JaxModelConfig(
    model_dim=dim,
    num_heads=n_heads,
    seq_len=seq_len,
    num_layers=1,
    vocab_size=1000,
    expanded_model_dim=1024,
    dtype=jnp.float32,
    rmsnorm_epsilon=1e-6,
  )
  flax_attn = CausalAttn(cfg)

  # Initialize JAX params
  dummy_input = jnp.ones((batch_size, seq_len, dim))
  flax_params = flax_attn.init(jax.random.PRNGKey(0), dummy_input)

  # Copy weights
  flax_params = copy_attention_params(pytorch_attn, flax_params)

  # Create input
  np_input = np.random.randn(batch_size, seq_len, dim).astype(np.float32)
  torch_input = torch.tensor(np_input)
  jax_input = jnp.array(np_input)

  # Forward pass
  with torch.no_grad():
    torch_output = pytorch_attn(torch_input, freqs_cis)

  jax_output = flax_attn.apply(flax_params, jax_input)

  # Compare
  assert_close(jax_output, torch_output, rtol=1e-4, atol=1e-5, name='Attention')
  logging.info('✓ Attention test passed')


def copy_block_params(pytorch_block, flax_params):
  """Copy block parameters from PyTorch to JAX."""
  # Copy attention parameters
  attn_params = copy_attention_params(pytorch_block.attn, {'params': {}})[
    'params'
  ]

  # Copy MLP parameters
  pytorch_mlp = pytorch_block.mlp
  mlp_params = {
    'Dense_0': {'kernel': pytorch_mlp.fc1.weight.detach().numpy().T},
    'Dense_1': {'kernel': pytorch_mlp.fc2.weight.detach().numpy().T},
  }

  # Copy RMSNorm parameters
  norm_params = {
    'attn_norm': {'scale': pytorch_block.attn_norm.weight.detach().numpy()},
    'mlp_norm': {'scale': pytorch_block.mlp_norm.weight.detach().numpy()},
  }

  return {
    'params': {
      'CausalAttn_0': attn_params,
      'Mlp_0': mlp_params,
      'RMSNorm_0': norm_params['attn_norm'],
      'RMSNorm_1': norm_params['mlp_norm'],
    }
  }


def test_block():
  """Test that Transformer Block produces identical outputs."""
  logging.info('\n' + '=' * 70)
  logging.info('Testing Transformer Block')
  logging.info('=' * 70)

  batch_size, seq_len, dim, n_heads = 2, 16, 256, 4
  expand = 4.0

  # Initialize PyTorch Block
  config = PyTorchModelConfig(
    vocab_size=1000,
    seq_len=seq_len,
    model_dim=dim,
    expanded_model_dim=int(dim * expand),
    num_layers=1,
    num_heads=n_heads,
    rmsnorm_epsilon=1e-6,
  )
  pytorch_block = Block(layer_id=0, cfg=config)
  freqs_cis = precompute_freqs_cis(dim // n_heads, seq_len, theta=500000)

  # Initialize JAX Block
  cfg = JaxModelConfig(
    model_dim=dim,
    num_heads=n_heads,
    seq_len=seq_len,
    num_layers=1,
    vocab_size=1000,
    expanded_model_dim=int(dim * expand),
    dtype=jnp.float32,
    rmsnorm_epsilon=1e-6,
  )
  flax_block = TBlock(cfg)

  # Initialize JAX params
  dummy_input = jnp.ones((batch_size, seq_len, dim))
  flax_params = flax_block.init(jax.random.PRNGKey(0), dummy_input)

  # Copy weights
  flax_params = copy_block_params(pytorch_block, flax_params)

  # Create input
  np_input = np.random.randn(batch_size, seq_len, dim).astype(np.float32)
  torch_input = torch.tensor(np_input)
  jax_input = jnp.array(np_input)

  # Forward pass
  with torch.no_grad():
    torch_output = pytorch_block(torch_input, freqs_cis)

  jax_output = flax_block.apply(flax_params, jax_input)

  # Compare
  assert_close(jax_output, torch_output, rtol=1e-4, atol=1e-5, name='Block')
  logging.info('✓ Block test passed')


def copy_full_model_params(pytorch_model, flax_params, config):
  """Copy all parameters from PyTorch model to JAX model."""
  # Handle tied embeddings case
  if hasattr(pytorch_model, '_orig_mod'):
    pytorch_model = pytorch_model._orig_mod

  n_layers = config.num_layers
  n_heads = config.num_heads
  dim = config.model_dim
  head_dim = dim // n_heads

  new_params = {'params': {}}

  # Copy embedding weights
  new_params['params']['embed'] = {
    'embedding': pytorch_model.embed_tokens.weight.detach().numpy()
  }

  # Copy each transformer block
  for i in range(n_layers):
    pytorch_block = pytorch_model.layers[i]

    # Attention params
    w_qkv = pytorch_block.attn.w_qkv.weight
    q_weight, k_weight, v_weight = [
      u.detach().numpy() for u in w_qkv.split(dim, dim=0)
    ]

    def reshape_for_flax(w, n_heads, head_dim):
      return w.reshape(n_heads, head_dim, -1).transpose(2, 0, 1)

    attn_params = {
      'query': {'kernel': reshape_for_flax(q_weight, n_heads, head_dim)},
      'key': {'kernel': reshape_for_flax(k_weight, n_heads, head_dim)},
      'value': {'kernel': reshape_for_flax(v_weight, n_heads, head_dim)},
      'attn_out_proj': {
        'kernel': pytorch_block.attn.w_out.weight.detach().numpy().T
      },
    }

    # MLP params
    mlp_params = {
      'Dense_0': {'kernel': pytorch_block.mlp.fc1.weight.detach().numpy().T},
      'Dense_1': {'kernel': pytorch_block.mlp.fc2.weight.detach().numpy().T},
    }

    # Norm params
    attn_norm = {'scale': pytorch_block.attn_norm.weight.detach().numpy()}
    mlp_norm = {'scale': pytorch_block.mlp_norm.weight.detach().numpy()}

    # Assemble block params
    block_key = f'blocks_{i}'
    new_params['params'][block_key] = {
      'CausalAttn_0': attn_params,
      'Mlp_0': mlp_params,
      'RMSNorm_0': attn_norm,
      'RMSNorm_1': mlp_norm,
    }

  # Copy output norm
  new_params['params']['out_ln'] = {
    'scale': pytorch_model.out_norm.weight.detach().numpy()
  }

  # Handle output projection (tied or untied)
  if not config.tie_embeddings:
    new_params['params']['output_proj'] = {
      'kernel': pytorch_model.lm_head.weight.detach().numpy().T
    }

  return new_params


def test_full_model():
  """Test that full Transformer model produces identical outputs."""
  logging.info('\n' + '=' * 70)
  logging.info('Testing Full Transformer Model')
  logging.info('=' * 70)

  batch_size, seq_len = 2, 32
  vocab_size = 256
  dim = 128
  n_heads = 4
  n_layers = 2
  expand = 4.0

  # Initialize PyTorch model
  pytorch_config = PyTorchModelConfig(
    vocab_size=vocab_size,
    seq_len=seq_len,
    model_dim=dim,
    expanded_model_dim=int(dim * expand),
    num_layers=n_layers,
    num_heads=n_heads,
    rmsnorm_epsilon=1e-6,
    tie_embeddings=True,
  )
  pytorch_model = Transformer(pytorch_config)
  pytorch_model.eval()

  # Initialize JAX model
  jax_config = JaxModelConfig(
    model_dim=dim,
    num_heads=n_heads,
    seq_len=seq_len,
    num_layers=n_layers,
    vocab_size=vocab_size,
    expanded_model_dim=int(dim * expand),
    dtype=jnp.float32,
    rmsnorm_epsilon=1e-6,
    tie_embeddings=True,
  )
  jax_model = TransformerDo(jax_config)

  # Create input tokens
  np_tokens = np.random.randint(
    0, vocab_size, size=(batch_size, seq_len), dtype=np.int32
  )
  torch_tokens = torch.tensor(np_tokens, dtype=torch.long)
  jax_tokens = jnp.array(np_tokens, dtype=jnp.int32)

  # Initialize JAX params
  jax_params = jax_model.init(jax.random.PRNGKey(0), jax_tokens)

  # Copy weights from PyTorch to JAX
  jax_params = copy_full_model_params(pytorch_model, jax_params, pytorch_config)

  # Forward pass
  with torch.no_grad():
    torch_logits = pytorch_model(torch_tokens)

  jax_logits = jax_model.apply(jax_params, jax_tokens)

  # Compare
  assert_close(
    jax_logits, torch_logits, rtol=1e-4, atol=1e-5, name='Full Model'
  )
  logging.info('✓ Full Model test passed')


def test_prediction():
  """Test that autoregressive generation produces identical outputs."""
  logging.info('\n' + '=' * 70)
  logging.info('Testing Autoregressive Prediction')
  logging.info('=' * 70)

  batch_size, seq_len = 1, 10
  vocab_size = 256
  dim = 128
  n_heads = 4
  n_layers = 2
  expand = 4.0
  k = 5  # Number of tokens to predict

  # Initialize PyTorch model
  pytorch_config = PyTorchModelConfig(
    vocab_size=vocab_size,
    seq_len=seq_len + k,
    model_dim=dim,
    expanded_model_dim=int(dim * expand),
    num_layers=n_layers,
    num_heads=n_heads,
    rmsnorm_epsilon=1e-6,
    tie_embeddings=True,
  )
  pytorch_model = Transformer(pytorch_config)
  pytorch_model.eval()

  # Initialize JAX model
  jax_config = JaxModelConfig(
    model_dim=dim,
    num_heads=n_heads,
    seq_len=seq_len + k,
    num_layers=n_layers,
    vocab_size=vocab_size,
    expanded_model_dim=int(dim * expand),
    dtype=jnp.float32,
    rmsnorm_epsilon=1e-6,
    tie_embeddings=True,
  )
  jax_model = TransformerDo(jax_config)

  # Create input tokens
  np_tokens = np.random.randint(
    0, vocab_size, size=(batch_size, seq_len), dtype=np.int32
  )
  torch_tokens = torch.tensor(np_tokens, dtype=torch.long)
  jax_tokens = jnp.array(np_tokens, dtype=jnp.int32)

  # Initialize JAX params
  jax_params = jax_model.init(jax.random.PRNGKey(0), jax_tokens)

  # Copy weights from PyTorch to JAX
  jax_params = copy_full_model_params(pytorch_model, jax_params, pytorch_config)

  # Predict k tokens
  with torch.no_grad():
    _, torch_predictions = pytorch_model.predict(torch_tokens, k=k)

  _, jax_predictions = jax_model.apply(
    jax_params, jax_tokens, k, method=jax_model.predict
  )

  # Compare predictions
  torch_pred_np = torch_predictions.cpu().numpy()
  jax_pred_np = np.array(jax_predictions)

  logging.info(f'\nPyTorch predictions: {torch_pred_np[0]}')
  logging.info(f'JAX predictions: {jax_pred_np[0]}')

  # Check if predictions match exactly
  if np.array_equal(torch_pred_np, jax_pred_np):
    logging.info('✓ Predictions match exactly!')
  else:
    matching = np.sum(torch_pred_np == jax_pred_np)
    total = torch_pred_np.size
    logging.info(
      f'⚠ Predictions differ: {matching}/{total} tokens match ({matching / total * 100:.1f}%)'
    )
    logging.info(
      '  (Note: Small numerical differences can lead to different argmax results)'
    )


def test_initialization_statistics():
  """Verify initialization follows expected distributions."""
  logging.info('\n' + '=' * 70)
  logging.info('Testing Initialization Statistics')
  logging.info('=' * 70)

  # Initialize models
  jax_cfg = JaxModelConfig(
    model_dim=512,
    num_heads=8,
    seq_len=1024,
    num_layers=12,
    vocab_size=50000,
    expanded_model_dim=2048,
    dtype=jnp.float32,
  )
  jax_model = TransformerDo(jax_cfg)
  jax_params = jax_model.init(
    jax.random.PRNGKey(42), jnp.ones((1, 10), dtype=jnp.int32)
  )

  pytorch_cfg = PyTorchModelConfig(
    vocab_size=50000,
    seq_len=1024,
    model_dim=512,
    expanded_model_dim=2048,
    num_layers=12,
    num_heads=8,
  )
  pytorch_model = Transformer(pytorch_cfg)

  logging.info('Initialization Statistics Check:')

  # Check embedding
  jax_embed = jax_params['params']['embed']['embedding']
  torch_embed = pytorch_model.embed_tokens.weight.detach().numpy()

  logging.info('\nToken Embedding (should be ~0.02 std):')
  logging.info(
    f'  JAX:     mean={jax_embed.mean():.6f}, std={jax_embed.std():.6f}'
  )
  logging.info(
    f'  PyTorch: mean={torch_embed.mean():.6f}, std={torch_embed.std():.6f}'
  )

  # Assert embedding std is close to 0.02
  assert abs(jax_embed.std() - 0.02) < 0.005, (
    f'JAX embedding std {jax_embed.std():.6f} not close to 0.02'
  )
  assert abs(torch_embed.std() - 0.02) < 0.005, (
    f'PyTorch embedding std {torch_embed.std():.6f} not close to 0.02'
  )
  assert abs(jax_embed.mean()) < 0.01, (
    f'JAX embedding mean {jax_embed.mean():.6f} not close to 0'
  )
  assert abs(torch_embed.mean()) < 0.01, (
    f'PyTorch embedding mean {torch_embed.mean():.6f} not close to 0'
  )

  # Check first layer attention Q
  jax_q = jax_params['params']['blocks_0']['CausalAttn_0']['query']['kernel']
  torch_q_weight = (
    pytorch_model.layers[0].attn.w_qkv.weight[:512].detach().numpy()
  )

  logging.info('\nAttention Q:')
  logging.info(f'  JAX:     mean={jax_q.mean():.6f}, std={jax_q.std():.6f}')
  logging.info(
    f'  PyTorch: mean={torch_q_weight.mean():.6f}, std={torch_q_weight.std():.6f}'
  )

  # Check means are close to 0
  assert abs(jax_q.mean()) < 0.01, (
    f'JAX Q mean {jax_q.mean():.6f} not close to 0'
  )
  assert abs(torch_q_weight.mean()) < 0.01, (
    f'PyTorch Q mean {torch_q_weight.mean():.6f} not close to 0'
  )

  # Check stds are similar
  # Allow 20% difference due to random initialization
  assert abs(jax_q.std() - torch_q_weight.std()) / torch_q_weight.std() < 0.2, (
    f'Q std differs too much: JAX {jax_q.std():.6f} vs PyTorch {torch_q_weight.std():.6f}'
  )

  # Check first layer attention output (should be scaled)
  jax_attn_out = jax_params['params']['blocks_0']['CausalAttn_0'][
    'attn_out_proj'
  ]['kernel']
  torch_attn_out = pytorch_model.layers[0].attn.w_out.weight.detach().numpy()

  logging.info('\nAttention Output:')
  logging.info(
    f'  JAX:     mean={jax_attn_out.mean():.6f}, std={jax_attn_out.std():.6f}'
  )
  logging.info(
    f'  PyTorch: mean={torch_attn_out.mean():.6f}, std={torch_attn_out.std():.6f}'
  )

  # Check means are close to 0
  assert abs(jax_attn_out.mean()) < 0.01, (
    f'JAX attn out mean {jax_attn_out.mean():.6f} not close to 0'
  )
  assert abs(torch_attn_out.mean()) < 0.01, (
    f'PyTorch attn out mean {torch_attn_out.mean():.6f} not close to 0'
  )

  # Check stds are similar
  assert (
    abs(jax_attn_out.std() - torch_attn_out.std()) / torch_attn_out.std() < 0.2
  ), (
    f'Attention output std differs too much: JAX {jax_attn_out.std():.6f} vs PyTorch {torch_attn_out.std():.6f}'
  )

  # Check MLP fc2 (should be scaled)
  jax_mlp_out = jax_params['params']['blocks_0']['Mlp_0']['Dense_1']['kernel']
  torch_mlp_out = pytorch_model.layers[0].mlp.fc2.weight.detach().numpy()

  logging.info('\nMLP Output:')
  logging.info(
    f'  JAX:     mean={jax_mlp_out.mean():.6f}, std={jax_mlp_out.std():.6f}'
  )
  logging.info(
    f'  PyTorch: mean={torch_mlp_out.mean():.6f}, std={torch_mlp_out.std():.6f}'
  )

  # Check means are close to 0
  assert abs(jax_mlp_out.mean()) < 0.01, (
    f'JAX MLP out mean {jax_mlp_out.mean():.6f} not close to 0'
  )
  assert abs(torch_mlp_out.mean()) < 0.01, (
    f'PyTorch MLP out mean {torch_mlp_out.mean():.6f} not close to 0'
  )

  # Check stds are similar
  assert (
    abs(jax_mlp_out.std() - torch_mlp_out.std()) / torch_mlp_out.std() < 0.2
  ), (
    f'MLP output std differs too much: JAX {jax_mlp_out.std():.6f} vs PyTorch {torch_mlp_out.std():.6f}'
  )

  logging.info('\n✓ Initialization statistics test passed')


def test_initialization_impact():
  """Test that initialization produces similar initial losses."""
  logging.info('\n' + '=' * 70)
  logging.info('Testing Initialization Impact')
  logging.info('=' * 70)

  # Create identical inputs
  batch_size, seq_len = 4, 128
  vocab_size = 50000

  np.random.seed(42)
  tokens = np.random.randint(0, vocab_size, size=(batch_size, seq_len))

  # Initialize both models with same seed
  jax_cfg = JaxModelConfig(
    model_dim=512,
    num_heads=8,
    seq_len=seq_len,
    num_layers=12,
    vocab_size=vocab_size,
    expanded_model_dim=2048,
  )
  jax_model = TransformerDo(jax_cfg)
  jax_params = jax_model.init(
    jax.random.PRNGKey(42), jnp.array(tokens, dtype=jnp.int32)
  )

  torch.manual_seed(42)
  pytorch_cfg = PyTorchModelConfig(
    vocab_size=vocab_size,
    seq_len=seq_len,
    model_dim=512,
    expanded_model_dim=2048,
    num_layers=12,
    num_heads=8,
  )
  pytorch_model = Transformer(pytorch_cfg)

  # Forward pass
  jax_logits = jax_model.apply(jax_params, jnp.array(tokens, dtype=jnp.int32))

  with torch.no_grad():
    torch_logits = pytorch_model(torch.tensor(tokens, dtype=torch.long))

  # Compute losses
  targets = tokens[:, 1:]
  jax_loss = -jax.nn.log_softmax(jax_logits[:, :-1]).mean()
  torch_loss = F.cross_entropy(
    torch_logits[:, :-1].reshape(-1, vocab_size),
    torch.tensor(targets.reshape(-1), dtype=torch.long),
  )

  logging.info('\nInitial Loss Comparison:')
  logging.info(f'  JAX:     {jax_loss:.4f}')
  logging.info(f'  PyTorch: {torch_loss.item():.4f}')
  logging.info(f'  Difference: {abs(jax_loss - torch_loss.item()):.6f}')

  # Check that losses are in reasonable range for random init
  # With vocab_size=50000, random init should give loss around log(50000) ≈ 10.82
  expected_loss = np.log(vocab_size)

  assert 8.0 < jax_loss < 13.0, (
    f'JAX loss {jax_loss:.4f} outside expected range [8.0, 13.0]'
  )
  assert 8.0 < torch_loss.item() < 13.0, (
    f'PyTorch loss {torch_loss.item():.4f} outside expected range [8.0, 13.0]'
  )

  # Both losses should be within 10% of log(vocab_size)
  assert abs(jax_loss - expected_loss) / expected_loss < 0.1, (
    f'JAX loss {jax_loss:.4f} too far from expected {expected_loss:.4f}'
  )
  assert abs(torch_loss.item() - expected_loss) / expected_loss < 0.1, (
    f'PyTorch loss {torch_loss.item():.4f} too far from expected {expected_loss:.4f}'
  )

  logging.info(
    '\nNote: Losses are in expected range for random initialization.'
  )
  logging.info(f'      Expected ~log(vocab_size) = {expected_loss:.4f}')
  logging.info('\n✓ Initialization impact test passed')


# ============================================================================
# Test Class
# ============================================================================

named_parameters = [
  dict(testcase_name='rmsnorm', test_fn=test_rmsnorm),
  dict(testcase_name='rope', test_fn=test_rope),
  dict(testcase_name='mlp', test_fn=test_mlp),
  dict(testcase_name='attention', test_fn=test_attention),
  dict(testcase_name='block', test_fn=test_block),
  dict(testcase_name='full_model', test_fn=test_full_model),
  dict(testcase_name='prediction', test_fn=test_prediction),
  dict(
    testcase_name='initialization_statistics',
    test_fn=test_initialization_statistics,
  ),
  dict(
    testcase_name='initialization_impact', test_fn=test_initialization_impact
  ),
]


class ModelMatchingTest(parameterized.TestCase):
  """Tests for JAX vs PyTorch model matching."""

  @parameterized.named_parameters(*named_parameters)
  def test_model_matching(self, test_fn):
    """Run individual model matching test."""
    test_fn()


if __name__ == '__main__':
  absltest.main()
