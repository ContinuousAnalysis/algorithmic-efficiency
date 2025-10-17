"""LM workload implemented in PyTorch."""

from itertools import islice
from typing import Any, Dict, Iterator, Optional, Tuple

import jax
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from algoperf import data_utils, param_utils, pytorch_utils, spec
from algoperf.workloads.lm.lm_pytorch.plainlm_model import (
  ModelConfig,
  Transformer,
)
from algoperf.workloads.lm.workload import BaseLmWorkload
from algoperf.workloads.lm.input_pipeline import get_data_iter

USE_PYTORCH_DDP, RANK, DEVICE, N_GPUS = pytorch_utils.pytorch_setup()


class LmWorkload(BaseLmWorkload):
  """LM PyTorch workload."""

  def init_model_fn(
      self,
      rng: spec.RandomState,
      dropout_rate: Optional[float] = None,
      aux_dropout_rate: Optional[float] = None) -> spec.ModelInitState:

    if hasattr(self, '_model'):
        # Reinitialize weights but keep same config
        self._model.apply(self._model._init_weights)
        self._model._scale_residual_branches()
        return self._model, None

    torch.manual_seed(rng[0])
    cfg = ModelConfig(
        vocab_size=self._vocab_size,
        seq_len=self._seq_len,
        dim=self._emb_dim,  # Model dimension
        expand=self._mlp_dim // self._emb_dim,  # MLP expansion factor
        # FIXME(rka97): fix expansion factor
        n_layers=self._n_layers,  # Number of transformer layers
        n_heads=self._n_heads,  # Number of attention heads
        rmsnorm_eps=1e-6,
        tie_embeddings=True
    )
    self._model = Transformer(cfg)
    self._param_shapes = param_utils.pytorch_param_shapes(self._model)
    self._param_types = param_utils.pytorch_param_types(self._param_shapes)
    self._model.to(DEVICE)

    if N_GPUS > 1:
        if USE_PYTORCH_DDP:
            self._model = DDP(self._model, device_ids=[RANK], output_device=RANK)
        else:
            self._model = torch.nn.DataParallel(self._model)

    return self._model, None

  def model_fn(
      self,
      params: spec.ParameterContainer,
      augmented_and_preprocessed_input_batch: Dict[str, spec.Tensor],
      model_state: spec.ModelAuxiliaryState,
      mode: spec.ForwardPassMode,
      rng: spec.RandomState,
      update_batch_norm: bool,
      dropout_rate: float = 0.0) -> Tuple[spec.Tensor, spec.ModelAuxiliaryState]:

    del model_state, rng, update_batch_norm, dropout_rate
    model = params

    # Convert one-hot inputs to token IDs if needed
    inputs = augmented_and_preprocessed_input_batch['inputs']
    if inputs.dim() == 3:  # one-hot encoded
        inputs = inputs.argmax(dim=-1)

    logits = model(inputs)
    return logits, None

  def _build_input_queue(
      self,
      data_rng: jax.random.PRNGKey,
      split: str,
      data_dir: str,
      global_batch_size: int,
      num_batches: Optional[int] = None,
      repeat_final_dataset: bool = False) -> Iterator[Dict[str, spec.Tensor]]:
    """Build an input queue for the given split."""
    local_batch_size = global_batch_size // N_GPUS
    loader = get_data_iter(
        data_rng=data_rng,
        split=split,
        data_dir=data_dir,
        global_batch_size=local_batch_size,
        num_batches=num_batches
    )
    if USE_PYTORCH_DDP:
       loader = islice(loader, RANK, None, N_GPUS)
    dtype = torch.int32
    for batch in loader:
      batch = {
          'inputs': torch.tensor(batch['inputs'], device=DEVICE, dtype=dtype),
          'targets': torch.tensor(batch['targets'], device=DEVICE, dtype=torch.int64),
          'weights': None,
      }
      yield batch

  def is_output_params(self, param_name: str) -> bool:
    """Return whether the given parameter is an output parameter."""
    return 'lm_head.weight' in param_name or 'lm_head.bias' in param_name

  # FIXME(rka97): Implement label smoothing
  def compute_weighted_cross_entropy(self, logits: spec.Tensor, labels: spec.Tensor, weights: spec.Tensor, label_smoothing: float = 0.0) -> Dict[str, spec.Tensor]:
    """Compute cross-entropy loss for language modeling in PyTorch."""
    vocab_size = logits.size(-1)

    if len(labels.shape) == len(logits.shape):
      # One-hot labels
      log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
      loss = -torch.sum(labels * log_probs, dim=-1)
    else:
      # Dense labels
      loss = torch.nn.functional.cross_entropy(
          logits.view(-1, vocab_size),
          labels.view(-1),
          reduction='none')
      loss = loss.view_as(labels)

    if weights is not None:
      loss = loss * weights

    n_valid = weights.sum() if weights is not None else torch.tensor(labels.numel(), dtype=torch.float32, device=labels.device)
    return {
        'summed': loss.sum(),
        'n_valid_examples': n_valid,
        'per_example': loss,
    }

  def _normalize_eval_metrics(
      self, num_examples: int, total_metrics: Dict[str, Any]
    ) -> Dict[str, float]:
      """Normalize eval metrics."""
      del num_examples
      if USE_PYTORCH_DDP:
        for metric in total_metrics.values():
          dist.all_reduce(metric)
      total_metrics = {k: v.item() for k, v in total_metrics.items()}
      eval_denominator = total_metrics.pop('denominator')
      return jax.tree.map(lambda x: float(x / eval_denominator), total_metrics)