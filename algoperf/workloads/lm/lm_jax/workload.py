"""LM workload implemented in Jax."""

from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import optax
from flax.training import common_utils

from algoperf import jax_sharding_utils, param_utils, spec
from algoperf.workloads.lm.input_pipeline import get_data_iter
from algoperf.workloads.lm.lm_jax.nanodo_model import (
  DoConfig,
  TransformerDo,
)
from algoperf.workloads.lm.workload import BaseLmWorkload


class LmWorkload(BaseLmWorkload):
  """LM JAX workload."""
  def _build_input_queue(self,
                         data_rng: jax.random.PRNGKey,
                         split: str,
                         data_dir: str,
                         global_batch_size: int,
                         num_batches: Optional[int] = None,
                         repeat_final_dataset: bool = False):
    """Build an input queue using pre-cached FineWeb dataset."""
    del num_batches
    del repeat_final_dataset
    loader = get_data_iter(
        data_rng=data_rng,
        split=split,
        data_dir=data_dir,
        global_batch_size=global_batch_size)
    loader = map(jax_sharding_utils.shard_along_batch_dim, loader)
    return loader

  def _build_hf_input_queue(self,
                         data_rng: jax.random.PRNGKey,
                         split: str,
                         data_dir: str,
                         global_batch_size: int,
                         num_batches: Optional[int] = None,
                         repeat_final_dataset: bool = False):
    """Build an input queue using HuggingFace FineWeb dataset."""
    del num_batches
    del repeat_final_dataset
    iter = get_data_iter(data_rng, split, data_dir, global_batch_size)
    return iter

  def init_model_fn(
      self,
      rng: spec.RandomState,
      dropout_rate: Optional[float] = None,
      aux_dropout_rate: Optional[float] = None) -> spec.ModelInitState:

    # Initialize NanoDO transformer model
    cfg = DoConfig(
        D=self._emb_dim,  # embedding dim
        H=self._n_heads,    # num heads
        L=self._seq_len,
        N=self._n_layers,    # num layers
        V=self._vocab_size,
        F=self._mlp_dim, # feedforward dim
        dtype=jnp.float32
    )
    self._model = TransformerDo(cfg)
    input_shape = (1, self._seq_len)  # For token IDs

    params_rng, init_rng = jax.random.split(rng)
    variables = jax.jit(self._model.init)({'params': params_rng},
                                        jnp.ones(input_shape, jnp.int32))
    params = variables['params']
    self._param_shapes = param_utils.jax_param_shapes(params)
    self._param_types = param_utils.jax_param_types(self._param_shapes)
    params = jax_sharding_utils.replicate(params)
    model_state = None
    return params, model_state

  def model_fn(
      self,
      params: spec.ParameterContainer,
      batch: Dict[str, spec.Tensor],
      model_state: spec.ModelAuxiliaryState,
      mode: spec.ForwardPassMode,
      rng: spec.RandomState,
      update_batch_norm: bool,
      dropout_rate: float = None) -> Tuple[spec.Tensor, spec.ModelAuxiliaryState]:
    del mode, rng, update_batch_norm, model_state, dropout_rate
    inputs = batch['inputs']
    # Convert one-hot inputs to token IDs if needed
    if inputs.ndim == 3:  # one-hot encoded
      inputs = jnp.argmax(inputs, axis=-1)
    logits = self._model.apply({'params': params}, inputs)
    return logits, None

  
  def compute_weighted_cross_entropy(
    self,
    logits: spec.Tensor,
    targets: spec.Tensor,
    weights: Optional[spec.Tensor] = None,
    label_smoothing: float = 0.1,
  ) -> Dict[str, spec.Tensor]:  # differentiable
    """Compute weighted cross entropy and entropy for log probs and targets.

    Args:
     logits: [batch, length, num_classes] float array.
     targets: categorical targets [batch, length] int array.
     weights: array of shape [batch, length].
     label_smoothing: label smoothing constant, used to determine the on and off
       values.

    Returns:
      {'summed': scalar summed loss, 'n_valid_examples': scalar number of
      valid examples in batch, 'per_example': 1-d array of per-example losses}
    """
    if logits.ndim != targets.ndim + 1:
      raise ValueError(
        f'Incorrect shapes. Got shape {logits.shape} logits and '
        f'{targets.shape} targets.'
      )
    smoothed_targets = optax.smooth_labels(
      common_utils.onehot(targets, self._vocab_size), label_smoothing
    )

    per_example_losses = -jnp.sum(
      smoothed_targets * jax.nn.log_softmax(logits), axis=-1
    )
    if weights is None:
      weights = jnp.ones_like(targets)
    per_example_losses = jnp.where(weights, per_example_losses, 0.0)
    summed_loss = per_example_losses.sum()
    n_valid_examples = weights.sum()
    return {
      'summed': summed_loss,
      'n_valid_examples': n_valid_examples,
      'per_example': per_example_losses,
    }

  def _normalize_eval_metrics(
    self, num_examples: int, total_metrics: Dict[str, Any]
  ) -> Dict[str, float]:
    """Normalize eval metrics."""
    del num_examples
    eval_denominator = total_metrics.pop('denominator')
    return jax.tree.map(lambda x: float(x / eval_denominator), total_metrics)


  def _eval_batch(self,
                  params: spec.ParameterContainer,
                  batch: Dict[str, spec.Tensor],
                  model_state: spec.ModelAuxiliaryState,
                  rng: spec.RandomState) -> spec.Tensor:
    """Evaluate the model on a single batch."""
    logits, _ = self.model_fn(
        params, batch, model_state, spec.ForwardPassMode.EVAL, rng, False)
    targets = batch['targets']

    # Calculate cross-entropy loss
    # TODO(kasimbeg): add weights?
    loss_metrics = self.compute_weighted_cross_entropy(logits, targets)
    return loss_metrics
