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
    ds = get_data_iter(
        data_rng=data_rng,
        split=split,
        data_dir=data_dir,
        global_batch_size=global_batch_size)
    ds = map(jax_sharding_utils.shard_along_batch_dim, ds)
    return ds

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
    # Compute log probabilities
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    # Extract log probability of the target class
    # Shape: [batch, length]
    target_log_probs = jnp.take_along_axis(
      log_probs, 
      targets[..., None], 
      axis=-1
    ).squeeze(-1)
    # Cross-entropy with smoothing: -(1 - α) * log_p[target] - α * mean(log_p)
    # The above formula is easy to derive from the definition of label smoothing and cross-entropy loss.
    confidence = 1.0 - label_smoothing
    smoothing_term = label_smoothing / self._vocab_size
    per_example_losses = -1.0 * (confidence * target_log_probs + smoothing_term * log_probs.sum(axis=-1))
    if weights is not None:
      per_example_losses = jnp.where(weights, per_example_losses, 0.0)
      n_valid_examples = weights.sum()
    else:
      n_valid_examples = targets.shape[0] * targets.shape[1]
    summed_loss = per_example_losses.sum()
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
