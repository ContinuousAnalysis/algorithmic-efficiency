"""LM workload implemented in Jax."""

from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp

from algoperf import jax_sharding_utils, param_utils, spec
from algoperf.workloads.lm.input_pipeline import get_data_iter
from algoperf.workloads.lm.lm_jax.nanodo_model import (
  ModelConfig,
  TransformerDo,
)
from algoperf.workloads.lm.workload import BaseLmWorkload


class LmWorkload(BaseLmWorkload):
  """LM JAX workload."""

  def _build_input_queue(
    self,
    data_rng: jax.random.PRNGKey,
    split: str,
    data_dir: str,
    global_batch_size: int,
    cache: Optional[bool] = None,
    repeat_final_dataset: Optional[bool] = None,
    num_batches: Optional[int] = None,
  ):
    """Build an input queue using pre-cached FineWeb dataset."""
    del cache, repeat_final_dataset
    ds = get_data_iter(
      data_rng=data_rng,
      split=split,
      data_dir=data_dir,
      batch_size=global_batch_size,
      num_batches=num_batches,
    )
    ds = map(jax_sharding_utils.shard_along_batch_dim, ds)
    return ds

  def init_model_fn(
    self,
    rng: spec.RandomState,
    dropout_rate: Optional[float] = None,
    aux_dropout_rate: Optional[float] = None,
  ) -> spec.ModelInitState:
    # Initialize NanoDO transformer model
    cfg = ModelConfig(
      model_dim=self._emb_dim,  # embedding dim
      num_heads=self._n_heads,  # num heads
      seq_len=self._seq_len,
      num_layers=self._n_layers,  # num layers
      vocab_size=self._vocab_size,
      expanded_model_dim=self._mlp_dim,  # feedforward dim
      dtype=jnp.float32,
    )
    self._model = TransformerDo(cfg)
    input_shape = (1, self._seq_len)  # For token IDs

    params_rng, init_rng = jax.random.split(rng)
    variables = jax.jit(self._model.init)(
      {'params': params_rng}, jnp.ones(input_shape, jnp.int32)
    )
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
    dropout_rate: float = 0.0,
  ) -> Tuple[spec.Tensor, spec.ModelAuxiliaryState]:
    del mode, rng, update_batch_norm, model_state, dropout_rate
    inputs = batch['inputs']
    # Convert one-hot inputs to token IDs if needed
    if inputs.ndim == 3:  # one-hot encoded
      inputs = jnp.argmax(inputs, axis=-1)
    logits = self._model.apply({'params': params}, inputs)
    return logits, None

  def loss_fn(
    self,
    label_batch: spec.Tensor,
    logits_batch: spec.Tensor,
    mask_batch: Optional[spec.Tensor] = None,
    label_smoothing: float = 0.0,
  ) -> Dict[str, spec.Tensor]:  # differentiable
    """Compute weighted cross entropy.

    Args:
     label_batch: categorical targets [batch, length] int array.
     logits_batch: [batch, length, num_classes] float array.
     mask_batch: weights array of shape [batch, length].
     label_smoothing: Label smoothing factor in [0, 1]. When > 0, the target
      distribution becomes (1 - label_smoothing) for the correct class and
      label_smoothing / vocab_size for all other classes. Default is 0.0 (no smoothing).

    Returns:
      {'summed': scalar summed loss, 'n_valid_examples': scalar number of
      valid examples in batch, 'per_example': 2d array of per-example losses}
    """
    if logits_batch.ndim != label_batch.ndim + 1:
      raise ValueError(
        f'Incorrect shapes. Got shape {logits_batch.shape} logits and '
        f'{label_batch.shape} targets.'
      )
    # Compute log probabilities
    log_probs = jax.nn.log_softmax(logits_batch, axis=-1)
    # Extract log probability of the target class
    # Shape: [batch, length]
    target_log_probs = jnp.take_along_axis(
      log_probs, label_batch[..., None], axis=-1
    ).squeeze(-1)
    # Cross-entropy with smoothing: -(1 - α) * log_p[target] - α * mean(log_p)
    # The above formula is easy to derive from the definition of label smoothing and cross-entropy loss.
    confidence = 1.0 - label_smoothing
    smoothing_term = label_smoothing / self._vocab_size
    per_example_losses = -1.0 * (
      confidence * target_log_probs + smoothing_term * log_probs.sum(axis=-1)
    )
    if mask_batch is not None:
      per_example_losses = mask_batch * per_example_losses
      n_valid_examples = mask_batch.sum()
    else:
      n_valid_examples = label_batch.shape[0] * label_batch.shape[1]
    summed_loss = per_example_losses.sum()
    return {
      'summed': summed_loss,
      'n_valid_examples': n_valid_examples,
      'per_example': per_example_losses,
    }

  def _eval_batch(
    self,
    params: spec.ParameterContainer,
    batch: Dict[str, spec.Tensor],
    model_state: spec.ModelAuxiliaryState,
    rng: spec.RandomState,
  ) -> spec.Tensor:
    """Evaluate the model on a single batch."""
    logits, _ = self.model_fn(
      params, batch, model_state, spec.ForwardPassMode.EVAL, rng, False
    )
    metrics = self.loss_fn(
      label_batch=batch['targets'],
      logits_batch=logits,
      mask_batch=batch['weights'],
    )
    return {
      'loss': metrics['summed'],
      'denominator': metrics['n_valid_examples'],
    }

  def _normalize_eval_metrics(
    self, num_examples: int, total_metrics: Dict[str, Any]
  ) -> Dict[str, float]:
    """Normalize eval metrics."""
    del num_examples
    eval_denominator = total_metrics.pop('denominator')
    return jax.tree.map(lambda x: float(x / eval_denominator), total_metrics)
