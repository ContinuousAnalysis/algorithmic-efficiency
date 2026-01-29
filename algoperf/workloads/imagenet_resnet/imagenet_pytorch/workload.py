"""ImageNet workload implemented in PyTorch."""

import contextlib
import functools
import itertools
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import transforms
from torchvision.datasets.folder import (
  IMG_EXTENSIONS,
  ImageFolder,
  default_loader,
)

import algoperf.random_utils as prng
from algoperf import data_utils, param_utils, pytorch_utils, spec
from algoperf.workloads.imagenet_resnet import imagenet_v2
from algoperf.workloads.imagenet_resnet.imagenet_pytorch import randaugment
from algoperf.workloads.imagenet_resnet.imagenet_pytorch.models import resnet50
from algoperf.workloads.imagenet_resnet.workload import (
  BaseImagenetResNetWorkload,
)

USE_PYTORCH_DDP, RANK, DEVICE, N_GPUS = pytorch_utils.pytorch_setup()


class CachedImageFolder(ImageFolder):
  """ImageFolder that caches the file listing to avoid repeated filesystem scans."""

  def __init__(
    self,
    root: Union[str, Path],
    cache_file: Optional[Union[str, Path]] = None,
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
    loader: Callable[[str], Any] = default_loader,
    is_valid_file: Optional[Callable[[str], bool]] = None,
    allow_empty: bool = False,
    rebuild_cache: bool = False,
    cache_build_timeout_minutes: int = 30,
  ):
    self.root = os.path.abspath(root)
    self.transform = transform
    self.target_transform = target_transform
    self.loader = loader
    self.extensions = IMG_EXTENSIONS if is_valid_file is None else None

    # Default cache location: .cache_index.json in the root directory
    if cache_file is None:
      cache_file = os.path.join(self.root, '.cache_index.json')
    self.cache_file = cache_file

    is_distributed = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if is_distributed else 0

    cache_exists = os.path.exists(self.cache_file)
    needs_rebuild = rebuild_cache or not cache_exists

    if needs_rebuild:
      # We only want one process to build the cache
      # and others to wait for it to finish.
      if rank == 0:
        self._build_and_save_cache(is_valid_file, allow_empty)
      if is_distributed:
        self._wait_for_cache(timeout_minutes=cache_build_timeout_minutes)
        dist.barrier()

    self._load_from_cache()

    self.targets = [s[1] for s in self.samples]
    self.imgs = self.samples

  def _wait_for_cache(self, timeout_minutes: int):
    """Poll for cache file to exist."""
    timeout_seconds = timeout_minutes * 60
    poll_interval = 5
    elapsed = 0

    while not os.path.exists(self.cache_file):
      if elapsed >= timeout_seconds:
        raise TimeoutError(
          f'Timed out waiting for cache file after {timeout_minutes} minutes: {self.cache_file}'
        )
      time.sleep(poll_interval)
      elapsed += poll_interval

  def _load_from_cache(self):
    """Load classes and samples from cache file."""
    with open(os.path.abspath(self.cache_file), 'r') as f:
      cache = json.load(f)
    self.classes = cache['classes']
    self.class_to_idx = cache['class_to_idx']
    # Convert relative paths back to absolute
    self.samples = [
      (os.path.join(self.root, rel_path), idx)
      for rel_path, idx in cache['samples']
    ]

  def _build_and_save_cache(self, is_valid_file, allow_empty):
    """Scan filesystem, build index, and save to cache."""
    self.classes, self.class_to_idx = self.find_classes(self.root)
    self.samples = self.make_dataset(
      self.root,
      class_to_idx=self.class_to_idx,
      extensions=self.extensions,
      is_valid_file=is_valid_file,
      allow_empty=allow_empty,
    )

    cache = {
      'classes': self.classes,
      'class_to_idx': self.class_to_idx,
      'samples': [
        (os.path.relpath(path, self.root), idx) for path, idx in self.samples
      ],
    }
    with open(os.path.abspath(self.cache_file), 'w') as f:
      json.dump(cache, f)


def imagenet_v2_to_torch(
  batch: Dict[str, spec.Tensor],
) -> Dict[str, spec.Tensor]:
  # Slice off the part of the batch for this device and then transpose from
  # [N, H, W, C] to [N, C, H, W]. Only transfer the inputs to GPU.
  new_batch = {}
  for k, v in batch.items():
    if USE_PYTORCH_DDP:
      new_v = v[RANK]
    else:
      new_v = v.reshape(-1, *v.shape[2:])
    if k == 'inputs':
      new_v = np.transpose(new_v, (0, 3, 1, 2))
    dtype = torch.long if k == 'targets' else torch.float
    new_batch[k] = torch.as_tensor(new_v, dtype=dtype, device=DEVICE)
  return new_batch


class ImagenetResNetWorkload(BaseImagenetResNetWorkload):
  def __init__(self, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    # Is set in submission_runner.py for workloads with PyTorch evaluation
    # data loaders via the `eval_num_workers` property.
    self._eval_num_workers = None

  @property
  def eval_num_workers(self) -> int:
    if self._eval_num_workers is None:
      raise ValueError(
        'eval_num_workers property must be set before workload is used.'
      )
    return self._eval_num_workers

  @eval_num_workers.setter
  def eval_num_workers(self, eval_num_workers: int):
    self._eval_num_workers = eval_num_workers

  def _build_dataset(
    self,
    data_rng: spec.RandomState,
    split: str,
    data_dir: str,
    global_batch_size: int,
    cache: Optional[bool] = None,
    repeat_final_dataset: Optional[bool] = None,
    use_mixup: bool = False,
    use_randaug: bool = False,
  ) -> Iterator[Dict[str, spec.Tensor]]:
    del cache
    del repeat_final_dataset
    if split == 'test':
      np_iter = imagenet_v2.get_imagenet_v2_iter(
        data_dir,
        global_batch_size,
        mean_rgb=self.train_mean,
        stddev_rgb=self.train_stddev,
        image_size=self.center_crop_size,
        resize_size=self.resize_size,
        framework='pytorch',
      )
      return map(imagenet_v2_to_torch, itertools.cycle(np_iter))

    is_train = split == 'train'
    normalize = transforms.Normalize(
      mean=[i / 255.0 for i in self.train_mean],
      std=[i / 255.0 for i in self.train_stddev],
    )
    if is_train:
      transform_config = [
        transforms.RandomResizedCrop(
          self.center_crop_size,
          scale=self.scale_ratio_range,
          ratio=self.aspect_ratio_range,
        ),
        transforms.RandomHorizontalFlip(),
      ]
      if use_randaug:
        transform_config.append(randaugment.RandAugment())
      transform_config.extend([transforms.ToTensor(), normalize])
      transform_config = transforms.Compose(transform_config)
    else:
      transform_config = transforms.Compose(
        [
          transforms.Resize(self.resize_size),
          transforms.CenterCrop(self.center_crop_size),
          transforms.ToTensor(),
          normalize,
        ]
      )

    folder = 'train' if 'train' in split else 'val'
    dataset = ImageFolder(
      os.path.join(data_dir, folder),
      transform=transform_config,
      cache_file='.imagenet_{}_cache_index.json'.format(split),
    )

    if split == 'eval_train':
      indices = list(range(self.num_train_examples))
      random.Random(int(data_rng[0])).shuffle(indices)
      dataset = torch.utils.data.Subset(
        dataset, indices[: self.num_eval_train_examples]
      )

    sampler = None
    if USE_PYTORCH_DDP:
      per_device_batch_size = global_batch_size // N_GPUS
      ds_iter_batch_size = per_device_batch_size
    else:
      ds_iter_batch_size = global_batch_size
    if USE_PYTORCH_DDP:
      if is_train:
        sampler = torch.utils.data.distributed.DistributedSampler(
          dataset, num_replicas=N_GPUS, rank=RANK, shuffle=True
        )
      else:
        sampler = data_utils.DistributedEvalSampler(
          dataset, num_replicas=N_GPUS, rank=RANK, shuffle=False
        )
    dataloader = torch.utils.data.DataLoader(
      dataset,
      batch_size=ds_iter_batch_size,
      shuffle=not USE_PYTORCH_DDP and is_train,
      sampler=sampler,
      num_workers=5 * N_GPUS,
      pin_memory=True,
      drop_last=is_train,
      persistent_workers=is_train,
      prefetch_factor=N_GPUS,
    )
    dataloader = data_utils.PrefetchedWrapper(dataloader, DEVICE)
    dataloader = data_utils.cycle(
      dataloader,
      custom_sampler=USE_PYTORCH_DDP,
      use_mixup=use_mixup,
      mixup_alpha=0.2,
    )
    return dataloader

  def init_model_fn(self, rng: spec.RandomState) -> spec.ModelInitState:
    torch.random.manual_seed(rng[0])

    if self.use_silu and self.use_gelu:
      raise RuntimeError('Cannot use both GELU and SiLU activations.')
    if self.use_silu:
      act_fnc = torch.nn.SiLU(inplace=True)
    elif self.use_gelu:
      act_fnc = torch.nn.GELU(approximate='tanh')
    else:
      act_fnc = torch.nn.ReLU(inplace=True)

    model = resnet50(act_fnc=act_fnc, bn_init_scale=self.bn_init_scale)
    self._param_shapes = param_utils.pytorch_param_shapes(model)
    self._param_types = param_utils.pytorch_param_types(self._param_shapes)
    model.to(DEVICE)
    if N_GPUS > 1:
      if USE_PYTORCH_DDP:
        model = DDP(model, device_ids=[RANK], output_device=RANK)
      else:
        model = torch.nn.DataParallel(model)
    return model, None

  def is_output_params(self, param_key: spec.ParameterKey) -> bool:
    return param_key in ['fc.weight', 'fc.bias']

  def model_fn(
    self,
    params: spec.ParameterContainer,
    augmented_and_preprocessed_input_batch: Dict[str, spec.Tensor],
    model_state: spec.ModelAuxiliaryState,
    mode: spec.ForwardPassMode,
    rng: spec.RandomState,
    update_batch_norm: bool,
    dropout_rate: float = 0.0,
  ) -> Tuple[spec.Tensor, spec.ModelAuxiliaryState]:
    del model_state
    del rng
    del dropout_rate

    model = params

    if mode == spec.ForwardPassMode.EVAL:
      if update_batch_norm:
        raise ValueError(
          'Batch norm statistics cannot be updated during evaluation.'
        )
      model.eval()

    if mode == spec.ForwardPassMode.TRAIN:
      model.train()
      model.apply(
        functools.partial(
          pytorch_utils.update_batch_norm_fn,
          update_batch_norm=update_batch_norm,
        )
      )

    contexts = {
      spec.ForwardPassMode.EVAL: torch.no_grad,
      spec.ForwardPassMode.TRAIN: contextlib.nullcontext,
    }

    with contexts[mode]():
      logits_batch = model(augmented_and_preprocessed_input_batch['inputs'])

    return logits_batch, None

  # Does NOT apply regularization, which is left to the submitter to do in
  # `update_params`.
  def loss_fn(
    self,
    label_batch: spec.Tensor,  # Dense or one-hot labels.
    logits_batch: spec.Tensor,
    mask_batch: Optional[spec.Tensor] = None,
    label_smoothing: float = 0.0,
  ) -> Dict[str, spec.Tensor]:  # differentiable
    """Evaluate the (masked) loss function at (label_batch, logits_batch).

    Return {'summed': scalar summed loss, 'n_valid_examples': scalar number of
    valid examples in batch, 'per_example': 1-d array of per-example losses}
    (not synced across devices).
    """
    per_example_losses = F.cross_entropy(
      logits_batch,
      label_batch,
      reduction='none',
      label_smoothing=label_smoothing,
    )
    # `mask_batch` is assumed to be shape [batch].
    if mask_batch is not None:
      per_example_losses *= mask_batch
      n_valid_examples = mask_batch.sum()
    else:
      n_valid_examples = len(per_example_losses)
    summed_loss = per_example_losses.sum()
    return {
      'summed': summed_loss,
      'n_valid_examples': torch.as_tensor(n_valid_examples, device=DEVICE),
      'per_example': per_example_losses,
    }

  def _compute_metrics(
    self, logits: spec.Tensor, labels: spec.Tensor, weights: spec.Tensor
  ) -> Dict[str, spec.Tensor]:
    """Return the mean accuracy and loss as a dict."""
    if weights is None:
      weights = torch.ones(len(logits), device=DEVICE)
    predicted = torch.argmax(logits, 1)
    # Not accuracy, but nr. of correct predictions.
    accuracy = ((predicted == labels) * weights).sum()
    summed_loss = self.loss_fn(labels, logits, weights)['summed']
    return {'accuracy': accuracy, 'loss': summed_loss}

  def _eval_model_on_split(
    self,
    split: str,
    num_examples: int,
    global_batch_size: int,
    params: spec.ParameterContainer,
    model_state: spec.ModelAuxiliaryState,
    rng: spec.RandomState,
    data_dir: str,
    global_step: int = 0,
  ) -> Dict[str, float]:
    """Run a full evaluation of the model."""
    del global_step
    data_rng, model_rng = prng.split(rng, 2)
    if split not in self._eval_iters:
      is_test = split == 'test'
      # These iterators repeat indefinitely.
      self._eval_iters[split] = self._build_input_queue(
        data_rng,
        split=split,
        global_batch_size=global_batch_size,
        data_dir=data_dir,
        cache=is_test,
        repeat_final_dataset=is_test,
      )

    total_metrics = {
      'accuracy': torch.tensor(0.0, device=DEVICE),
      'loss': torch.tensor(0.0, device=DEVICE),
    }
    num_batches = int(math.ceil(num_examples / global_batch_size))
    for _ in range(num_batches):
      batch = next(self._eval_iters[split])
      logits, _ = self.model_fn(
        params,
        batch,
        model_state,
        spec.ForwardPassMode.EVAL,
        model_rng,
        update_batch_norm=False,
      )
      weights = batch.get('weights')
      batch_metrics = self._compute_metrics(logits, batch['targets'], weights)
      total_metrics = {
        k: v + batch_metrics[k] for k, v in total_metrics.items()
      }
    if USE_PYTORCH_DDP:
      for metric in total_metrics.values():
        dist.all_reduce(metric)
    return {k: float(v.item() / num_examples) for k, v in total_metrics.items()}


class ImagenetResNetSiLUWorkload(ImagenetResNetWorkload):
  @property
  def use_silu(self) -> bool:
    return True

  @property
  def validation_target_value(self) -> float:
    return 0.75445

  @property
  def test_target_value(self) -> float:
    return 0.6323


class ImagenetResNetGELUWorkload(ImagenetResNetWorkload):
  @property
  def use_gelu(self) -> bool:
    return True

  @property
  def validation_target_value(self) -> float:
    return 0.76765

  @property
  def test_target_value(self) -> float:
    return 0.6519


class ImagenetResNetLargeBNScaleWorkload(ImagenetResNetWorkload):
  @property
  def bn_init_scale(self) -> float:
    return 8.0

  @property
  def validation_target_value(self) -> float:
    return 0.76526

  @property
  def test_target_value(self) -> float:
    return 0.6423
