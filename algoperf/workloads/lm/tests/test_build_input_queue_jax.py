import jax
import jax.numpy as jnp

from algoperf.profiler import PassThroughProfiler
from algoperf.workloads.lm.lm_jax.workload import LmWorkload
import os

RANK = os.environ.get('RANK', 0)

def test_dataloader_jax():
  # Test config.
  rng_seed = 1996
  data_dir = '/home/ak4605/data/finewebedu/'
  split = 'train'
  global_batch_size = 64
  dtype = jnp.int32
  seq_len = 2048

  workload = LmWorkload()
  data_rng = jax.random.PRNGKey(rng_seed)
  input_queue = workload._build_input_queue(
      data_rng=data_rng,
      split=split,
      data_dir=data_dir,
      global_batch_size=global_batch_size)

  for _ in range(1):

    batch = next(input_queue)
    print(f"RANK {RANK} got batch")

    assert type(batch) == dict
    assert 'inputs' in batch
    assert 'targets' in batch

    inputs, targets = batch['inputs'], batch['targets']
    print(f"RANK {RANK} inputs.shape: {inputs.shape}")
    print(f"RANK {RANK} targets.shape: {targets.shape}")
    print(f"RANK {RANK} type(inputs): {type(inputs)}")

    jax.debug.inspect_array_sharding(inputs, callback=print)
    assert inputs.dtype == dtype
    assert targets.dtype == dtype

    assert inputs.shape == (global_batch_size, seq_len)
    assert targets.shape == (global_batch_size, seq_len)

    assert jnp.equal(inputs[:, 1:], targets[:, :-1]).all()
    print(f"RANK {RANK} inputs[0, :10]: {inputs[0, :10]}")

  print(f"=== ALL TEST PASSED ===")


def main():
  profiler = PassThroughProfiler()
  test_dataloader_jax()


if __name__ == '__main__':
  main()
