#!/usr/bin/env python3
"""Benchmark step times for JAX and PyTorch across all workloads.

This script runs each workload for 101 steps with both JAX and PyTorch,
captures the step_time_ms metric, and produces a comparison table.
"""

import argparse
import re
import subprocess
from pathlib import Path

# Base workloads to benchmark
WORKLOADS = [
  'imagenet_resnet',
]

FRAMEWORKS = ['jax', 'pytorch']
MAX_STEPS = 201
OUTPUT_DIR = Path('/home/ak4605/aef2/benchmark_outputs')


def get_data_dir(workload: str, framework: str) -> str:
  """Map workload to its data directory."""
  if workload in ['imagenet_resnet', 'imagenet_vit']:
    return '/opt/data/imagenet/' + framework
  elif workload in ['librispeech_conformer', 'librispeech_deepspeech']:
    return '/opt/data/librispeech'
  elif workload == 'criteo1tb':
    return '/opt/data/criteo1tb'
  elif workload == 'fastmri':
    return '/opt/data/fastmri'
  elif workload == 'ogbg':
    return '/opt/data/ogbg'
  elif workload == 'wmt':
    return '/opt/data/wmt'
  else:
    return '/opt/'


def run_workload(workload: str, framework: str, output_file: Path) -> bool:
  """Run a workload and capture output to file."""
  data_dir = get_data_dir(workload, framework)
  experiment_dir = '/home/ak4605/experiments'

  # Clean up previous experiment directories
  for item in Path(experiment_dir).glob(f'{workload}*'):
    if item.is_dir():
      subprocess.run(['rm', '-rf', str(item)], check=True)

  # Build command based on framework
  submission_path = (
    f'algorithms/baselines/external_tuning/{framework}_nadamw_full_budget.py'
  )
  tuning_search_space = (
    'algorithms/baselines/external_tuning/tuning_search_space.json'
  )

  if framework == 'jax':
    cmd = [
      'python',
      'submission_runner.py',
      f'--framework={framework}',
      f'--workload={workload}',
      f'--data_dir={data_dir}',
      f'--experiment_dir={experiment_dir}',
      f'--experiment_name={workload}_benchmark',
      f'--submission_path={submission_path}',
      f'--tuning_search_space={tuning_search_space}',
      f'--max_global_steps={MAX_STEPS}',
      '--skip_evals',
      '--nosave_checkpoints',
      '--nosave_intermediate_checkpoints',
    ]
    # For JAX, activate the jax conda environment
    activate_cmd = 'source $(conda info --base)/etc/profile.d/conda.sh && conda activate ap11_jax && '
  else:
    cmd = [
      'torchrun',
      '--nproc_per_node=4',
      '--standalone',
      'submission_runner.py',
      f'--framework={framework}',
      f'--workload={workload}',
      f'--data_dir={data_dir}',
      f'--experiment_dir={experiment_dir}',
      f'--experiment_name={workload}_benchmark',
      f'--submission_path={submission_path}',
      f'--tuning_search_space={tuning_search_space}',
      f'--max_global_steps={MAX_STEPS}',
      '--skip_evals',
      '--nosave_checkpoints',
      '--nosave_intermediate_checkpoints',
    ]
    # For PyTorch, activate the torch conda environment
    activate_cmd = 'source $(conda info --base)/etc/profile.d/conda.sh && conda activate ap11_torch_latest && '

  # Run the command with shell to handle conda activation
  full_cmd = activate_cmd + ' '.join(cmd)
  print(f'Running: {workload} with {framework}')
  print(f'Output will be saved to: {output_file}')

  with open(output_file, 'w') as f:
    result = subprocess.run(
      full_cmd,
      shell=True,
      executable='/bin/bash',
      stdout=f,
      stderr=subprocess.STDOUT,
      cwd='/home/ak4605/aef2/',
    )

  return result.returncode == 0


def parse_step_time(output_file: Path) -> float | None:
  """Parse the last step_time_ms from output file."""
  if not output_file.exists():
    return None

  with open(output_file, 'r') as f:
    content = f.read()

  # Find all step_time_ms values
  # Pattern matches: step_time_ms=123.456 or 'step_time_ms': 123.456
  pattern = r'step_time_ms[=:]\s*([\d.]+)'
  matches = re.findall(pattern, content)

  if matches:
    # Return the last value (most recent EMA)
    return float(matches[-1])
  return None


def parse_args():
  parser = argparse.ArgumentParser(
    description='Benchmark step times for JAX and PyTorch across workloads.'
  )
  group = parser.add_mutually_exclusive_group()
  group.add_argument(
    '--torch-only',
    action='store_true',
    help='Only run PyTorch experiments; read existing JAX results from files.',
  )
  group.add_argument(
    '--jax-only',
    action='store_true',
    help='Only run JAX experiments; read existing PyTorch results from files.',
  )
  group.add_argument(
    '--just-read',
    action='store_true',
    help='Do not run any experiments; just read and compare existing outputs.',
  )
  return parser.parse_args()


def main():
  args = parse_args()

  # Create output directory
  OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

  results = {}

  # Determine which frameworks to run vs read from files
  if args.just_read:
    frameworks_to_run = []
    frameworks_to_read = FRAMEWORKS
  elif args.torch_only:
    frameworks_to_run = ['pytorch']
    frameworks_to_read = ['jax']
  elif args.jax_only:
    frameworks_to_run = ['jax']
    frameworks_to_read = ['pytorch']
  else:
    frameworks_to_run = FRAMEWORKS
    frameworks_to_read = []

  # Run all workloads
  for workload in WORKLOADS:
    results[workload] = {}

    # Read existing results from files
    for framework in frameworks_to_read:
      output_file = OUTPUT_DIR / f'{workload}_{framework}.out'
      step_time = parse_step_time(output_file)
      results[workload][framework] = step_time
      if step_time:
        print(f'\nLoaded existing {framework.upper()} result for {workload}: {step_time:.2f} ms')
      else:
        print(f'\nNo existing {framework.upper()} result found for {workload}')

    # Run experiments for specified frameworks
    for framework in frameworks_to_run:
      output_file = OUTPUT_DIR / f'{workload}_{framework}.out'

      print(f'\n{"=" * 60}')
      print(f'Benchmarking {workload} with {framework}')
      print(f'{"=" * 60}')

      success = run_workload(workload, framework, output_file)

      if success:
        step_time = parse_step_time(output_file)
        results[workload][framework] = step_time
        print(
          f'Step time: {step_time:.2f} ms' if step_time else 'Step time: N/A'
        )
      else:
        results[workload][framework] = None
        print(f'Failed to run {workload} with {framework}')

  # Print results table
  print('\n\n')
  print('=' * 80)
  print('STEP TIME COMPARISON (ms)')
  print('=' * 80)
  print(
    f'{"Workload":<30} {"JAX (ms)":<15} {"PyTorch (ms)":<15} {"Ratio (PT/JAX)":<15}'
  )
  print('-' * 80)

  for workload in WORKLOADS:
    jax_time = results[workload].get('jax')
    pytorch_time = results[workload].get('pytorch')

    jax_str = f'{jax_time:.2f}' if jax_time else 'N/A'
    pytorch_str = f'{pytorch_time:.2f}' if pytorch_time else 'N/A'

    if jax_time and pytorch_time:
      ratio = pytorch_time / jax_time
      ratio_str = f'{ratio:.2f}x'
    else:
      ratio_str = 'N/A'

    print(f'{workload:<30} {jax_str:<15} {pytorch_str:<15} {ratio_str:<15}')

  print('=' * 80)

  # Save results to file
  results_file = OUTPUT_DIR / 'results.txt'
  with open(results_file, 'w') as f:
    f.write('STEP TIME COMPARISON (ms)\n')
    f.write('=' * 80 + '\n')
    f.write(
      f'{"Workload":<30} {"JAX (ms)":<15} {"PyTorch (ms)":<15} {"Ratio (PT/JAX)":<15}\n'
    )
    f.write('-' * 80 + '\n')

    for workload in WORKLOADS:
      jax_time = results[workload].get('jax')
      pytorch_time = results[workload].get('pytorch')

      jax_str = f'{jax_time:.2f}' if jax_time else 'N/A'
      pytorch_str = f'{pytorch_time:.2f}' if pytorch_time else 'N/A'

      if jax_time and pytorch_time:
        ratio = pytorch_time / jax_time
        ratio_str = f'{ratio:.2f}x'
      else:
        ratio_str = 'N/A'

      f.write(
        f'{workload:<30} {jax_str:<15} {pytorch_str:<15} {ratio_str:<15}\n'
      )

    f.write('=' * 80 + '\n')

  print(f'\nResults saved to: {results_file}')


if __name__ == '__main__':
  main()
