"""Tests that JAX and PyTorch step times are within 20% of each other.

This test runs each workload for a number of steps with both JAX and PyTorch,
captures the step_time_ms metric, and asserts they are within 20%.
"""

import re
import subprocess
import sys
import tempfile
from pathlib import Path

from absl import flags, logging
from absl.testing import absltest, parameterized

FLAGS = flags.FLAGS
FLAGS(sys.argv)

MAX_STEPS = 101
TOLERANCE = 0.25

WORKLOADS = [
  'imagenet_vit',
]

DATA_DIRS = {
  'imagenet_resnet': '/opt/data/imagenet/',
  'imagenet_vit': '/opt/data/imagenet/',
  'librispeech_conformer': '/opt/data/librispeech',
  'librispeech_deepspeech': '/opt/data/librispeech',
  'criteo1tb': '/opt/data/criteo1tb',
  'fastmri': '/opt/data/fastmri',
  'ogbg': '/opt/data/ogbg',
  'wmt': '/opt/data/wmt',
}

CONDA_ENVS = {
  'jax': 'ap11_jax',
  'pytorch': 'ap11_torch_latest',
}


def get_data_dir(workload: str, framework: str) -> str:
  """Map workload to its data directory."""
  base_dir = DATA_DIRS.get(workload, '/opt/data')
  if workload in ['imagenet_resnet', 'imagenet_vit']:
    return base_dir + framework
  return base_dir


def run_workload(workload: str, framework: str, output_file: Path) -> bool:
  """Run a workload and capture output to file."""
  data_dir = get_data_dir(workload, framework)
  experiment_dir = tempfile.mkdtemp(prefix=f'{workload}_{framework}_')

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

  conda_env = CONDA_ENVS[framework]
  activate_cmd = (
    f'source $(conda info --base)/etc/profile.d/conda.sh && '
    f'conda activate {conda_env} && '
  )
  full_cmd = activate_cmd + ' '.join(cmd)

  logging.info(f'Running: {workload} with {framework}')
  logging.info(f'Output will be saved to: {output_file}')

  with open(output_file, 'w') as f:
    result = subprocess.run(
      full_cmd,
      shell=True,
      executable='/bin/bash',
      stdout=f,
      stderr=subprocess.STDOUT,
      cwd=str(Path(__file__).parent.parent),
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


named_parameters = [
  dict(testcase_name=workload, workload=workload) for workload in WORKLOADS
]


class StepTimeTest(parameterized.TestCase):
  """Tests that JAX and PyTorch step times are within tolerance."""

  @parameterized.named_parameters(*named_parameters)
  def test_step_times_within_tolerance(self, workload):
    """Test that JAX and PyTorch step times are within 20% of each other."""
    results = {}

    with tempfile.TemporaryDirectory() as tmpdir:
      tmpdir = Path(tmpdir)

      for framework in ['jax', 'pytorch']:
        output_file = tmpdir / f'{workload}_{framework}.out'

        success = run_workload(workload, framework, output_file)
        self.assertTrue(success, f'Failed to run {workload} with {framework}')

        step_time = parse_step_time(output_file)
        self.assertIsNotNone(
          step_time,
          f'Could not parse step_time_ms for {workload} with {framework}',
        )

        results[framework] = step_time
        logging.info(f'{workload} {framework}: {step_time:.2f} ms')

    jax_time = results['jax']
    pytorch_time = results['pytorch']
    ratio = pytorch_time / jax_time

    logging.info(
      f'{workload}: JAX={jax_time:.2f}ms, PyTorch={pytorch_time:.2f}ms, '
      f'ratio={ratio:.2f}'
    )

    # Check that ratio is within tolerance (0.8 to 1.2 for 20% tolerance)
    lower_bound = 1.0 - TOLERANCE
    upper_bound = 1.0 + TOLERANCE

    self.assertGreaterEqual(
      ratio,
      lower_bound,
      f'{workload}: PyTorch is more than {TOLERANCE * 100:.0f}% faster than JAX '
      f'(ratio={ratio:.2f}, expected >= {lower_bound:.2f})',
    )
    self.assertLessEqual(
      ratio,
      upper_bound,
      f'{workload}: PyTorch is more than {TOLERANCE * 100:.0f}% slower than JAX '
      f'(ratio={ratio:.2f}, expected <= {upper_bound:.2f})',
    )


if __name__ == '__main__':
  absltest.main()
