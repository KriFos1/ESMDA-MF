import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

import numpy as np

import pilot_rml_index45 as runner
from rml_standalone import atomic_json, atomic_npz


def prepared_fixture(directory):
    atomic_npz(directory / 'inputs.npz', prior_centers=np.zeros((25, 8)),
               observations=np.zeros(2), variance=np.ones(2), targets=np.zeros((25, 2)))
    study = {'count': 25, 'index': 45, 'dims': [2, 1, 2],
             'rh_limits': [-1., np.log(40.)],
             'inputs_sha256': runner.digest(directory / 'inputs.npz'),
             'source_sha256': {name: runner.digest(runner.ROOT / name) for name in runner.MODULES}}
    atomic_json(directory / 'study.json', study)
    runner.write_array(directory, {'count': 25, 'index': 45, 'simulator': {'cpus': 5}})
    return study


def test_deployment_archive_imports_without_workspace_dependencies(tmp_path):
    prepared_fixture(tmp_path)
    runner.bundle(argparse.Namespace(run_dir=tmp_path))
    extracted = tmp_path / 'extracted'
    extracted.mkdir()
    with tarfile.open(tmp_path / 'deployment.tar.gz') as archive:
        archive.extractall(extracted, filter='data')
    code = extracted / 'code'
    environment = {**os.environ, 'PYTHONPATH': str(code) + os.pathsep + str(code / 'pxwf-src')}
    checked = subprocess.run(
        [sys.executable, '-c', 'import pilot_rml_index45, run_DA_sequential; '
         'from pxwf.execution.backend import LocalBackend; '
         'pilot_rml_index45.read_study(__import__("pathlib").Path("../run"))'],
        cwd=code, env=environment, text=True, capture_output=True)
    assert checked.returncode == 0, checked.stderr
    syntax = subprocess.run(['bash', '-n', str(extracted / 'run/rml-array.sbatch')],
                            capture_output=True, text=True)
    assert syntax.returncode == 0, syntax.stderr


def test_partial_collection_preserves_missing_member_columns(tmp_path):
    prepared_fixture(tmp_path)
    member = tmp_path / 'member-007'
    member.mkdir()
    atomic_npz(member / 'fit.npz', model=np.ones(8), prediction=np.ones(2))
    atomic_json(member / 'result.json', {'member': 7, 'converged': True,
                                        'reason': 'gradient_tolerance',
                                        'identity': runner.digest(tmp_path / 'study.json')})
    runner.collect(argparse.Namespace(run_dir=tmp_path, verify_mean=False))
    with np.load(tmp_path / 'rml-ensemble.npz') as ensemble:
        assert ensemble['raw_models'].shape == (8, 25)
        np.testing.assert_array_equal(ensemble['raw_models'][:, 7], np.ones(8))
        assert np.isnan(ensemble['raw_models'][:, 6]).all()
        assert ensemble['converged'].sum() == 1
    summary = json.loads((tmp_path / 'summary.json').read_text())
    assert not summary['complete']
    assert summary['members'][6]['reason'] == 'missing'
    assert not (tmp_path / 'moments.npz').exists()


def test_lm_array_requests_candidate_steps_on_physical_cores(tmp_path):
    study = {'count': 25, 'index': 45, 'resources': {'memory_gib': 48, 'walltime': '01:00:00'},
             'simulator': {'cpus': 8, 'ranks': 2, 'concurrency': 4, 'physical_cores': True}}
    runner.write_array(tmp_path, study)
    script = (tmp_path / 'rml-array.sbatch').read_text()
    # 8 physical cores per task = 16 Slurm CPUs (hyperthreads): all 4 two-rank steps fit.
    for line in ('#SBATCH --array=0-24\n', '#SBATCH --ntasks=8\n', '#SBATCH --cpus-per-task=16\n',
                 '#SBATCH --mem=48G\n', '#SBATCH --time=01:00:00\n'):
        assert line in script
    assert '--hint' not in script
    assert subprocess.run(['bash', '-n', str(tmp_path / 'rml-array.sbatch')]).returncode == 0


def test_lm_simulator_config_builds_srun_step_resources(monkeypatch, tmp_path):
    from esmda_remote2d import SimBackendConfig, make_remote2d
    monkeypatch.setenv('SLURM_JOB_ID', '1')
    config = SimBackendConfig(backend='local', cpus=8, ranks=2, concurrency=4, mkl_threads=8,
                              mkl_cbwr='AVX2,STRICT', srun_steps=True, physical_cores=True)
    backend = make_remote2d(config, (2, 5, 2), 5., 5., 5., [('12kHz', '43ft')]).backend()
    assert backend.srun_steps and backend.workers == 4
    resources = backend.resources
    assert (resources.ranks, resources.cpus_per_rank, resources.mkl_threads) == (2, 8, 8)
    assert resources.srun_options == ('--exact', '--mem=0', '--hint=nomultithread')
    assert resources.solver_environment == {'MKL_CBWR': 'AVX2,STRICT'}
    import pytest
    with pytest.raises(ValueError, match='srun_steps'):
        make_remote2d(SimBackendConfig(backend='local', ranks=2), (2, 5, 2), 5., 5., 5.,
                      [('12kHz', '43ft')]).backend()
