"""Deploy the ESMDA-MF 2D simulator on the remotes, like the Dashboard does.

The Dashboard runs its 2D forward batches through ``pxwf.execution``
(``LocalBackend`` / ``SlurmBackend`` / ``NodeBackend``) with a standalone
``fwd2D`` worker. This module reuses that exact worker so the ESMDA-MF
sequential run can select, per run:

* ``local``  -- the identical worker in isolated local processes
               (``fwd2D`` on the local PATH, OneAPI/HDFql auto-discovered);
* ``slurm``  -- a Slurm-based remote HPC cluster (``sbatch`` array jobs over
               the ``--host`` SSH alias);
* ``node``   -- a scheduler-less cluster reached over a traditional SSH
               pipeline (detached task pool on one node per batch).

Selection is through CLI arguments / environment variables:
    --sim-backend {local,slurm,node}   (env DA_SIM_BACKEND)
    --host, --root, --python, --partition, --account, --qos,
    --nodes, --ranks, --cpus, --memory, --walltime, --concurrency,
    --chunk, --workers, --oneapi, --hdfql, --binary-dir, --mpi, --pmi,
    --scratch   (env DA_SIM_HOST, DA_SIM_ROOT, ...)

The worker speaks the Dashboard wire format: one (2, rows, cols)
log-resistivity tile per member, and a per-tool UDAR (8-component) response.
Predictions use canonical UDAR order. Full-grid exact-adjoint derivatives
are with respect to ln(Rh), ln(Rv); their cell axes are transposed into the
optimizer's (nx, nz), z-fastest layout.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

# The UDAR geo-signal layout the worker returns per tool, in the solver's
# canonical order (matches ``pxwf.execution.worker.COMPONENTS``).
UDAR_COMPONENTS = ("USDP", "USDA", "UADP", "UADA", "UHRP", "UHRA", "UHAP", "UHAA")


def classify_solver_start_failure(exc):
    """Classify confirmed single-field 2D solver failures from saved receipts."""
    if not isinstance(exc, RuntimeError):
        return None
    match = re.fullmatch(r"Batch ([0-9a-f]{32}): FAILED\. Details: (.+/status\.json)", str(exc))
    if match is None:
        return None
    batch_id, status_path = match.groups()
    try:
        status = json.loads(Path(status_path).read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(status, dict):
        return None
    evaluations = status.get('evaluations', [])
    if not isinstance(evaluations, list) or len(evaluations) != 1 or not isinstance(evaluations[0], dict):
        return None
    error = evaluations[0].get('error')
    if (status.get('batch_id') != batch_id or status.get('state') != 'FAILED'
            or not status.get('connected', True)
            or evaluations[0].get('batch_id') != batch_id
            or evaluations[0].get('state') != 'FAILED' or not isinstance(error, str)):
        return None
    if error.startswith('CalledProcessError:'):
        exit_code = re.search(r'\bexit status (174|135|139)\b', error)
        if exit_code is not None:
            return f'exit_{exit_code.group(1)}'
    if error == 'RuntimeError: Solver reported a convergence error; see convergence_error.txt':
        return 'solver_convergence'
    return None


def is_solver_exit_174(exc):
    return classify_solver_start_failure(exc) == 'exit_174'


def _env(name, default=None):
    value = os.environ.get(name)
    return value if value not in (None, "") else default


def _positive_int(value, default):
    try:
        parsed = int(value)
        return parsed if parsed >= 1 else default
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class SimBackendConfig:
    backend: str = "local"
    host: str = "HPC"
    root: str = "pxwf-runs"
    python: str = "pxwf-runtime/.venv/bin/python"
    partition: str = "comp"
    account: str = ""
    qos: str = ""
    nodes: int = 1
    ranks: int = 1
    cpus: int = 8
    memory: int = 16
    walltime: str = "01:00:00"
    concurrency: int = 4
    chunk: int = 1
    workers: int = 8
    oneapi: str = ""
    hdfql: str = ""
    binary_dir: str = ""
    mpi: str = "pmi2"
    pmi: str = "/usr/lib64/libpmi2.so"
    scratch: str = ""
    jobs_dir: str = ""
    solver_tolerance: float = 1e-6
    poll_max_delay_seconds: float = 5.
    reconnect_grace_seconds: float = 180.
    solver_grid: tuple = (20, 4, 40, 30, 20, 30, 2, 1, 2)
    # Local backend inside a Slurm allocation: run each solve as an `srun` job
    # step with `ranks` MPI ranks x `cpus` CPUs, MKL-threaded DGEMM and a
    # pinned MKL code path. These change the floating-point forward result,
    # so they are part of a study's identity.
    mkl_threads: int = 1
    mkl_cbwr: str = ""
    srun_steps: bool = False
    physical_cores: bool = False
    # Material TI axis of the solver's inner grid: "gradient" (legacy: dominant
    # gradient of -2 ln Rh - ln Rv per solver cell) or "bedding" (well-dip/bedding
    # axis everywhere; needs a build whose adjoint spec declares the input).
    orientation: str = "gradient"

    @classmethod
    def from_env(cls):
        return cls(
            backend=str(_env("DA_SIM_BACKEND", "local")).lower(),
            host=_env("DA_SIM_HOST", "HPC"),
            root=_env("DA_SIM_ROOT", "pxwf-runs"),
            python=_env("DA_SIM_PYTHON", "pxwf-runtime/.venv/bin/python"),
            partition=_env("DA_SIM_PARTITION", "comp"),
            account=_env("DA_SIM_ACCOUNT", ""),
            qos=_env("DA_SIM_QOS", ""),
            nodes=_positive_int(_env("DA_SIM_NODES"), 1),
            ranks=_positive_int(_env("DA_SIM_RANKS"), 1),
            cpus=_positive_int(_env("DA_SIM_CPUS"), 8),
            memory=_positive_int(_env("DA_SIM_MEMORY"), 16),
            walltime=_env("DA_SIM_WALLTIME", "01:00:00"),
            concurrency=_positive_int(_env("DA_SIM_CONCURRENCY"), 4),
            chunk=_positive_int(_env("DA_SIM_CHUNK"), 1),
            workers=_positive_int(_env("DA_SIM_WORKERS"), 8),
            oneapi=_env("DA_SIM_ONEAPI", ""),
            hdfql=_env("DA_SIM_HDFQL", ""),
            binary_dir=_env("DA_SIM_BINARY_DIR", ""),
            mpi=_env("DA_SIM_MPI", "pmi2"),
            pmi=_env("DA_SIM_PMI", "/usr/lib64/libpmi2.so"),
            scratch=_env("DA_SIM_SCRATCH", ""),
            jobs_dir=_env("DA_SIM_JOBS_DIR", ""),
            solver_tolerance=float(_env("DA_SIM_TOLERANCE", 1e-6)),
            poll_max_delay_seconds=float(_env("DA_SIM_POLL_MAX_DELAY", 5.)),
            reconnect_grace_seconds=float(_env("DA_SIM_RECONNECT_GRACE", 180.)),
            solver_grid=tuple(int(v) for v in
                              _env("DA_SIM_SOLVER_GRID", "20,4,40,30,20,30,2,1,2").split(",")),
            orientation=_env("DA_SIM_ORIENTATION", "gradient"),
        )

    def add_cli(self, parser):
        parser.add_argument("--sim-backend", choices=["local", "slurm", "node"],
                            default=self.backend,
                            help="Where the 2D simulator runs (default: %(default)s)")
        parser.add_argument("--host", default=self.host,
                            help="SSH host alias for slurm/node backends")
        parser.add_argument("--root", default=self.root,
                            help="Remote work directory (relative to $HOME or absolute)")
        parser.add_argument("--python", dest="sim_python", default=self.python,
                            help="Remote project-environment Python (numpy+h5py)")
        parser.add_argument("--partition", default=self.partition, help="Slurm partition")
        parser.add_argument("--account", default=self.account, help="Slurm account")
        parser.add_argument("--qos", default=self.qos, help="Slurm QoS")
        parser.add_argument("--nodes", type=int, default=self.nodes)
        parser.add_argument("--ranks", type=int, default=self.ranks)
        parser.add_argument("--cpus", type=int, default=self.cpus)
        parser.add_argument("--memory", type=int, default=self.memory)
        parser.add_argument("--walltime", default=self.walltime)
        parser.add_argument("--concurrency", type=int, default=self.concurrency)
        parser.add_argument("--chunk", type=int, default=self.chunk)
        parser.add_argument("--workers", type=int, default=self.workers,
                            help="Concurrent evaluations per node (node backend)")
        parser.add_argument("--oneapi", default=self.oneapi,
                            help="OneAPI setup script (blank: discover)")
        parser.add_argument("--hdfql", default=self.hdfql,
                            help="HDFql directory (blank: discover)")
        parser.add_argument("--binary-dir", dest="binary_dir", default=self.binary_dir,
                            help="fwd2D directory (blank: PATH / ~/.local/bin)")
        parser.add_argument("--mpi", default=self.mpi, help="srun MPI plugin")
        parser.add_argument("--pmi", default=self.pmi, help="PMI library")
        parser.add_argument("--scratch", default=self.scratch,
                             help="Node-local scratch root (blank: shared FS)")
        parser.add_argument("--solver-tolerance", type=float, default=self.solver_tolerance)
        parser.add_argument("--poll-max-delay", type=float, default=self.poll_max_delay_seconds,
                            help="Maximum array-job polling sleep, seconds")
        parser.add_argument("--reconnect-grace", type=float, default=self.reconnect_grace_seconds,
                            help="Seconds a poll may stay disconnected before manual reconnect")
        parser.add_argument("--solver-grid", default=",".join(str(v) for v in self.solver_grid),
                            metavar="UX,UY,UZ,OX,OY,OZ,AX,AY,AZ",
                            help="fwd2D custom grid: uniform and optimal node counts and "
                                 "material averaging (default: %(default)s)")
        parser.add_argument("--orientation", choices=["gradient", "bedding"], default=self.orientation,
                            help="Material TI axis: legacy per-cell gradient axis, or the bedding axis "
                                 "everywhere (needs an orientation-capable build)")

    def from_args(self, args):
        kwargs = dict(
            backend=args.sim_backend,
            host=args.host,
            root=args.root,
            python=args.sim_python,
            partition=args.partition,
            account=args.account,
            qos=args.qos,
            nodes=args.nodes,
            ranks=args.ranks,
            cpus=args.cpus,
            memory=args.memory,
            walltime=args.walltime,
            concurrency=args.concurrency,
            chunk=args.chunk,
            workers=args.workers,
            oneapi=args.oneapi,
            hdfql=args.hdfql,
            binary_dir=args.binary_dir,
            mpi=args.mpi,
            pmi=args.pmi,
            scratch=args.scratch,
            solver_tolerance=args.solver_tolerance,
            poll_max_delay_seconds=args.poll_max_delay,
            reconnect_grace_seconds=args.reconnect_grace,
            solver_grid=tuple(int(v) for v in args.solver_grid.split(",")),
            orientation=args.orientation,
        )
        return SimBackendConfig(**{**asdict(self), **kwargs})

    def to_env(self):
        """Environment mapping that reproduces this config via ``from_env``.

        Lets a parent harness (e.g. ``validate_gate.py``) drive the backend and
        every slurm knob for child ``run_DA_sequential`` processes through the
        environment instead of repeating per-run CLI flags.
        """
        values = {
            "backend": self.backend, "host": self.host, "root": self.root,
            "python": self.python, "partition": self.partition,
            "account": self.account, "qos": self.qos, "nodes": self.nodes,
            "ranks": self.ranks, "cpus": self.cpus, "memory": self.memory,
            "walltime": self.walltime, "concurrency": self.concurrency,
            "chunk": self.chunk, "workers": self.workers, "oneapi": self.oneapi,
            "hdfql": self.hdfql, "binary_dir": self.binary_dir, "mpi": self.mpi,
            "pmi": self.pmi, "scratch": self.scratch, "jobs_dir": self.jobs_dir,
            "solver_tolerance": self.solver_tolerance,
            "poll_max_delay_seconds": self.poll_max_delay_seconds,
            "reconnect_grace_seconds": self.reconnect_grace_seconds,
            "solver_grid": ",".join(str(v) for v in self.solver_grid),
            "orientation": self.orientation,
        }
        return {
            "DA_SIM_BACKEND": values["backend"],
            "DA_SIM_HOST": values["host"],
            "DA_SIM_ROOT": values["root"],
            "DA_SIM_PYTHON": values["python"],
            "DA_SIM_PARTITION": values["partition"],
            "DA_SIM_ACCOUNT": values["account"],
            "DA_SIM_QOS": values["qos"],
            "DA_SIM_NODES": str(values["nodes"]),
            "DA_SIM_RANKS": str(values["ranks"]),
            "DA_SIM_CPUS": str(values["cpus"]),
            "DA_SIM_MEMORY": str(values["memory"]),
            "DA_SIM_WALLTIME": values["walltime"],
            "DA_SIM_CONCURRENCY": str(values["concurrency"]),
            "DA_SIM_CHUNK": str(values["chunk"]),
            "DA_SIM_WORKERS": str(values["workers"]),
            "DA_SIM_ONEAPI": values["oneapi"],
            "DA_SIM_HDFQL": values["hdfql"],
            "DA_SIM_BINARY_DIR": values["binary_dir"],
            "DA_SIM_MPI": values["mpi"],
            "DA_SIM_PMI": values["pmi"],
            "DA_SIM_SCRATCH": values["scratch"],
            "DA_SIM_JOBS_DIR": values["jobs_dir"],
            "DA_SIM_TOLERANCE": str(values["solver_tolerance"]),
            "DA_SIM_POLL_MAX_DELAY": str(values["poll_max_delay_seconds"]),
            "DA_SIM_RECONNECT_GRACE": str(values["reconnect_grace_seconds"]),
            "DA_SIM_SOLVER_GRID": values["solver_grid"],
            "DA_SIM_ORIENTATION": values["orientation"],
        }

class Remote2D:
    """Drive ESMDA-MF 2D forward/Jacobian calls through the Dashboard worker."""

    def __init__(self, config: SimBackendConfig, dims, dX, dY, dZ, tools, jobs_dir=None):
        self.config = config
        self.dims = tuple(int(v) for v in dims)
        self.dX, self.dY, self.dZ = float(dX), float(dY), float(dZ)
        self.tools = list(tools)
        self.jobs_dir = jobs_dir or config.jobs_dir
        self._backend = None
        if not 0 < config.solver_tolerance < 1:
            raise ValueError("Solver tolerance must lie between 0 and 1")
        if len(config.solver_grid) != 9:
            raise ValueError("solver_grid must be nine integers")
        if not np.isfinite(config.poll_max_delay_seconds) or config.poll_max_delay_seconds < 1:
            raise ValueError('Maximum polling delay must be finite and at least one second')
        if not np.isclose(self.dY, self.dX):
            raise ValueError("The worker requires dY == dX")
        if not self.tools:
            raise ValueError("At least one tool is required")

    def _import_pxwf(self):
        here = os.path.dirname(os.path.abspath(__file__))
        src = os.path.normpath(os.path.join(here, "..", "Dashboard", "src"))
        for path in (src, os.environ.get("PXWF_SRC", "")):
            if path and os.path.isdir(path) and path not in sys.path:
                sys.path.insert(0, path)
        try:
            from pxwf.execution.backend import (LocalBackend, NodeBackend,
                                                SlurmBackend)
            from pxwf.execution.config import RemoteProfile, SlurmResources
        except ImportError as exc:  # pragma: no cover - environment error
            raise RuntimeError(
                "Could not import the Dashboard execution package. Run this script "
                "with the Dashboard virtualenv (the one that already imports EMsim) "
                "or set PXWF_SRC to the Dashboard src/ directory."
            ) from exc
        return LocalBackend, NodeBackend, SlurmBackend, RemoteProfile, SlurmResources

    def backend(self):
        if self._backend is not None:
            return self._backend
        (LocalBackend, NodeBackend, SlurmBackend,
         RemoteProfile, SlurmResources) = self._import_pxwf()
        cfg = self.config
        store = self.jobs_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                              ".esmda-jobs")
        if cfg.backend == "local":
            overrides = {}
            if cfg.oneapi:
                overrides["oneapi"] = cfg.oneapi
            if cfg.hdfql:
                overrides["hdfql"] = cfg.hdfql
            if cfg.binary_dir:
                overrides["binary_dir"] = cfg.binary_dir
            if not cfg.srun_steps and (cfg.ranks != 1 or cfg.physical_cores):
                raise ValueError("Multiple MPI ranks or core binding need srun_steps")
            options = (("--exact", "--mem=0") + (("--hint=nomultithread",) if cfg.physical_cores else ())
                       if cfg.srun_steps else ())
            resources = SlurmResources(nodes=1, ranks=cfg.ranks, cpus_per_rank=cfg.cpus,
                                       concurrency=1, chunk_size=cfg.chunk,
                                       mkl_threads=cfg.mkl_threads, srun_options=options,
                                       solver_environment=({"MKL_CBWR": cfg.mkl_cbwr}
                                                           if cfg.mkl_cbwr else {}))
            self._backend = LocalBackend(store=store, resources=resources,
                                         overrides=overrides or None,
                                         workers=cfg.concurrency, srun_steps=cfg.srun_steps)
        elif cfg.backend == "slurm":
            profile = RemoteProfile(cfg.host, cfg.root, cfg.python, cfg.oneapi, cfg.hdfql,
                                    cfg.binary_dir, cfg.mpi, cfg.pmi, cfg.scratch)
            resources = SlurmResources(cfg.partition, cfg.account, cfg.qos, cfg.nodes,
                                       cfg.ranks, cfg.cpus, cfg.memory, cfg.walltime,
                                       cfg.concurrency, cfg.chunk)
            self._backend = SlurmBackend(profile, resources, store=store, mode="array")
        elif cfg.backend == "node":
            profile = RemoteProfile(cfg.host, cfg.root, cfg.python, cfg.oneapi, cfg.hdfql,
                                    cfg.binary_dir)
            resources = SlurmResources(cpus_per_rank=cfg.cpus, concurrency=cfg.concurrency,
                                       chunk_size=cfg.chunk)
            self._backend = NodeBackend(profile, resources, store=store, workers=cfg.workers)
        else:
            raise ValueError(f"Unknown sim backend: {cfg.backend!r}")
        return self._backend

    # -- geometry / metadata ------------------------------------------------

    def runtime_identity(self):
        backend = self.backend()
        if hasattr(backend, "probe"):
            return backend.probe("2D")
        from pxwf.execution import worker
        return worker.discover("2D", backend.overrides)

    def _spacing_m(self):
        ft_to_m = 1.0 / 3.28084
        return self.dX * ft_to_m, self.dZ * ft_to_m

    def _tool_pairs(self):
        pairs = []
        for tool in self.tools:
            freq, sub = ast.literal_eval(tool) if isinstance(tool, str) else tool
            pairs.append([str(freq), str(sub)])
        return pairs

    def build_metadata(self, logging_point_idx, shift, well, jacobian=False,
                       tolerate_failures=False):
        """Worker metadata for the tile centred on the current log point.

        ``shift`` is the ESMDA-MF ``UTA2D_sim.model['shift']`` (feet) and
        ``well`` the benchmark wellpath (metres) at ``logging_point_idx``.
        """
        nx, ny, nz = self.dims
        dx_m, dz_m = self._spacing_m()
        shift_m = {"x": float(shift["x"]) / 3.28084,
                   "y": float(shift["y"]) / 3.28084,
                   "z": float(shift["z"]) / 3.28084}
        well_x = float(np.asarray(well["X"]).ravel()[logging_point_idx])
        well_y = float(np.asarray(well["Y"]).ravel()[logging_point_idx])
        well_z = float(np.asarray(well["Z"]).ravel()[logging_point_idx])
        well_md = float(np.asarray(well["Distance"]).ravel()[logging_point_idx])
        inclination = float(np.asarray(well["Inclination"]).ravel()[logging_point_idx])
        azimuth = float(np.asarray(well["Azimuth"]).ravel()[logging_point_idx])
        return {
            "fidelity": "2D",
            "tools": self._tool_pairs(),
            "spacing": [dx_m, dz_m],
            "y_layers": ny,
            "origin": [shift_m["x"], shift_m["y"], shift_m["z"]],
            "well": {"X": well_x, "Y": well_y, "Z": well_z, "Distance": well_md,
                     "Inclination": inclination, "Azimuth": azimuth},
            "bit_column": nx // 2,
            "log_index": 0,
            "jacobian": bool(jacobian),
            "full_jacobian": bool(jacobian),
            "tolerate_failures": bool(tolerate_failures),
            "evaluation_timeout_seconds": 180.0,
            "solver_tolerance": self.config.solver_tolerance,
            "poll_max_delay_seconds": self.config.poll_max_delay_seconds,
            "reconnect_grace_seconds": self.config.reconnect_grace_seconds,
            "solver_grid": list(self.config.solver_grid),
            "orientation": self.config.orientation,
        }

    # -- field / result mapping --------------------------------------------

    def to_tile(self, param_dict):
        """(nx*nz) log(rh)/log(rv) columns -> one (2, nz, nx) worker tile.

        ``param_dict['rh']``/``['rv']`` are Fortran-ordered (nx, ny, nz) as the
        in-process wrapper builds them; the first y-slice is the 2D state and
        the worker wants (rows=nz, cols=nx) per field.
        """
        nx, ny, nz = self.dims
        rh = np.asarray(param_dict["rh"], dtype=float).reshape(nx, ny, nz, order="F")
        rv = np.asarray(param_dict["rv"], dtype=float).reshape(nx, ny, nz, order="F")
        return np.stack([rh[:, 0, :].T, rv[:, 0, :].T])

    def _prediction(self, response):
        response = np.asarray(response, dtype=float)
        if response.shape != (len(UDAR_COMPONENTS) * len(self.tools),) or not np.isfinite(response).all():
            raise ValueError("Expected a finite canonical UDAR response for every tool")
        return response

    def run_pred(self, param_dict, logging_point_idx, shift, well):
        metadata = self.build_metadata(logging_point_idx, shift, well, jacobian=False)
        fields = self.to_tile(param_dict)[None]
        collected = self.backend().evaluate_batch(metadata, fields)
        return self._prediction(collected[0][0])

    def run_pred_multi(self, param_dicts, logging_point_idx, shift, well,
                       evaluation_timeout_seconds=None):
        """Forward-only batch for independent checks of saved posterior fields."""
        if not param_dicts:
            return np.empty((0, len(UDAR_COMPONENTS) * len(self.tools)))
        metadata = self.build_metadata(logging_point_idx, shift, well, tolerate_failures=True)
        if evaluation_timeout_seconds is not None:
            metadata['evaluation_timeout_seconds'] = evaluation_timeout_seconds
        fields = np.stack([self.to_tile(param_dict) for param_dict in param_dicts])
        return self.backend().evaluate_batch(metadata, fields)[0]

    def run_grad(self, param_dict, logging_point_idx, shift, well):
        if not self.backend().trusts_ad("2D"):
            raise RuntimeError("2D gradients require an exact discrete-adjoint build with adjoint-spec.json")
        metadata = self.build_metadata(logging_point_idx, shift, well, jacobian=True)
        fields = self.to_tile(param_dict)[None]
        collected = self.backend().evaluate_batch(metadata, fields)
        predictions = collected[0]
        full_jacobians = collected[2] if len(collected) > 2 else None
        if full_jacobians is None:
            raise RuntimeError(
                "The worker did not return a full-grid Jacobian; rebuild the "
                "Dashboard worker with the full_jacobian output enabled.")
        pred_vec = self._prediction(predictions[0])
        j_rh, j_rv = self._adapt_full_jacobian(full_jacobians[0])
        return pred_vec, j_rh, j_rv

    def run_grad_multi(self, param_dicts, logging_point_idx, shift, well):
        """One batched forward+adjoint array job for several fields.

        Returns one ``(prediction, j_rh, j_rv)`` per input, with ``None``
        Jacobian blocks for fields the solver could not evaluate (the worker
        publishes NaN predictions for tolerated failures). Any inconsistency
        in the collected result degrades every affected field to a failure
        entry rather than aborting the caller.
        """
        if not self.backend().trusts_ad("2D"):
            raise RuntimeError("2D gradients require an exact discrete-adjoint build with adjoint-spec.json")
        if not param_dicts:
            return []
        metadata = self.build_metadata(logging_point_idx, shift, well, jacobian=True,
                                       tolerate_failures=True)
        fields = np.stack([self.to_tile(param_dict) for param_dict in param_dicts])
        collected = self.backend().evaluate_batch(metadata, fields)
        predictions = collected[0]
        full_jacobians = collected[2] if len(collected) > 2 else None
        if full_jacobians is None:
            raise RuntimeError(
                "The worker did not return a full-grid Jacobian; rebuild the "
                "Dashboard worker with the full_jacobian output enabled.")
        aligned = len(full_jacobians) == len(predictions)
        results = []
        for index, prediction in enumerate(predictions):
            jac = full_jacobians[index] if aligned else None
            if not np.isfinite(prediction).all() or jac is None or np.size(jac) == 0:
                results.append((prediction, None, None))
                continue
            pred_vec = self._prediction(prediction)
            j_rh, j_rv = self._adapt_full_jacobian(jac)
            results.append((pred_vec, j_rh, j_rv))
        return results

    def _adapt_full_jacobian(self, jac):
        """Reorder worker (nz, nx) blocks into optimizer (nx, nz) blocks."""
        jac = np.asarray(jac, dtype=float)
        nx, ny, nz = self.dims
        n_tools = len(self.tools)
        n_udar_per_tool = len(UDAR_COMPONENTS)
        npt = nx * nz
        if jac.shape != (n_udar_per_tool * n_tools, 2 * npt):
            raise ValueError(f"Unexpected full Jacobian shape {jac.shape}")
        if not np.isfinite(jac).all():
            raise ValueError("Non-finite full Jacobian")
        blocks = jac.reshape(n_tools * n_udar_per_tool, 2, nz, nx)
        blocks = blocks.transpose(0, 1, 3, 2).reshape(n_tools * n_udar_per_tool, 2, npt)
        return blocks[:, 0, :], blocks[:, 1, :]


def make_remote2d(config, dims, dX, dY, dZ, tools):
    return Remote2D(config, dims, dX, dY, dZ, tools)
