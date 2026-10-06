from types import SimpleNamespace

import numpy as np


def test_array_polling_cap_and_default(monkeypatch, tmp_path):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'Dashboard/src'))
    from pxwf.execution.backend import SlurmBackend
    import pxwf.execution.backend as backend

    sleeps = []
    monkeypatch.setattr(backend.time, 'sleep', sleeps.append)
    handle = SimpleNamespace(batch_id='fake', job_id='fake', directory=str(tmp_path))

    def run(metadata):
        calls = [0]

        def poll(handle):
            calls[0] += 1
            return {'connected': True, 'state': 'READY' if calls[0] == 10 else 'RUNNING'}

        fake = SimpleNamespace(mode='array', transport=SimpleNamespace(attempts=[]), store=tmp_path,
                               submit_batch=lambda *args: handle, poll=poll,
                               collect=lambda *args, **kwargs: (np.ones((1, 8)), []),
                               _post_collect=lambda *args: None,
                               _transport_usage=lambda *args: (0, 0))
        SlurmBackend.evaluate_batch(fake, metadata, np.zeros((1, 2, 2, 2)))

    run({'poll_max_delay_seconds': 5.})
    assert max(sleeps) == 5.
    sleeps.clear()
    run({})
    assert max(sleeps) == 30.
