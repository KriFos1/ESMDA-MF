"""Detach a controller command and record its lifecycle in a status file."""

import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

terminated = {'flag': False}


def _record_termination(signum, frame):
    terminated['flag'] = True
    raise SystemExit(1)


signal.signal(signal.SIGTERM, _record_termination)


def main():
    status_path = Path(sys.argv[1])
    command = sys.argv[2:]
    status = {'state': 'RUNNING', 'supervisor_pid': os.getpid(), 'command': command,
              'cwd': os.getcwd(),
              'started_utc': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S%z')}
    status_path.write_text(json.dumps(status, indent=2))
    started = time.monotonic()
    try:
        code = subprocess.call(command)
    except SystemExit:
        code = 1
    except BaseException as exc:
        code = 1
        status['error'] = str(exc)
    if terminated['flag']:
        status['error'] = status.get('error', 'terminated by signal')
    status.update(state='COMPLETED' if code == 0 and not terminated['flag'] else 'FAILED',
                  returncode=code,
                  ended_utc=datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S%z'),
                  elapsed_seconds=time.monotonic() - started)
    status_path.write_text(json.dumps(status, indent=2))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
