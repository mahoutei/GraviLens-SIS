import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_cli_help():
    r = subprocess.run([sys.executable, '-m', 'gravilens_sis.main', '--help'],
                       capture_output=True, text=True, cwd=str(ROOT / 'GraviLens'))
    assert r.returncode == 0, r.stderr
    assert 'usage' in r.stdout.lower()
