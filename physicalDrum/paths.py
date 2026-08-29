from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data"
OUTPUT = ROOT / "output"
SCRATCH = ROOT / "scratch"


def ensure_dirs():
    for d in (DATA, OUTPUT):
        d.mkdir(parents=True, exist_ok=True)