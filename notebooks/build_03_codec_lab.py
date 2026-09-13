"""Generate notebooks/03_codec_lab.ipynb. Source of cells: build_codec_labs.py."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_codec_labs import write_all

if __name__ == "__main__":
    write_all()
