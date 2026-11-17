"""Run this study from any current working directory."""
import os
import sys
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parent))
from tfim.workspace import main
if __name__ == '__main__':
    main(BASE)
