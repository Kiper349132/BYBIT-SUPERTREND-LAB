import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
# Never write logs/results/research into the program folder during tests.
os.environ.setdefault("LAB_HOME", tempfile.mkdtemp(prefix="lab_home_"))
