from pathlib import Path

BASE_DIR = Path(__file__).parents[1].resolve()
DATA_DIR = BASE_DIR / "data"
RESULT_DIR = BASE_DIR / "result"

DATA_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)
