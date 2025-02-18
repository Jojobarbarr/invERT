from pathlib import Path

num: int = 2
events: str = "DYKE_DYKE_DYKE"
filepath: Path = Path(f"../../../dataset/{num}/models_by_code/models/{events}")

files = filepath.glob("*")
for file in files:
    if not file.suffixes == [".g12", ".gz"]:
        file.unlink()
