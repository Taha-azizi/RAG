from pathlib import Path
from typing import List, Tuple

def load_text_files(folder_path: str, pattern: str = "*.txt") -> Tuple[List[str], List[str]]:
    """Load all text files under folder_path matching pattern.
    Returns a tuple of (texts, sources).
    """
    texts: List[str] = []
    sources: List[str] = []
    path = Path(folder_path)
    if not path.exists():
        raise FileNotFoundError(f"Data folder not found: {folder_path}")
    for file in sorted(path.glob(pattern)):
        try:
            with open(file, "r", encoding="utf-8") as f:
                texts.append(f.read())
                sources.append(str(file))
        except Exception as e:
            print(f"[WARN] Could not read {file}: {e}")
    if not texts:
        raise FileNotFoundError(f"No files matched pattern {pattern} in {folder_path}")
    return texts, sources
