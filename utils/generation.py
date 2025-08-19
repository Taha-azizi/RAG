import shutil
import subprocess
import os
os.environ["PYTHONIOENCODING"] = "utf-8"

def generate_with_ollama(model: str, prompt: str, timeout: int = 120) -> str:
    """Call Ollama locally to generate a completion for the given prompt."""
    if shutil.which("ollama") is None:
        raise RuntimeError("Ollama CLI not found. Install from https://ollama.com and ensure it is on your PATH.")
    cmd = ["ollama", "run", model, prompt]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, encoding="utf-8")
    if res.returncode != 0:
        raise RuntimeError(f"Ollama error: {res.stderr}")
    return res.stdout.strip()
