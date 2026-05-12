"""Wrapper around llama.cpp's convert_hf_to_gguf.py (spec §8)."""
from __future__ import annotations
import shutil
import subprocess
import os
from pathlib import Path


def export_to_gguf(checkpoint_dir: Path, out_path: Path, quant: str = "q8_0") -> None:
    """Convert a saved HF-compatible directory to a quantized GGUF.

    Requires `convert_hf_to_gguf.py` from llama.cpp on PATH (or LLAMA_CPP_DIR env).
    """
    llama_dir = os.environ.get("LLAMA_CPP_DIR")
    if llama_dir:
        script = Path(llama_dir) / "convert_hf_to_gguf.py"
        if not script.exists():
            raise RuntimeError(f"convert_hf_to_gguf.py not found at {script}")
        cmd = ["python", str(script), str(checkpoint_dir), "--outfile", str(out_path), "--outtype", quant]
    else:
        which = shutil.which("convert_hf_to_gguf.py")
        if which is None:
            raise RuntimeError(
                "convert_hf_to_gguf.py not found. Set LLAMA_CPP_DIR env or pip install llama-cpp tools."
            )
        cmd = ["python", which, str(checkpoint_dir), "--outfile", str(out_path), "--outtype", quant]
    subprocess.run(cmd, check=True)
