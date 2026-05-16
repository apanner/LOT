#!/usr/bin/env python3
"""Colab dependency checks for the LOTUS external engine.

Colab usually ships a working CUDA PyTorch stack. Installing LOT's full
`requirements.txt` can downgrade or break that stack because it pins torch
and CUDA wheel indexes. This helper checks imports first and installs only
missing non-Torch runtime packages needed by LOTUS inference and the external
engine.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys


REQUIRED_IMPORTS: dict[str, str] = {
    "accelerate": "accelerate",
    "cv2": "opencv-python-headless",
    "diffusers": "diffusers",
    "easydict": "easydict",
    "ftfy": "ftfy",
    "huggingface_hub": "huggingface_hub",
    "imageio": "imageio",
    "imageio_ffmpeg": "imageio-ffmpeg",
    "Imath": "OpenEXR",
    "numpy": "numpy",
    "omegaconf": "omegaconf",
    "OpenEXR": "OpenEXR",
    "peft": "peft",
    "PIL": "Pillow",
    "safetensors": "safetensors",
    "torchvision": "torchvision",
    "tqdm": "tqdm",
    "transformers": "transformers",
}

TORCH_IMPORTS = ("torch", "torchvision")


def _has_module(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def ensure_colab_dependencies() -> list[str]:
    """Install missing non-Torch packages and return the installed packages."""
    missing_torch = [module for module in TORCH_IMPORTS if not _has_module(module)]
    if missing_torch:
        raise RuntimeError(
            "Missing Colab Torch stack modules: "
            + ", ".join(missing_torch)
            + ". Select a GPU runtime or install torch/torchvision manually."
        )

    missing_packages: list[str] = []
    seen: set[str] = set()
    for module_name, package_name in REQUIRED_IMPORTS.items():
        if module_name in TORCH_IMPORTS:
            continue
        if _has_module(module_name):
            continue
        if package_name not in seen:
            seen.add(package_name)
            missing_packages.append(package_name)

    if not missing_packages:
        print("[OK] LOTUS dependencies already available; no pip install needed.")
        return []

    print("[INSTALL] Missing LOTUS dependencies: " + ", ".join(missing_packages))
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", *missing_packages],
        check=True,
    )
    print("[OK] Installed missing LOTUS dependencies.")
    return missing_packages


if __name__ == "__main__":
    ensure_colab_dependencies()
