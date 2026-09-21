"""Environment and hardware verification script."""

import sys
import platform
from pathlib import Path
from loguru import logger

from src.utils.config import load_config, resolve_device


def run_env_check(config_path: str = "config/config.yaml") -> dict:
    """Check Python, CUDA, PyTorch, and config loading."""
    logger.info("Running System & Environment Verification...")

    # Python info
    py_ver = platform.python_version()
    logger.info(f"Python Version: {py_ver} ({sys.executable})")

    # Load configuration
    try:
        cfg = load_config(config_path)
        logger.success(f"Configuration loaded successfully from {config_path}")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        raise

    # PyTorch and device detection
    device_info = {}
    try:
        import torch

        device_str = resolve_device(cfg.system.device)
        device_info["torch_version"] = torch.__version__
        device_info["resolved_device"] = device_str
        device_info["cuda_available"] = torch.cuda.is_available()

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            device_info["gpu_name"] = gpu_name
            device_info["vram_gb"] = round(vram_gb, 2)
            logger.success(
                f"PyTorch CUDA active: {gpu_name} ({vram_gb:.2f} GB VRAM) -> Using device '{device_str}'"
            )
        else:
            logger.info(f"PyTorch running on CPU -> Using device '{device_str}'")

    except ImportError:
        logger.warning("PyTorch not installed in this environment yet.")
        device_info["status"] = "torch_not_installed"

    # Directory verification
    required_dirs = [
        Path(cfg.data.cache_dir),
        Path("reports"),
        Path("models_store"),
    ]
    for d in required_dirs:
        d.mkdir(parents=True, exist_ok=True)
        logger.info(f"Verified directory: {d}")

    logger.success("Environment check completed successfully.")
    return {
        "python_version": py_ver,
        "device_info": device_info,
        "config": cfg.model_dump(),
    }


if __name__ == "__main__":
    run_env_check()
