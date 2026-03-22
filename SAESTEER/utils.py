"""Shared utility functions for SAESTEER pipeline."""

import gc
import os

import psutil
import torch

def print_gpu_utilization(device=None):
    """Print current GPU memory usage.

    Args:
        device: CUDA device index or string (e.g. 0, "cuda:1").
                If None, prints for the current default device.
    """
    if device is not None:
        if isinstance(device, str) and device.startswith("cuda:"):
            device = int(device.split(":")[1])
        allocated = torch.cuda.memory_allocated(device) / 1024**3
        reserved = torch.cuda.memory_reserved(device) / 1024**3
    else:
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"GPU memory allocated: {allocated:.2f} GB")
    print(f"GPU memory reserved: {reserved:.2f} GB")


def print_system_utilization():
    """Print current CPU and system memory usage."""
    process = psutil.Process(os.getpid())
    print(f"CPU memory used: {process.memory_info().rss / 1024**3:.2f} GB")
    print(f"System memory used: {psutil.virtual_memory().used / 1024**3:.2f} GB")
    print(f"System memory available: {psutil.virtual_memory().available / 1024**3:.2f} GB")


def clear_memory():
    """Force garbage collection and clear CUDA cache."""
    gc.collect()
    torch.cuda.empty_cache()


def setup_environment():
    """Load .env, login to HuggingFace, and configure CUDA allocator.

    Consolidates environment setup that was previously duplicated
    across step1, step2, and eval_only scripts.
    """
    # Clean env vars that can cause latin-1 encoding issues in huggingface_hub
    for var in ['HF_HUB_USER_AGENT', 'HTTP_PROXY', 'HTTPS_PROXY',
                'http_proxy', 'https_proxy', 'ALL_PROXY', 'all_proxy']:
        os.environ.pop(var, None)

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("DO_NOT_TRACK", "1")

    from dotenv import load_dotenv
    load_dotenv()

    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        from huggingface_hub import login
        login(token=hf_token)

    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key
