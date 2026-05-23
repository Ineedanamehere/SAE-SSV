"""Shared utility functions for SAESTEER pipeline."""

from contextlib import nullcontext
import gc
import os

import torch

try:
    import psutil
except ImportError:
    psutil = None

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
    if psutil is None:
        print("psutil is not installed; CPU/system memory utilization is unavailable.")
        return
    process = psutil.Process(os.getpid())
    print(f"CPU memory used: {process.memory_info().rss / 1024**3:.2f} GB")
    print(f"System memory used: {psutil.virtual_memory().used / 1024**3:.2f} GB")
    print(f"System memory available: {psutil.virtual_memory().available / 1024**3:.2f} GB")


def clear_memory():
    """Force garbage collection and clear CUDA cache."""
    gc.collect()
    torch.cuda.empty_cache()


def resolve_precision_dtype(precision=None, *, device="cuda"):
    """Resolve the requested mixed-precision dtype for CUDA autocast/model loading."""
    requested = (precision or os.environ.get("SAE_SSV_PRECISION", "fp16")).strip().lower()
    if requested in {"fp32", "float32", "none"}:
        return torch.float32
    if requested in {"fp16", "float16"}:
        return torch.float16
    if requested in {"bf16", "bfloat16"}:
        if torch.cuda.is_available() and str(device).startswith("cuda"):
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
            print("SAE_SSV_PRECISION=bf16 requested, but CUDA bf16 is not supported; falling back to fp16.")
            return torch.float16
        return torch.bfloat16
    raise ValueError(
        "Unsupported SAE_SSV_PRECISION="
        f"{requested!r}. Use one of: fp16, bf16, fp32."
    )


def autocast_context(device="cuda", dtype=None):
    """Return a CUDA autocast context when mixed precision is enabled."""
    dtype = dtype or resolve_precision_dtype(device=device)
    if dtype == torch.float32 or not torch.cuda.is_available() or not str(device).startswith("cuda"):
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype)


def compile_enabled():
    """Whether small-module torch.compile optimizations are enabled."""
    raw = os.environ.get("SAE_SSV_COMPILE", "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


class _CompiledWithFallback:
    """Call a compiled function/module, falling back to eager if compilation fails."""

    def __init__(self, original, compiled, name):
        self.original = original
        self.compiled = compiled
        self.name = name
        self.disabled = False

    def __call__(self, *args, **kwargs):
        if self.disabled:
            return self.original(*args, **kwargs)
        try:
            return self.compiled(*args, **kwargs)
        except Exception as exc:
            self.disabled = True
            print(f"torch.compile failed at runtime for {self.name}; falling back to eager mode. Reason: {exc}")
            return self.original(*args, **kwargs)


def maybe_compile(module_or_fn, *, name="module"):
    """Compile a small stable module/callable when enabled, with a safe fallback."""
    if not compile_enabled():
        return module_or_fn
    if not hasattr(torch, "compile"):
        return module_or_fn
    try:
        compiled = torch.compile(module_or_fn)
        print(f"Enabled torch.compile for {name}.")
        return _CompiledWithFallback(module_or_fn, compiled, name)
    except Exception as exc:
        print(f"Could not torch.compile {name}; using eager mode. Reason: {exc}")
        return module_or_fn


def configure_torch_performance(device="cuda"):
    """Configure conservative CUDA performance defaults."""
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    dtype = resolve_precision_dtype(device=device)
    compile_state = "on" if compile_enabled() else "off"
    print(f"Runtime performance config: precision={dtype}, tf32=on, small_module_compile={compile_state}")
    return dtype


def dataloader_kwargs():
    """Common DataLoader performance knobs for tensor batches."""
    num_workers = int(os.environ.get("SAE_SSV_DATALOADER_WORKERS", "0"))
    kwargs = {
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = int(os.environ.get("SAE_SSV_DATALOADER_PREFETCH", "2"))
    return kwargs


def setup_environment():
    """Load .env, force online Hugging Face access, and configure runtime env.

    Consolidates environment setup that was previously duplicated
    across step1, step2, and eval_only scripts.
    """
    # Clean env vars that can cause latin-1 encoding issues in huggingface_hub
    for var in ['HF_HUB_USER_AGENT', 'HTTP_PROXY', 'HTTPS_PROXY',
                'http_proxy', 'https_proxy', 'ALL_PROXY', 'all_proxy']:
        os.environ.pop(var, None)

    os.environ.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF",
        "expandable_segments:True,max_split_size_mb:512",
    )
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("DO_NOT_TRACK", "1")

    # This repository now assumes internet access is available and should
    # fetch models, SAEs, and datasets from Hugging Face as needed.
    for var in ["HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"]:
        os.environ.pop(var, None)

    from dotenv import load_dotenv
    load_dotenv()

    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        # Keep authentication process-local and avoid mutating the shared token store.
        os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", hf_token)

    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key
