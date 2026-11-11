#!/usr/bin/env python3
"""Check that this machine can run Basir: Python, GPU, key libraries and observability.

python scripts/check_env.py              # versions + CUDA check (exit 1 without CUDA)
python scripts/check_env.py --langfuse   # also send one smoke trace to Langfuse
"""

from __future__ import annotations

import argparse
from pathlib import Path
import platform
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utilities.config import get_settings
from utilities.environment import package_versions


def report_versions() -> None:
    print(f"python        {platform.python_version()} ({sys.executable})")
    for name, installed in package_versions().items():
        print(f"{name:<13} {installed or 'MISSING'}")


def report_gpu() -> bool:
    import torch

    if not torch.cuda.is_available():
        print("cuda          NOT AVAILABLE")
        return False
    props = torch.cuda.get_device_properties(0)
    print(f"cuda          {torch.version.cuda}, {props.name}, {props.total_memory / 2**30:.1f} GB")
    x = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
    torch.cuda.synchronize()
    print(f"cuda matmul   ok ({(x @ x).float().abs().mean().item():.3f})")
    return True


def report_settings() -> None:
    s = get_settings()
    print(f"device        {s.effective_device}")
    print(f"datasets_dir  {s.datasets_dir}")
    print(f"mlflow        {s.mlflow_uri}")
    print(f"langfuse      {'enabled' if s.langfuse_enabled else 'disabled (no keys)'}")


def langfuse_smoke() -> bool:
    from utilities.tracing import flush, get_tracer, observe

    tracer = get_tracer()
    if tracer is None:
        print("langfuse      skipped: set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY in .env")
        return False
    if not tracer.auth_check():
        print("langfuse      AUTH FAILED: check the keys and LANGFUSE_BASE_URL")
        return False

    @observe(name="basir.smoke", capture=True)
    def smoke(message: str) -> str:
        return message.upper()

    smoke("hello from check_env")
    flush()
    print("langfuse      smoke trace 'basir.smoke' sent")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--langfuse", action="store_true", help="Send one smoke trace to Langfuse.")
    args = parser.parse_args(argv)

    report_versions()
    cuda_ok = report_gpu()
    report_settings()
    langfuse_ok = langfuse_smoke() if args.langfuse else True
    return 0 if cuda_ok and langfuse_ok else 1


if __name__ == "__main__":
    sys.exit(main())
