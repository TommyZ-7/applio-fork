"""Explicit, process-wide precision opt-ins for dedicated realtime workers."""

import os
import sys
from pathlib import Path

import torch


def _graph_disabled_marker() -> Path:
    """Where a failed graph capture records that eager mode must be used next.

    The client exits with code 86 after a capture failure so a supervisor can
    restart it. If the restart does not flip APPLIO_DISABLE_CUDA_GRAPHS, the
    service would retry the same capture forever, so the failure is also
    recorded on disk and consulted on the next start.
    """
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "applio" / "realtime-cuda-graphs-disabled"


def cuda_graphs_previously_failed() -> bool:
    try:
        return _graph_disabled_marker().exists()
    except OSError:
        return False


def record_cuda_graph_failure() -> None:
    try:
        marker = _graph_disabled_marker()
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(
            "A realtime CUDA graph capture failed in this environment.\n"
            "Eager inference is used from now on. Delete this file to retry.\n",
            encoding="utf-8",
        )
        print(f"[!] CUDA graphs disabled for future runs ({marker}).", file=sys.stderr)
    except OSError:
        pass


def enable_experimental_tf32(device, gpu_name):
    if (
        os.environ.get("APPLIO_EXPERIMENTAL_TF32") != "1"
        or not str(device).startswith("cuda")
        or "NVIDIA" not in (gpu_name or "").upper()
        or "ZLUDA" in (gpu_name or "").upper()
        or getattr(torch.version, "hip", None)
        or torch.cuda.get_device_capability(device)[0] < 8
    ):
        return False
    # Configure before model warmup/capture. This affects all CUDA matmul in
    # this process; it must not be toggled around asynchronous graph replay.
    torch.backends.cuda.matmul.allow_tf32 = True
    print("[Experimental] NVIDIA TF32 matmul enabled for this realtime process.")
    return True
