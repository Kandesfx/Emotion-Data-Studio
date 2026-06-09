"""Runtime resource planning for CPU/GPU/RAM utilization.

The goal is to use local hardware aggressively but safely for long video jobs:
- honor user runtime mode (auto/cpu/cuda)
- choose CPU worker/thread counts from available cores
- expose GPU availability and memory when torch is installed
- apply torch CPU thread settings once at app startup/pipeline startup
"""

from __future__ import annotations

import os
import platform
from dataclasses import asdict, dataclass


@dataclass
class ResourcePlan:
    runtime_mode: str
    device: str
    cpu_count: int
    cpu_threads: int
    ffmpeg_threads: int
    pipeline_workers: int
    has_cuda: bool
    gpu_name: str | None
    gpu_memory_gb: float | None
    ram_gb: float | None
    torch_dtype: str

    def to_dict(self) -> dict:
        return asdict(self)


class ResourceManager:
    def __init__(self):
        self._cached_plan: ResourcePlan | None = None

    def plan(self, force_refresh: bool = False) -> ResourcePlan:
        if self._cached_plan is not None and not force_refresh:
            return self._cached_plan

        runtime_mode = os.getenv("RUNTIME_MODE", "auto").lower().strip()
        cpu_count = os.cpu_count() or 4
        cpu_threads = self._int_env("EDS_CPU_THREADS", max(1, min(cpu_count, max(2, cpu_count - 1))))
        ffmpeg_threads = self._int_env("EDS_FFMPEG_THREADS", max(1, min(cpu_count, max(2, cpu_count // 2))))
        pipeline_workers = self._int_env("EDS_PIPELINE_WORKERS", 1)
        ram_gb = self._ram_gb()

        has_cuda, gpu_name, gpu_memory_gb = self._cuda_info()
        if runtime_mode == "cuda" and not has_cuda:
            device = "cpu"
        elif runtime_mode == "cpu":
            device = "cpu"
        elif has_cuda:
            device = "cuda"
        else:
            device = "cpu"

        torch_dtype = "float16" if device == "cuda" else "float32"
        self._cached_plan = ResourcePlan(
            runtime_mode=runtime_mode,
            device=device,
            cpu_count=cpu_count,
            cpu_threads=cpu_threads,
            ffmpeg_threads=ffmpeg_threads,
            pipeline_workers=max(1, pipeline_workers),
            has_cuda=has_cuda,
            gpu_name=gpu_name,
            gpu_memory_gb=gpu_memory_gb,
            ram_gb=ram_gb,
            torch_dtype=torch_dtype,
        )
        return self._cached_plan

    def apply(self) -> ResourcePlan:
        plan = self.plan(force_refresh=True)
        os.environ.setdefault("OMP_NUM_THREADS", str(plan.cpu_threads))
        os.environ.setdefault("MKL_NUM_THREADS", str(plan.cpu_threads))
        os.environ.setdefault("NUMEXPR_NUM_THREADS", str(plan.cpu_threads))
        try:
            import torch
            torch.set_num_threads(max(1, plan.cpu_threads))
            if plan.cpu_count > 2:
                torch.set_num_interop_threads(max(1, min(4, plan.cpu_count // 2)))
            if plan.device == "cuda":
                torch.backends.cudnn.benchmark = True
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass
        return plan

    @staticmethod
    def _int_env(key: str, default: int) -> int:
        try:
            value = int(os.getenv(key, str(default)))
            return max(1, default if value <= 0 else value)
        except Exception:
            return max(1, default)

    @staticmethod
    def _cuda_info() -> tuple[bool, str | None, float | None]:
        try:
            import torch
            if not torch.cuda.is_available():
                return False, None, None
            props = torch.cuda.get_device_properties(0)
            return True, props.name, round(props.total_memory / (1024 ** 3), 2)
        except Exception:
            return False, None, None

    @staticmethod
    def _ram_gb() -> float | None:
        try:
            if platform.system().lower() == "windows":
                import ctypes
                class MEMORYSTATUSEX(ctypes.Structure):
                    _fields_ = [
                        ("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                    ]
                status = MEMORYSTATUSEX()
                status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
                ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
                return round(status.ullTotalPhys / (1024 ** 3), 2)
        except Exception:
            return None
        return None


resource_manager = ResourceManager()
