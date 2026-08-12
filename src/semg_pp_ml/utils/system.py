"""
System information utilities.
"""
from __future__ import annotations

import platform
import psutil
import subprocess
import math
from dataclasses import dataclass, asdict
from typing import List, Union


@dataclass
class GPUInfo:
    name: str
    vendor: str
    vram_mb: Union[int, str]


@dataclass
class SystemInfo:
    os: str
    cpu_name: str
    cpu_cores: int
    ram_gb: int
    gpus: List[GPUInfo]

    @property
    def gpu_available(self) -> bool:
        return len(self.gpus) > 0

    @property
    def gpu_name(self) -> str:
        return self.gpus[0].name if self.gpus else "CPU Mode"

    def to_dict(self):
        return asdict(self)


def get_gpu_info() -> List[GPUInfo]:
    """Detect GPUs using GPUtil (NVIDIA) and OS commands (AMD/Intel)."""
    try:
        import GPUtil
        gpus = GPUtil.getGPUs()
        return [
            GPUInfo(name=gpu.name, vendor="NVIDIA", vram_mb=int(gpu.memoryTotal))
            for gpu in gpus
        ]
    except Exception:
        pass

    # Fallback to OS detection
    gpus = []
    os_name = platform.system()

    if os_name == "Windows":
        try:
            cmd = [
                "powershell", "-Command",
                "Get-CimInstance Win32_VideoController | ForEach-Object { $_.Name + '|' + $_.AdapterRAM }"
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    parts = line.split('|')
                    name = parts[0].strip()
                    vram_bytes = parts[1].strip() if len(parts) > 1 and parts[1].strip() else "0"
                    try:
                        vram_mb = int(vram_bytes) // (1024 * 1024)
                    except ValueError:
                        vram_mb = "Unknown"

                    vendor = "AMD" if "amd" in name.lower() or "radeon" in name.lower() else \
                             "Intel" if "intel" in name.lower() else "Other"
                    gpus.append(GPUInfo(name=name, vendor=vendor, vram_mb=vram_mb))
        except Exception:
            pass

    elif os_name == "Linux":
        try:
            cmd = "lspci | grep -iE 'vga|3d|display'"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
            for line in result.stdout.strip().split('\n'):
                if line:
                    parts = line.split(":")
                    if len(parts) >= 3:
                        name = ':'.join(parts[2:]).strip()
                        vendor = "AMD" if "amd" in name.lower() or "radeon" in name.lower() else \
                                 "Intel" if "intel" in name.lower() else "Other"
                        gpus.append(GPUInfo(name=name, vendor=vendor, vram_mb="Unknown"))
        except Exception:
            pass

    return gpus


def get_system_info() -> SystemInfo:
    """Get complete system information."""
    # OS
    os_name = f"{platform.system()} {platform.release()}"

    # CPU
    cpu_name = platform.processor()
    cpu_cores = psutil.cpu_count(logical=True) or 0

    # RAM
    ram_bytes = psutil.virtual_memory().total
    ram_gb = math.ceil(ram_bytes / (1024 ** 3))

    # GPU
    gpus = get_gpu_info()

    return SystemInfo(
        os=os_name,
        cpu_name=cpu_name,
        cpu_cores=cpu_cores,
        ram_gb=ram_gb,
        gpus=gpus
    )


def print_system_info(sys_info: SystemInfo):
    """Pretty print system info."""
    print("=" * 50)
    print("         SYSTEM SPECIFICATIONS")
    print("=" * 50)
    print(f"OS   : {sys_info.os}")
    print(f"CPU  : {sys_info.cpu_name} ({sys_info.cpu_cores} Cores)")
    print(f"RAM  : {sys_info.ram_gb} GB")
    print("-" * 50)
    if sys_info.gpus:
        for i, gpu in enumerate(sys_info.gpus, 1):
            if isinstance(gpu.vram_mb, int) and gpu.vram_mb > 0:
                vram_gb = round(gpu.vram_mb / 1024, 1)
                vram_display = f"{gpu.vram_mb} MB (~{vram_gb} GB)"
            else:
                vram_display = str(gpu.vram_mb)
            print(f"GPU {i}: {gpu.name}")
            print(f"       Vendor : {gpu.vendor}")
            print(f"       VRAM   : {vram_display}")
    else:
        print("GPU  : No GPU detected")
    print("=" * 50)