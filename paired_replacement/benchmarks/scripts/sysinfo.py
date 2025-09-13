#!/usr/bin/env python3
import json
import os
import platform
import socket
import subprocess
from pathlib import Path


def sh(cmd):
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
        return out
    except Exception:
        return ""


def linux_info():
    info = {}
    lscpu = sh(["bash", "-lc", "LC_ALL=C lscpu"])
    if lscpu:
        for line in lscpu.splitlines():
            if not line or ":" not in line:
                continue
            k, v = [x.strip() for x in line.split(":", 1)]
            info[k] = v
    meminfo = sh(["bash", "-lc", "cat /proc/meminfo || true"])
    if meminfo:
        for line in meminfo.splitlines():
            if ":" in line:
                k, v = [x.strip() for x in line.split(":", 1)]
                info[f"mem_{k}"] = v
    return info


def mac_info():
    info = {}
    # sysctl CPU
    sysctl_cpu = sh(["bash", "-lc", "sysctl -a | egrep 'machdep.cpu.|hw.(logicalcpu|physicalcpu|memsize|cpufrequency)' || true"])
    if sysctl_cpu:
        for line in sysctl_cpu.splitlines():
            if ":" in line:
                k, v = [x.strip() for x in line.split(":", 1)]
                info[k] = v
    # system_profiler (can be slow; limit to essential)
    sp = sh(["bash", "-lc", "system_profiler SPHardwareDataType | sed -n '1,200p' || true"])
    if sp:
        info["system_profiler"] = sp
    return info


def win_info():
    info = {}
    # wmic is deprecated; rely on platform only
    return info


def main():
    data = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
    }
    try:
        import torch  # type: ignore
        data["torch"] = torch.__version__
    except Exception:
        data["torch"] = None

    system = platform.system().lower()
    if system == "linux":
        data["details"] = linux_info()
    elif system == "darwin":
        data["details"] = mac_info()
    elif system == "windows":
        data["details"] = win_info()
    else:
        data["details"] = {}

    # CPU counts
    try:
        import psutil  # type: ignore
        data["cpu_logical"] = psutil.cpu_count(logical=True)
        data["cpu_physical"] = psutil.cpu_count(logical=False)
        vm = psutil.virtual_memory()
        data["mem_total_bytes"] = getattr(vm, "total", None)
        try:
            freq = psutil.cpu_freq()
            if freq:
                data["cpu_freq_mhz"] = freq.max or freq.current
        except Exception:
            pass
    except Exception:
        data["cpu_logical"] = os.cpu_count()

    out_path = Path("benchmarks/sysinfo.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(out_path)


if __name__ == "__main__":
    main()

