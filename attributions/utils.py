import os


def get_available_cpus():
    """
    Determine the number of available CPUs for processing.
    Tries Kubernetes cgroup limits first, then falls back to os.cpu_count().
    """
    # Try Kubernetes cgroup CPU limit first
    try:
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "r") as f:
            quota = int(f.read().strip())
        with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us", "r") as f:
            period = int(f.read().strip())
        if quota > 0:
            return max(1, quota // period)
    except (FileNotFoundError, IOError, ValueError):
        # For newer cgroup v2 in Kubernetes
        try:
            with open("/sys/fs/cgroup/cpu.max", "r") as f:
                quota_info = f.read().strip().split()
                if quota_info[0] != "max":
                    quota = int(quota_info[0])
                    period = int(quota_info[1])
                    return max(1, quota // period)
        except (FileNotFoundError, IOError, ValueError, IndexError):
            pass

    # Fall back to os.cpu_count() with a safety margin
    cpu_count = os.cpu_count()
    if cpu_count is None:
        return 1  # Be conservative if we can't determine

    # Use most CPUs, but leave some headroom
    return max(1, cpu_count - min(2, cpu_count // 4))