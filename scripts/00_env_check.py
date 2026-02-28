from __future__ import annotations

import os
import platform
import sys

def main() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("KMP_AFFINITY", "disabled")
    os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")
    os.environ.setdefault("KMP_USE_SHM", "0")
    os.environ.setdefault("KMP_SHM_DISABLE", "1")

    print(f"python_version={sys.version.split()[0]}")
    print(f"platform={platform.platform()}")
    try:
        from importlib import metadata

        print(f"numpy={metadata.version('numpy')}")
        print(f"pandas={metadata.version('pandas')}")
        print(f"sklearn={metadata.version('scikit-learn')}")
    except Exception:
        print("numpy=unknown")
        print("pandas=unknown")
        print("sklearn=unknown")
    print(f"cpu_count={os.cpu_count()}")
    try:
        import psutil  # type: ignore

        mem = psutil.virtual_memory().total / (1024**3)
        print(f"ram_gb={mem:.2f}")
    except Exception:
        print("ram_gb=unknown")
    print("gpu_available=unknown")


if __name__ == "__main__":
    main()
