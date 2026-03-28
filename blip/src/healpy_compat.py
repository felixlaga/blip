import importlib
import importlib.util
import sys

import numpy as np


_INITIALIZED = False


def ensure_healpy_compat():
    global _INITIALIZED
    if _INITIALIZED:
        return

    try:
        import scipy.integrate as scipy_integrate

        if not hasattr(scipy_integrate, "trapz"):
            scipy_integrate.trapz = np.trapz
    except Exception:
        pass

    try:
        spec = importlib.util.find_spec("healpy")
        if spec is not None and spec.submodule_search_locations:
            package_dir = next(iter(spec.submodule_search_locations))
            if package_dir not in sys.path:
                sys.path.append(package_dir)
    except Exception:
        pass

    for name in [
        "healpy._pixelfunc",
        "healpy._sphtools",
        "healpy._query_disc",
        "healpy._healpy_pixel_lib",
    ]:
        try:
            module = importlib.import_module(name)
            sys.modules.setdefault(name.split(".")[-1], module)
        except Exception:
            continue

    _INITIALIZED = True
