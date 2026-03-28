import numpy as np
import sys
import importlib

try:
    import scipy.integrate as _scipy_integrate

    # Older healpy releases import scipy.integrate.trapz directly.
    # SciPy removed that alias; provide the NumPy equivalent for compatibility.
    if not hasattr(_scipy_integrate, "trapz"):
        _scipy_integrate.trapz = np.trapz
except Exception:
    pass

try:
    # Some healpy wheels expect these compiled helpers to also exist as
    # top-level modules, not only under the healpy package namespace.
    for _name in [
        "healpy._pixelfunc",
        "healpy._sphtools",
        "healpy._query_disc",
        "healpy._healpy_pixel_lib",
    ]:
        _module = importlib.import_module(_name)
        sys.modules.setdefault(_name.split(".")[-1], _module)
except Exception:
    pass
