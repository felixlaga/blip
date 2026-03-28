import importlib.util
from importlib.machinery import EXTENSION_SUFFIXES
from pathlib import Path
import sys

__version__ = "0.1.0"
__author__ = 'Sharan Banagiri'


def _patch_scipy_integrate_trapz():
    """
    Restore ``scipy.integrate.trapz`` for older healpy releases.

    SciPy 1.15 removed the alias in favor of ``trapezoid``. healpy 1.15.2
    still imports ``trapz`` directly, so add a minimal alias before any
    healpy imports run.
    """

    try:
        import scipy.integrate as integrate
    except ImportError:
        return

    if not hasattr(integrate, "trapz") and hasattr(integrate, "trapezoid"):
        integrate.trapz = integrate.trapezoid


def _patch_healpy_extension_path():
    """
    Help older healpy wheels resolve sibling extension modules.

    Some healpy 1.15.x builds import compiled modules such as ``_pixelfunc``
    as top-level modules during package initialization. Expose the installed
    healpy package directory on ``sys.path`` so those lookups succeed.
    """

    try:
        spec = importlib.util.find_spec("healpy")
    except (ImportError, ValueError):
        return

    if spec is None or not spec.submodule_search_locations:
        return

    pkg_dir = Path(next(iter(spec.submodule_search_locations)))
    if not any((pkg_dir / ("_pixelfunc" + suffix)).exists() for suffix in EXTENSION_SUFFIXES):
        return

    pkg_dir_str = str(pkg_dir)
    if pkg_dir_str not in sys.path:
        sys.path.insert(0, pkg_dir_str)


_patch_healpy_extension_path()
_patch_scipy_integrate_trapz()
