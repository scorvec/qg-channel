"""qgchannel: a multi-level quasi-geostrophic channel with its own storms, for imposed-stratosphere experiments."""

from .diagnostics import annular_mode, folds, total_pv, zonal_stats
from .experiments import PROFILES, control, imposed, polar_anomaly, target, to_channel
from .model import DAY, Channel, Config, build, default_setup

__all__ = [
    "DAY",
    "PROFILES",
    "Channel",
    "Config",
    "annular_mode",
    "build",
    "control",
    "default_setup",
    "folds",
    "imposed",
    "polar_anomaly",
    "target",
    "to_channel",
    "total_pv",
    "zonal_stats",
]
__version__ = "0.1.0"
