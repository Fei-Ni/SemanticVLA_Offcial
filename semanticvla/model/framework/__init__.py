"""
Framework factory utilities.
Automatically builds registered framework implementations based on configuration.

Each framework module registers itself via FRAMEWORK_REGISTRY, e.g.:

    from semanticvla.model.tools import FRAMEWORK_REGISTRY

    @FRAMEWORK_REGISTRY.register("SemanticVLA")
    class SemanticVLA(...):
        ...
"""

import importlib
import pkgutil

from semanticvla.model.tools import FRAMEWORK_REGISTRY


try:
    pkg_path = __path__
except NameError:
    pkg_path = None

# Auto-import all framework submodules to trigger registration.
if pkg_path is not None:
    for _, module_name, _ in pkgutil.iter_modules(pkg_path):
        try:
            importlib.import_module(f"{__name__}.{module_name}")
        except Exception as e:
            print(f"Warning: Failed to auto-import framework submodule {module_name}: {e}")


def build_framework(cfg):
    """
    Build a framework model from config.
    Args:
        cfg: Config object (OmegaConf / namespace) containing cfg.framework.name.
    Returns:
        nn.Module: Instantiated framework model.
    """
    if not hasattr(cfg.framework, "name"):
        cfg.framework.name = cfg.framework.framework_py  # legacy fallback

    framework_id = cfg.framework.name
    if framework_id not in FRAMEWORK_REGISTRY._registry:
        raise NotImplementedError(f"Framework {framework_id} is not implemented.")

    return FRAMEWORK_REGISTRY[framework_id](cfg)


__all__ = ["build_framework", "FRAMEWORK_REGISTRY"]
