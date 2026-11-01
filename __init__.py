"""
@title: Impact Pack - FaceDetailer Cycle Denoise
@description: A FaceDetailer variant that lets you set a separate denoise value for each cycle pass.
"""

import logging
import os
import sys

_NODE_DIR = os.path.dirname(os.path.abspath(__file__))
_CUSTOM_NODES_DIR = os.path.dirname(_NODE_DIR)

# The upstream repo is named "ComfyUI-Impact-Pack", but users rename it freely and
# case-sensitivity differs per filesystem, so match loosely.
_IMPACT_PACK_NAMES = {"comfyuiimpactpack", "impactpack"}


def _find_impact_pack():
    try:
        siblings = os.listdir(_CUSTOM_NODES_DIR)
    except OSError:
        return None

    for name in siblings:
        normalized = name.lower().replace("-", "").replace("_", "").replace(" ", "")
        if normalized not in _IMPACT_PACK_NAMES:
            continue
        candidate = os.path.join(_CUSTOM_NODES_DIR, name)
        if os.path.isdir(os.path.join(candidate, "modules", "impact")):
            return candidate
    return None


def _ensure_importable():
    """Import check first: if ComfyUI already loaded Impact Pack, sys.path is set up."""
    try:
        import modules.impact.hooks  # noqa: F401
        return True
    except ImportError:
        pass

    impact_pack_dir = _find_impact_pack()
    if impact_pack_dir is None:
        return False

    for path in (impact_pack_dir, os.path.join(impact_pack_dir, "modules")):
        if path not in sys.path:
            sys.path.insert(0, path)

    try:
        import modules.impact.hooks  # noqa: F401
        return True
    except ImportError:
        return False


if _ensure_importable():
    from .facedetailer_cycle_denoise import FaceDetailerCycleDenoise

    NODE_CLASS_MAPPINGS = {
        "FaceDetailerCycleDenoise": FaceDetailerCycleDenoise,
    }

    NODE_DISPLAY_NAME_MAPPINGS = {
        "FaceDetailerCycleDenoise": "FaceDetailer (cycle denoise)",
    }
else:
    logging.error(
        "[FaceDetailerCycleDenoise] Cannot import Impact Pack. "
        "Install ComfyUI-Impact-Pack into ComfyUI/custom_nodes and restart ComfyUI. "
        "This node is a companion to Impact Pack and cannot load without it."
    )
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
