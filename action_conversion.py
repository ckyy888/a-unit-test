"""action_conversion.py
========================
Utility helpers that translate a *native* WebArena action produced by any
agent (`browser_env.actions.Action`) into the minimal REST payloads understood
by the *browser-in-browser* backend found under
`unit_tests/browser-in-browser/backend/`.

The backend supports only a handful of endpoints:
    POST /click     {"x": <0-1>, "y": <0-1>}           – coordinate click
    POST /keyboard  {"key": <string>}                   – press a single key
    POST /scroll    {"dx": <float>, "dy": <float>}     – scroll deltas (fraction of viewport)
    POST /goto      {"url": <str>}
    POST /back      {}                                  – history back
    POST /forward   {}                                  – history forward

This file exposes a single public helper
    convert_action(action, obs_metadata=None) -> (route, payload)

`obs_metadata` is the per-observation metadata returned by `browser_env` – it
contains bounding-box information for each accessibility-tree node.  If it is
provided we can convert element-id based CLICK/HOVER/TYPE actions into
coordinates; otherwise we fall back to a centre click (0.5, 0.5).

The module is intentionally dependency-free (only NumPy for numeric work) so it
can be imported from CPU tests or other runners without bringing in Playwright
or the full browser stack.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

# Import shared constants from the core codebase
from browser_env.actions import ActionTypes, _id2key

__all__ = ["convert_action", "element_id_to_center"]

# ---------------------------------------------------------------------------
# Helper – element-id → normalised centre coordinate
# ---------------------------------------------------------------------------


def element_id_to_center(
    element_id: str, obs_metadata: Optional[dict[str, Any]]
) -> Dict[str, float]:
    """Return normalised (x, y) in [0,1]² for element id.

    The observation metadata carries bounding boxes using absolute pixel
    coordinates (left, top, width, height) plus the browser window top-left
    offset.  The calculation mirrors logic in `TextObservationProcessor`.

    Now supports both the original WebArena format and the browser-in-browser format.
    """
    if not obs_metadata:
        return {"x": 0.5, "y": 0.5}

    # Handle browser-in-browser format (obs_nodes_info directly in metadata)
    if "obs_nodes_info" in obs_metadata:
        nodes = obs_metadata["obs_nodes_info"]
        browser_cfg = obs_metadata.get("browser_config", {})
    # Handle original WebArena format (nested under "text")
    elif "text" in obs_metadata:
        nodes = obs_metadata["text"].get("obs_nodes_info", {})
        browser_cfg = obs_metadata["text"].get("browser_config", {})
    else:
        return {"x": 0.5, "y": 0.5}

    node_info = nodes.get(str(element_id))
    if not node_info:
        return {"x": 0.5, "y": 0.5}

    # Extract bounding box - require bound to be available
    bound = node_info.get("bound")
    if not bound or len(bound) < 4:
        raise ValueError(
            f"No valid bounding box found for element_id '{element_id}'. Expected 'bound' with 4 values, got: {bound}"
        )

    left, top, width, height = bound

    # window offset if present – fall back to (0,0)
    win_left = browser_cfg.get("win_left_bound", 0)
    win_top = browser_cfg.get("win_upper_bound", 0)

    centre_x = (left - win_left) + width / 2
    centre_y = (top - win_top) + height / 2

    # viewport size – default to 1280×800 if unknown
    viewport = browser_cfg.get("viewport_size", {"width": 1280, "height": 800})
    vw, vh = viewport.get("width", 1280), viewport.get("height", 800)

    return {"x": centre_x / vw, "y": centre_y / vh}


# ---------------------------------------------------------------------------
# Main public helper
# ---------------------------------------------------------------------------


def convert_action(
    action: Dict[str, Any],
    obs_metadata: Optional[dict[str, Any]] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Translate *action* into a `(route, payload)` pair for backend HTTP call.

    Unsupported / unknown action types raise `ValueError` so that the caller
    can decide what to do (e.g. skip or stop).
    """

    atype = action.get("action_type")

    # ------------------------------------------------------------------ CLICK
    if atype in {ActionTypes.CLICK, ActionTypes.MOUSE_CLICK}:
        if "coordinates" in action:
            coords = action["coordinates"]
        else:
            coords = element_id_to_center(
                action.get("element_id", ""), obs_metadata
            )
        return ("/click", {"x": coords["x"], "y": coords["y"]})

    # -------------------------------------------------------------- KEYBOARD
    if atype in {ActionTypes.TYPE, ActionTypes.KEYBOARD_TYPE}:
        # agent stores `text` as list[int] key-ids or raw string
        text = action.get("text", "")
        if isinstance(text, list):
            keys_str = "".join(_id2key[i] for i in text)
        else:
            keys_str = str(text)
        # Send one key at a time – CPU tests usually expect single keys/Enter
        return ("/keyboard", {"key": keys_str})

    # ---------------------------------------------------------------- SCROLL
    if atype == ActionTypes.SCROLL:
        # The WebArena scroll action may store direction (up/down) or dx/dy coords.
        if "direction" in action:
            dy = -1.0 if "up" in action["direction"] else 1.0
            return ("/scroll", {"dx": 0.0, "dy": dy})
        if "coords" in action:
            dx, dy = action["coords"]
            return ("/scroll", {"dx": float(dx), "dy": float(dy)})
        # Fallback small scroll down
        return ("/scroll", {"dx": 0.0, "dy": 0.3})

    # -------------------------------------------------------------- NAVIGATE
    if atype == ActionTypes.GOTO_URL:
        return ("/goto", {"url": action.get("url", "")})
    if atype == ActionTypes.GO_BACK:
        return ("/back", {})
    if atype == ActionTypes.GO_FORWARD:
        return ("/forward", {})

    # ------------------------------------------------------------- UNSUPPORTED
    raise ValueError(
        f"Action type {atype} not supported by browser-in-browser backend"
    )
