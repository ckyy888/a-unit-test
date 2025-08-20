#!/usr/bin/env python3
"""Simple runner for WebArena CPU tests.

This script mirrors the high-level loop in `run.py` but is specialised for the
`unit_tests/CPU` interactive tests that rely on the Browser-in-Browser harness.

Usage examples:
    # Run one specific CPU test
    python general_test_runner.py --test_file unit_tests/CPU/test_gflight_date_picker.py \
        --url "https://www.google.com/maps"

    # Run every CPU test found in the directory
    python general_test_runner.py

The script will:
1.  Build a `PromptAgent` using the same `construct_agent` helper as `run.py`.
2.  Dynamically import the test module and locate the exported test function
    (async or sync) that accepts a `run_agent_func` callback.
3.  Provide a thin wrapper around the agent so the test can obtain actions from
    screenshots + accessibility-tree text.
4.  Print the JSON result returned by the test to stdout.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import datetime
import importlib.util
import inspect
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from PIL import Image

# Internal helper for element-ID → backend payload conversion
from action_conversion import convert_action  # type: ignore

# Local imports – reuse core agent construction logic
from agent.agent import PromptAgent, construct_agent  # type: ignore
from browser_env.helper_functions import get_action_description  # type: ignore

###############################################################################
# Agent wrapper
###############################################################################


def _b64_to_npjpg(b64: str) -> np.ndarray:
    """Decode base-64 image data to RGB ndarray; fall back to 1×1 black."""
    import io

    try:
        raw = base64.b64decode(b64)
        with Image.open(io.BytesIO(raw)) as img:
            return np.array(img.convert("RGB"))
    except Exception:
        return np.zeros((1, 1, 3), dtype=np.uint8)


class CPUAgentWrapper:
    """Turns a `PromptAgent` into the callback expected by CPU tests."""

    def __init__(self, args: argparse.Namespace):
        self.agent: PromptAgent = construct_agent(args)
        self.action_set_tag = args.action_set_tag
        
        # Create logs directory if it doesn't exist
        self.logs_dir = Path("unit_test_logs")
        self.logs_dir.mkdir(exist_ok=True)
        
        # Initialize log file path (will be set when we know the test name)
        self.log_file_path: Optional[Path] = None
        self.log_entries: List[str] = []

    # ---------------------------------------------------------------------
    def __call__(
        self, initial_observation: Dict[str, Any]
    ) -> List[Dict[str, Any]]:  # noqa: D401,E501 – pylint
        """Return a list of actions given the test's initial observation."""

        # 1. Build minimal trajectory for the prompt-constructor
        image_np: np.ndarray | None = None
        b64_img = initial_observation.get("initial_screenshot")
        if b64_img and b64_img not in {
            "placeholder_screenshot_data",
            "websocket_screenshot_failed",
        }:
            import io  # local import to avoid unnecessary dependency when unused

            image_np = _b64_to_npjpg(b64_img)

        text_obs = (
            initial_observation.get("accessibility_tree")
            or initial_observation.get("initial_accessibility_tree")
            or ""
        )

        # Extract observation metadata for element ID to coordinate conversion
        obs_metadata = initial_observation.get("obs_metadata", {})

        obs = {"text": text_obs, "image": image_np}
        trajectory = [
            {
                "observation": obs,
                "info": {
                    "page": initial_observation.get("url"),
                    "observation_metadata": obs_metadata,
                },
            }
        ]

        # 2. Ask agent for next action
        intent = initial_observation.get(
            "task_description", "Interact with page"
        )
        action = self.agent.next_action(
            trajectory, intent, meta_data={"action_history": ["None"]}
        )

        # 3. Store the accessibility tree and metadata that the agent used for this decision
        # This will be used later to convert element IDs to coordinates, even if DOM changes
        action["agent_decision_tree"] = text_obs
        action["agent_decision_metadata"] = obs_metadata

        # 4. Normalise action so that element-ID clicks have coordinates, etc.
        normalised_action = _normalise_action(action, obs_metadata)
        
        # 5. Log agent reasoning and human-readable action description
        self._log_agent_decision(action, obs_metadata, text_obs, intent)
        
        return [normalised_action]

    def set_test_name(self, test_name: str) -> None:
        """Set the test name for logging purposes."""
        self.log_file_path = self.logs_dir / f"{test_name}_log.txt"
        self.log_entries = []
        
        # Create the log file with header
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_file_path, 'w') as f:
            f.write(f"=== Agent Reasoning and Actions Log for {test_name} ===\n")
            f.write(f"Test started at: {timestamp}\n\n")

    def _log_agent_decision(
        self, 
        action: Dict[str, Any], 
        obs_metadata: Dict[str, Any], 
        accessibility_tree: str,
        intent: str
    ) -> None:
        """Log the agent's reasoning and human-readable action description."""
        if self.log_file_path is None:
            return
            
        # Extract reasoning from raw_prediction
        raw_prediction = action.get("raw_prediction", "")
        
        # Generate human-readable action description
        try:
            action_description = get_action_description(
                action, 
                {"text": obs_metadata}, 
                self.action_set_tag, 
                self.agent.prompt_constructor
            )
        except Exception as e:
            # Fallback to basic action description if get_action_description fails
            action_type = action.get("action_type", "unknown")
            if hasattr(action_type, 'name'):
                action_type_str = action_type.name.lower()
            else:
                action_type_str = str(action_type).split('.')[-1].lower() if '.' in str(action_type) else str(action_type)
            
            element_id = action.get("element_id")
            if element_id:
                # Try to get semantic description from accessibility tree
                semantic_desc = self._extract_element_description(element_id, accessibility_tree)
                action_description = f"{action_type_str} [{element_id}] {semantic_desc}"
            else:
                action_description = f"{action_type_str}"
                if action.get("text"):
                    action_description += f" [{action['text']}]"
                elif action.get("direction"):
                    action_description += f" [{action['direction']}]"
                elif action.get("url"):
                    action_description += f" [{action['url']}]"
                elif action.get("answer"):
                    action_description += f" [{action['answer']}]"

        # Create log entry with timestamp
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}]\n"
        log_entry += f"Task: {intent}\n"
        log_entry += f"Agent Reasoning:\n{raw_prediction}\n"
        log_entry += f"Action Taken: {action_description}\n"
        log_entry += f"{'='*60}\n\n"
        
        # Write to log file
        with open(self.log_file_path, 'a') as f:
            f.write(log_entry)
            
        self.log_entries.append(log_entry)

    def _extract_element_description(self, element_id: str, accessibility_tree: str) -> str:
        """Extract a human-readable description of an element from the accessibility tree."""
        lines = accessibility_tree.split('\n')
        for line in lines:
            if f'[{element_id}]' in line:
                # Extract the text content after the element ID
                parts = line.split(f'[{element_id}]', 1)
                if len(parts) > 1:
                    description = parts[1].strip()
                    # Clean up the description
                    if description.startswith(' '):
                        description = description[1:]
                    if description:
                        return f"({description})"
        return ""


# ---------------------------------------------------------------------------


def _normalise_action(
    action: Dict[str, Any], obs_metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Ensure the dict has the minimal fields CPU tests expect.

    Uses `action_conversion.convert_action` to compute coordinate payloads for
    element-ID-based clicks, scrolls, etc., while retaining the full original
    structure so that individual tests can pick whatever fields they need.

    Now supports obs_metadata for element ID to coordinate conversion.
    """

    try:
        route, payload = convert_action(action, obs_metadata)

        if route == "/click":
            action = action.copy()
            # Add/overwrite coordinates so tests can use them directly.
            action["coordinates"] = payload

        elif route == "/scroll":
            action = action.copy()
            action["coords"] = [payload["dx"], payload["dy"]]

        elif route == "/keyboard":
            action = action.copy()
            action["text"] = payload["key"]

        elif route in {"/goto", "/back", "/forward"}:
            # No additional mutation needed – tests rarely consume these yet.
            pass

    except ValueError:
        # Unsupported action for backend – leave unchanged so test can handle.
        pass

    return action


###############################################################################
# Test discovery & execution
###############################################################################


def _discover_test_function(mod) -> Callable[..., Any]:  # type: ignore
    """Return the first callable in module that expects a `run_agent_func` arg."""
    for name, obj in vars(mod).items():
        if callable(obj) and name.startswith("test"):
            sig = inspect.signature(obj)
            if any(
                p.name == "run_agent_func" for p in sig.parameters.values()
            ):
                return obj
    raise RuntimeError("No compatible test function found in module")


async def _run_single_test_async(
    test_func: Callable[..., Any],
    args: argparse.Namespace,
    run_agent_cb: Callable[[Dict[str, Any]], List[Dict[str, Any]]],
):
    """Execute coroutine test function and return its result."""
    kwargs = {
        k: getattr(args, k)
        for k in ("url", "backend_url", "config_path")
        if getattr(args, k, None)
    }
    return await test_func(kwargs.get("url"), run_agent_cb, **kwargs)  # type: ignore[arg-type]


def _run_single_test(
    test_func: Callable[..., Any],
    args: argparse.Namespace,
    run_agent_cb: Callable[[Dict[str, Any]], List[Dict[str, Any]]],
):
    """Execute sync test function and return its result."""
    kwargs = {
        k: getattr(args, k)
        for k in ("url", "backend_url", "config_path")
        if getattr(args, k, None)
    }
    return test_func(kwargs.get("url"), run_agent_cb, **kwargs)  # type: ignore[arg-type]


###############################################################################
# CLI
###############################################################################


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run WebArena CPU browser-in-browser tests with a PromptAgent"
    )
    p.add_argument(
        "--test_file", help="Path to the CPU test file", default=None
    )
    p.add_argument(
        "--config_path",
        help="Optional path to JSON config for test",
        default="",
    )

    # Minimal agent flags (extend as needed)
    p.add_argument("--provider", default="openai")
    p.add_argument("--model", default="gpt-4")
    p.add_argument("--mode", default="chat")
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--context_length", type=int, default=4096)
    p.add_argument("--max_tokens", type=int, default=256)
    p.add_argument("--action_set_tag", default="id_accessibility_tree")
    p.add_argument("--max_obs_length", type=int, default=1920)

    # Hidden/advanced flags expected by downstream helpers
    p.add_argument("--agent_type", default="prompt", help=argparse.SUPPRESS)
    p.add_argument(
        "--instruction_path",
        default="agent/prompts/jsons/p_direct_id_actree_2s.json",
        help=argparse.SUPPRESS,
    )
    p.add_argument("--stop_token", default=None, help=argparse.SUPPRESS)
    p.add_argument("--max_retry", type=int, default=3, help=argparse.SUPPRESS)
    p.add_argument("--model_endpoint", default="", help=argparse.SUPPRESS)

    return p


def main():
    args = build_parser().parse_args()

    # Guarantee any legacy attributes expected deeper in the stack
    for attr, default in {
        "agent_type": "prompt",
        "instruction_path": "agent/prompts/jsons/p_direct_id_actree_2s.json",
        "stop_token": None,
        "max_retry": 3,
        "model_endpoint": "",
    }.items():
        if not hasattr(args, attr):
            setattr(args, attr, default)

    # ------------------------------------------------------------------
    # Discover which files to run
    test_files: List[Path]
    if args.test_file:
        test_files = [Path(args.test_file)]
    else:
        test_files = sorted(Path("unit_tests/CPU").glob("test_*.py"))

    agent_wrapper = CPUAgentWrapper(args)

    results = {}
    for tf in test_files:
        mod_name = tf.stem
        spec = importlib.util.spec_from_file_location(mod_name, tf)
        module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        assert spec and spec.loader
        spec.loader.exec_module(module)  # type: ignore[misc]

        test_func = _discover_test_function(module)
        print(f"Running {test_func.__name__} from {tf}…", flush=True)

        # Set the test name for logging
        agent_wrapper.set_test_name(mod_name)

        if inspect.iscoroutinefunction(test_func):
            outcome = asyncio.run(
                _run_single_test_async(test_func, args, agent_wrapper)
            )
        else:
            outcome = _run_single_test(test_func, args, agent_wrapper)

        results[str(tf)] = outcome

        def _strip_ndarray_and_screenshots(item):  # type: ignore[ann-return]
            """Recursively remove NumPy arrays and screenshot data (base64 strings)."""
            if isinstance(item, dict):
                result = {}
                for k, v in item.items():
                    # Skip screenshot fields that contain base64 data
                    if any(
                        screenshot_key in k.lower()
                        for screenshot_key in [
                            "screenshot",
                            "initial_screenshot",
                            "screenshot_after",
                        ]
                    ):
                        if (
                            isinstance(v, str) and len(v) > 100
                        ):  # Likely base64 data
                            result[k] = f"<screenshot_data_length_{len(v)}>"
                        else:
                            result[k] = v
                    elif not isinstance(v, np.ndarray):
                        result[k] = _strip_ndarray_and_screenshots(v)
                return result
            if isinstance(item, list):
                return [
                    _strip_ndarray_and_screenshots(v)
                    for v in item
                    if not isinstance(v, np.ndarray)
                ]
            return item

        cleaned_outcome = _strip_ndarray_and_screenshots(outcome)

        print(json.dumps(cleaned_outcome, indent=2))

    # Aggregate success information
    summary = {
        "total": len(results),
        "passed": sum(1 for r in results.values() if r.get("success")),
    }
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))

    # Exit code
    sys.exit(0 if summary["passed"] == summary["total"] else 1)


if __name__ == "__main__":
    main()
