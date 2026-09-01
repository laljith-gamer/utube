"""Puter provider logic (optional standalone integration if needed).
This file contains the subprocess call to puter_cli.js.
"""
from __future__ import annotations

import json
import logging
import subprocess
from typing import Any

from ..utils import repo_root

LOG = logging.getLogger("utube.puter")


class PuterProvider:
    @staticmethod
    def chat(
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
        json_mode: bool = False,
    ) -> dict[str, Any]:
        cli_script = repo_root() / "pipeline" / "providers" / "puter_cli.js"
        
        # Prepare payload
        # Note: puter.js might not support strict JSON mode natively depending on the model,
        # but we pass the prompt anyway.
        payload = {
            "model": model,
            "messages": messages,
            "stream": False
        }
        
        try:
            result = subprocess.run(
                ["node", str(cli_script), "chat", json.dumps(payload)],
                capture_output=True,
                text=True
            )
            
            # Use a regex to extract the first JSON block from stdout or stderr
            # to avoid node.js assertions and other junk printed
            import re
            combined_output = result.stdout + "\n" + result.stderr
            m = re.search(r"(\{.*\})", combined_output, re.DOTALL)
            if not m:
                if result.returncode != 0:
                    err_msg = result.stderr.strip() or result.stdout.strip()
                    raise RuntimeError(f"Puter CLI failed (code {result.returncode}): {err_msg}")
                raise RuntimeError(f"Puter CLI returned empty/invalid output: {combined_output}")
                
            json_str = m.group(1)
            try:
                out = json.loads(json_str)
            except json.JSONDecodeError:
                if result.returncode != 0:
                    err_msg = result.stderr.strip() or result.stdout.strip()
                    raise RuntimeError(f"Puter CLI failed (code {result.returncode}): {err_msg}")
                raise RuntimeError(f"Puter CLI returned invalid JSON: {json_str}")
            
            # If the script returned a JSON error block or exited with error
            if "error" in out or result.returncode != 0:
                return out
                
            return out
                
        except subprocess.CalledProcessError as e:
            err_msg = e.stderr.strip() or e.stdout.strip()
            raise RuntimeError(f"Puter CLI failed: {err_msg}")
