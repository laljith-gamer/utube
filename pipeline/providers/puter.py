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
    ) -> str:
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
                text=True,
                check=True
            )
            out = json.loads(result.stdout)
            if "error" in out:
                raise RuntimeError(f"Puter API Error: {out['error']}")
            
            # Puter JS chat returns OpenAI-compatible format
            # e.g. { message: { content: "..." } } or { choices: [ { message: { content: "..." } } ] }
            if "message" in out and "content" in out["message"]:
                return out["message"]["content"]
            elif "choices" in out and len(out["choices"]) > 0:
                return out["choices"][0]["message"]["content"]
            else:
                return json.dumps(out)
                
        except subprocess.CalledProcessError as e:
            err_msg = e.stderr.strip() or e.stdout.strip()
            raise RuntimeError(f"Puter CLI failed: {err_msg}")
