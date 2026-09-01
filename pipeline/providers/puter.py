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
    _verified_opus_model: str | None = None

    @classmethod
    def preflight(cls, auth_token: str | None = None) -> str:
        if cls._verified_opus_model:
            return cls._verified_opus_model

        cli_script = repo_root() / "pipeline" / "providers" / "puter_cli.js"
        
        import os
        tokens = [auth_token] if auth_token else []
        for i in range(2, 9):
            t = os.environ.get(f"PUTER_AUTH_TOKEN_{i}")
            if t and t not in tokens:
                tokens.append(t)
        if not tokens:
            tokens = [None]
            
        last_error = None
        for current_token in tokens:
            my_env = None
            if current_token:
                my_env = os.environ.copy()
                my_env["PUTER_AUTH_TOKEN"] = current_token
                
            try:
                result = subprocess.run(
                    ["node", str(cli_script), "listModels"],
                    capture_output=True,
                    text=True,
                    env=my_env
                )
                import re
                combined_output = result.stdout + "\n" + result.stderr
                m = re.search(r"(\[.*\]|\{.*\})", combined_output, re.DOTALL)
                if not m:
                    if result.returncode != 0:
                        err_msg = result.stderr.strip() or result.stdout.strip()
                        raise RuntimeError(f"Puter CLI listModels failed (code {result.returncode}): {err_msg}")
                    raise RuntimeError(f"Puter CLI listModels returned empty/invalid output: {combined_output}")
                
                json_str = m.group(1)
                try:
                    models = json.loads(json_str)
                except json.JSONDecodeError:
                    if result.returncode != 0:
                        err_msg = result.stderr.strip() or result.stdout.strip()
                        raise RuntimeError(f"Puter CLI listModels failed (code {result.returncode}): {err_msg}")
                    raise RuntimeError(f"Puter CLI listModels returned invalid JSON: {json_str}")
                    
                if isinstance(models, dict) and "error" in models:
                    err_str = str(models.get("error", ""))
                    if "rate limit" in err_str.lower() or "usage" in err_str.lower() or "unauthorized" in err_str.lower():
                        last_error = f"Puter preflight failed: {err_str}"
                        continue
                    raise RuntimeError(f"Puter preflight failed: {models['error']}")
                    
                if not isinstance(models, list):
                    raise RuntimeError(f"Puter listModels returned unexpected format: {type(models)}")
                    
                available_ids = [m.get("id", "") for m in models]
                
                target = None
                for priority in ["claude-opus-4-8", "claude-opus-4-6"]:
                    for mid in available_ids:
                        if priority in mid:
                            target = mid
                            break
                    if target:
                        break
                
                if not target:
                    for mid in available_ids:
                        if "claude-opus" in mid:
                            target = mid
                            break
                
                if not target:
                    for mid in available_ids:
                        if "claude-sonnet-4-6" in mid:
                            target = mid
                            break
                            
                if not target:
                    raise RuntimeError(f"No preferred Claude model found in Puter catalog. Available: {available_ids}")
                    
                LOG.info("Puter preflight authentication: OK")
                LOG.info("Puter preflight model discovery: OK")
                LOG.info("Puter preflight preferred model: %s", target)
                LOG.info("Puter preflight provider: claude")
                LOG.info("Puter preflight status: READY")
                
                cls._verified_opus_model = target
                return target
                
            except subprocess.CalledProcessError as e:
                err_msg = e.stderr.strip() or e.stdout.strip()
                last_error = f"Puter CLI listModels failed: {err_msg}"
                continue
            except Exception as e:
                last_error = str(e)
                continue
                
        raise RuntimeError(last_error or "Puter preflight failed: All tokens exhausted or invalid.")

    @staticmethod
    def chat(
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
        json_mode: bool = False,
        auth_token: str | None = None,
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
        
        import os
        tokens = [auth_token] if auth_token else []
        for i in range(2, 9):
            t = os.environ.get(f"PUTER_AUTH_TOKEN_{i}")
            if t and t not in tokens:
                tokens.append(t)
        if not tokens:
            tokens = [None]
            
        last_out = None
        for current_token in tokens:
            my_env = None
            if current_token:
                my_env = os.environ.copy()
                my_env["PUTER_AUTH_TOKEN"] = current_token
                
            try:
                result = subprocess.run(
                    ["node", str(cli_script), "chat", json.dumps(payload)],
                    capture_output=True,
                    text=True,
                    env=my_env
                )
                
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
                
                if "error" in out or result.returncode != 0:
                    err_str = str(out.get("error", ""))
                    if "No usage left" in err_str or "rate limit" in err_str.lower():
                        LOG.warning("Puter account out of credits or rate limited. Falling back to next token if available...")
                        last_out = out
                        continue
                    return out
                    
                out["_account_index"] = tokens.index(current_token) + 1 if current_token in tokens else 1
                return out
                    
            except subprocess.CalledProcessError as e:
                err_msg = e.stderr.strip() or e.stdout.strip()
                last_out = {"error": f"Puter CLI failed: {err_msg}"}
                continue
            except Exception as e:
                last_out = {"error": str(e)}
                continue
                
        if last_out is not None:
            return last_out
        raise RuntimeError("Puter CLI failed: All tokens exhausted.")
