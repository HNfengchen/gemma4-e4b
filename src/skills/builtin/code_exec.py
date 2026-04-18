from __future__ import annotations

import subprocess
import json

from src.skills.base import Skill


class CodeExecSkill(Skill):
    name = "code_exec"
    description = "Execute Python code and return the output. Use for computations, data processing, or running scripts."
    parameters = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Python code to execute.",
            },
            "timeout": {
                "type": "integer",
                "description": "Execution timeout in seconds (default: 30).",
            },
        },
        "required": ["code"],
    }

    def execute(self, code: str = "", timeout: int = 30, **kwargs) -> str:
        try:
            result = subprocess.run(
                ["python", "-c", code],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            output_parts = []
            if result.stdout:
                output_parts.append(result.stdout)
            if result.stderr:
                output_parts.append(f"[STDERR]\n{result.stderr}")
            if result.returncode != 0:
                output_parts.append(f"[EXIT CODE: {result.returncode}]")
            return "\n".join(output_parts) if output_parts else "(no output)"
        except subprocess.TimeoutExpired:
            return f"Error: Execution timed out after {timeout} seconds"
        except Exception as exc:
            return f"Error: {exc}"


class ShellExecSkill(Skill):
    name = "shell_exec"
    description = "Execute a shell command and return the output. Use with caution."
    parameters = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "Shell command to execute.",
            },
            "timeout": {
                "type": "integer",
                "description": "Execution timeout in seconds (default: 30).",
            },
        },
        "required": ["command"],
    }

    def execute(self, command: str = "", timeout: int = 30, **kwargs) -> str:
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            output_parts = []
            if result.stdout:
                output_parts.append(result.stdout)
            if result.stderr:
                output_parts.append(f"[STDERR]\n{result.stderr}")
            if result.returncode != 0:
                output_parts.append(f"[EXIT CODE: {result.returncode}]")
            return "\n".join(output_parts) if output_parts else "(no output)"
        except subprocess.TimeoutExpired:
            return f"Error: Execution timed out after {timeout} seconds"
        except Exception as exc:
            return f"Error: {exc}"


class WebFetchSkill(Skill):
    name = "web_fetch"
    description = "Fetch content from a URL and return it as text. Useful for reading web pages or API responses."
    parameters = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The URL to fetch.",
            },
            "max_length": {
                "type": "integer",
                "description": "Maximum length of returned content (default: 5000).",
            },
        },
        "required": ["url"],
    }

    def execute(self, url: str = "", max_length: int = 5000, **kwargs) -> str:
        try:
            import urllib.request

            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (compatible; Gemma4Bot/1.0)"},
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                content = resp.read().decode("utf-8", errors="replace")
            if len(content) > max_length:
                content = content[:max_length] + f"\n... (truncated, total {len(content)} chars)"
            return content
        except Exception as exc:
            return f"Error fetching URL: {exc}"
