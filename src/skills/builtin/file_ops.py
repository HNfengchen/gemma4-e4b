from __future__ import annotations

import datetime
import json
from pathlib import Path

from src.skills.base import Skill


class FileOpsSkill(Skill):
    name = "file_ops"
    description = "Read, write, list, or check existence of files on the local filesystem."
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["read", "write", "list", "exists"],
                "description": "The file operation to perform.",
            },
            "path": {
                "type": "string",
                "description": "The file or directory path.",
            },
            "content": {
                "type": "string",
                "description": "Content to write (only for 'write' action).",
            },
        },
        "required": ["action", "path"],
    }

    def execute(self, action: str = "", path: str = "", content: str = "", **kwargs) -> str:
        try:
            p = Path(path)
            if action == "read":
                if not p.exists():
                    return f"Error: File not found: {path}"
                text = p.read_text(encoding="utf-8")
                max_len = 10000
                if len(text) > max_len:
                    text = text[:max_len] + f"\n... (truncated, total {len(text)} chars)"
                return text
            elif action == "write":
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(content, encoding="utf-8")
                return f"Successfully wrote {len(content)} chars to {path}"
            elif action == "list":
                if not p.is_dir():
                    return f"Error: Not a directory: {path}"
                items = []
                for item in sorted(p.iterdir()):
                    size = ""
                    if item.is_file():
                        try:
                            size = f" ({item.stat().st_size} bytes)"
                        except OSError:
                            pass
                    items.append(f"{'[DIR]' if item.is_dir() else '[FILE]'} {item.name}{size}")
                return "\n".join(items) if items else "(empty directory)"
            elif action == "exists":
                return json.dumps({"exists": p.exists(), "is_file": p.is_file(), "is_dir": p.is_dir()})
            else:
                return f"Error: Unknown action '{action}'"
        except Exception as exc:
            return f"Error: {exc}"


class DateTimeSkill(Skill):
    name = "datetime"
    description = "Get the current date and time, or convert between timezones."
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["now", "format"],
                "description": "'now' returns current datetime, 'format' returns formatted datetime.",
            },
            "format_str": {
                "type": "string",
                "description": "strftime format string (only for 'format' action). Default: '%Y-%m-%d %H:%M:%S'",
            },
        },
        "required": ["action"],
    }

    def execute(self, action: str = "now", format_str: str = "%Y-%m-%d %H:%M:%S", **kwargs) -> str:
        now = datetime.datetime.now()
        if action == "now":
            return now.isoformat()
        elif action == "format":
            return now.strftime(format_str)
        return f"Error: Unknown action '{action}'"


class SystemInfoSkill(Skill):
    name = "system_info"
    description = "Get system information including OS, CPU, memory, and disk usage."
    parameters = {
        "type": "object",
        "properties": {
            "info_type": {
                "type": "string",
                "enum": ["basic", "memory", "disk"],
                "description": "Type of system info to retrieve.",
            }
        },
        "required": ["info_type"],
    }

    def execute(self, info_type: str = "basic", **kwargs) -> str:
        import platform
        import os

        if info_type == "basic":
            return json.dumps({
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "python": platform.python_version(),
            }, indent=2)
        elif info_type == "memory":
            try:
                import psutil
                mem = psutil.virtual_memory()
                return json.dumps({
                    "total_gb": round(mem.total / (1024**3), 2),
                    "available_gb": round(mem.available / (1024**3), 2),
                    "used_percent": mem.percent,
                }, indent=2)
            except ImportError:
                return "psutil not installed. Install with: pip install psutil"
        elif info_type == "disk":
            try:
                import psutil
                disk = psutil.disk_usage("/")
                return json.dumps({
                    "total_gb": round(disk.total / (1024**3), 2),
                    "used_gb": round(disk.used / (1024**3), 2),
                    "free_gb": round(disk.free / (1024**3), 2),
                    "used_percent": disk.percent,
                }, indent=2)
            except ImportError:
                return "psutil not installed. Install with: pip install psutil"
        return f"Error: Unknown info_type '{info_type}'"
