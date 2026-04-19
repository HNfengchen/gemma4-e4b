from __future__ import annotations

import re
import json
import math
import base64
import datetime
import collections
import itertools
import os
import sys
from pathlib import Path
from typing import Any

from src.skills.base import Skill


class MarkdownSkill(Skill):
    def __init__(self, name: str, description: str, parameters: dict[str, Any], code: str):
        self.name = name
        self.description = description
        self.parameters = parameters
        self._code = code
        self._compiled = None

    def execute(self, **kwargs) -> str:
        if self._compiled is None:
            global_ns = {
                "__builtins__": __builtins__,
                "json": json,
                "math": math,
                "base64": base64,
                "datetime": datetime,
                "collections": collections,
                "itertools": itertools,
                "os": os,
                "sys": sys,
                "Path": Path,
                "re": re,
            }
            local_ns: dict[str, Any] = {}
            try:
                exec(self._code, global_ns, local_ns)
            except Exception as e:
                return f"Error compiling skill '{self.name}': {e}"
            self._compiled = local_ns.get("execute")
            if self._compiled is None:
                return f"Error: skill '{self.name}' code must define an 'execute' function"

        try:
            result = self._compiled(**kwargs)
            return str(result)
        except Exception as e:
            return f"Error executing skill '{self.name}': {e}"


def parse_skill_md(filepath: str | Path) -> MarkdownSkill:
    filepath = Path(filepath)
    content = filepath.read_text(encoding="utf-8")

    frontmatter: dict[str, Any] = {}
    fm_match = re.match(r"^---\s*\n(.*?)\n---\s*\n", content, re.DOTALL)
    body = content
    if fm_match:
        fm_text = fm_match.group(1)
        body = content[fm_match.end():]
        try:
            import yaml
            frontmatter = yaml.safe_load(fm_text) or {}
        except ImportError:
            for line in fm_text.splitlines():
                line = line.strip()
                if ":" in line:
                    key, _, val = line.partition(":")
                    frontmatter[key.strip()] = val.strip()

    name = frontmatter.get("name", filepath.stem)
    description = frontmatter.get("description", "")
    parameters = frontmatter.get("parameters", {
        "type": "object",
        "properties": {},
    })

    code = ""
    code_blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", body, re.DOTALL)
    if code_blocks:
        code = code_blocks[0].strip()

    if not description:
        desc_match = re.search(r"^##\s*Description\s*\n+(.*?)(?=\n##|\Z)", body, re.DOTALL | re.MULTILINE)
        if desc_match:
            description = desc_match.group(1).strip()

    if not code:
        impl_match = re.search(r"^##\s*Implementation\s*\n+```(?:python)?\s*\n(.*?)```", body, re.DOTALL | re.MULTILINE)
        if impl_match:
            code = impl_match.group(1).strip()

    if not parameters.get("properties") and not parameters.get("type"):
        param_section = re.search(r"^##\s*Parameters\s*\n+(.*?)(?=\n##|\Z)", body, re.DOTALL | re.MULTILINE)
        if param_section:
            props: dict[str, Any] = {}
            required: list[str] = []
            for line in param_section.group(1).strip().splitlines():
                m = re.match(r"-\s+(\w+)\s*\((\w+)(?:,\s*(required|optional))?\):\s*(.*)", line)
                if m:
                    pname, ptype, preq, pdesc = m.groups()
                    props[pname] = {"type": ptype, "description": pdesc}
                    if preq == "required":
                        required.append(pname)
            if props:
                parameters = {"type": "object", "properties": props, "required": required}

    return MarkdownSkill(name=name, description=description, parameters=parameters, code=code)


def load_skills_from_directory(dirpath: str | Path) -> list[MarkdownSkill]:
    dirpath = Path(dirpath)
    skills: list[MarkdownSkill] = []
    if not dirpath.is_dir():
        return skills
    for f in sorted(dirpath.glob("*.md")):
        try:
            skill = parse_skill_md(f)
            skills.append(skill)
        except Exception as e:
            print(f"[WARN] Failed to parse skill file {f}: {e}")
    return skills
