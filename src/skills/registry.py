from __future__ import annotations

from typing import Any

from .base import Skill


class SkillRegistry:
    _skills: dict[str, Skill]

    def __init__(self):
        self._skills = {}

    def register(self, skill: Skill) -> None:
        if skill.name in self._skills:
            raise ValueError(f"Skill '{skill.name}' already registered")
        self._skills[skill.name] = skill

    def unregister(self, name: str) -> None:
        self._skills.pop(name, None)

    def get(self, name: str) -> Skill | None:
        return self._skills.get(name)

    def list_skills(self) -> list[str]:
        return list(self._skills.keys())

    def get_openai_tools(self) -> list[dict[str, Any]]:
        return [skill.to_openai_tool() for skill in self._skills.values()]

    def execute(self, name: str, **kwargs) -> str:
        skill = self.get(name)
        if skill is None:
            raise ValueError(f"Skill '{name}' not found")
        return skill.execute(**kwargs)


_global_registry = SkillRegistry()


def get_registry() -> SkillRegistry:
    return _global_registry
