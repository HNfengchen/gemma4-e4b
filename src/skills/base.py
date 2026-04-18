from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Skill(ABC):
    name: str = ""
    description: str = ""
    parameters: dict[str, Any] = {}

    @abstractmethod
    def execute(self, **kwargs) -> str:
        raise NotImplementedError

    def to_openai_tool(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
