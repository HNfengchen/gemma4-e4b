from __future__ import annotations

import math
from src.skills.base import Skill


class CalculatorSkill(Skill):
    name = "calculator"
    description = "Evaluate a mathematical expression. Supports basic arithmetic, trigonometric functions, logarithms, and constants like pi and e."
    parameters = {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "The mathematical expression to evaluate, e.g. '2 + 3 * 4' or 'sin(pi/2)'",
            }
        },
        "required": ["expression"],
    }

    _SAFE_DICT = {
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sum": sum,
        "pow": pow,
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "asin": math.asin,
        "acos": math.acos,
        "atan": math.atan,
        "log": math.log,
        "log10": math.log10,
        "log2": math.log2,
        "exp": math.exp,
        "ceil": math.ceil,
        "floor": math.floor,
        "pi": math.pi,
        "e": math.e,
    }

    def execute(self, expression: str = "", **kwargs) -> str:
        try:
            result = eval(expression, {"__builtins__": {}}, self._SAFE_DICT)
            return str(result)
        except Exception as exc:
            return f"Error evaluating expression: {exc}"
