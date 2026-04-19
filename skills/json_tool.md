---
name: json_tool
description: Parse, format, validate, or query JSON data using JSONPath-like expressions
parameters:
  type: object
  properties:
    action:
      type: string
      description: "Action to perform: parse, format, validate, get"
    json_str:
      type: string
      description: The JSON string to process
    key:
      type: string
      description: Key to extract (for 'get' action, dot notation supported e.g. 'data.items')
  required:
    - action
    - json_str
---

# JSON Tool Skill

Process and manipulate JSON data.

## Implementation

```python
import json

def execute(action: str = "parse", json_str: str = "", key: str = "", **kwargs):
    try:
        if action == "validate":
            json.loads(json_str)
            return "Valid JSON"
        elif action == "parse":
            data = json.loads(json_str)
            return json.dumps(data, indent=2, ensure_ascii=False)
        elif action == "format":
            data = json.loads(json_str)
            return json.dumps(data, indent=2, ensure_ascii=False)
        elif action == "get":
            data = json.loads(json_str)
            if not key:
                return json.dumps(data, ensure_ascii=False)
            parts = key.split(".")
            current = data
            for part in parts:
                if isinstance(current, dict):
                    current = current.get(part)
                elif isinstance(current, list):
                    try:
                        current = current[int(part)]
                    except (ValueError, IndexError):
                        return f"Key not found: {key}"
                else:
                    return f"Key not found: {key}"
                if current is None:
                    return f"Key not found: {key}"
            return json.dumps(current, indent=2, ensure_ascii=False)
        else:
            return f"Unknown action: {action}. Available: parse, format, validate, get"
    except json.JSONDecodeError as e:
        return f"Invalid JSON: {e}"
    except Exception as e:
        return f"Error: {e}"
```
