---
name: text_transform
description: "Transform text with various operations - uppercase, lowercase, reverse, base64 encode/decode, and word count"
parameters:
  type: object
  properties:
    text:
      type: string
      description: The input text to transform
    operation:
      type: string
      description: "The transformation to apply: uppercase, lowercase, reverse, base64_encode, base64_decode, word_count"
  required:
    - text
    - operation
---

# Text Transform Skill

Transforms text using various operations.

## Usage Examples

- "Convert 'hello world' to uppercase" → operation: uppercase
- "Reverse the text 'abcdef'" → operation: reverse
- "Count words in 'hello world foo'" → operation: word_count

## Implementation

```python
import base64

def execute(text: str = "", operation: str = "lowercase", **kwargs):
    ops = {
        "uppercase": lambda t: t.upper(),
        "lowercase": lambda t: t.lower(),
        "reverse": lambda t: t[::-1],
        "base64_encode": lambda t: base64.b64encode(t.encode()).decode(),
        "base64_decode": lambda t: base64.b64decode(t.encode()).decode(),
        "word_count": lambda t: str(len(t.split())),
    }
    fn = ops.get(operation)
    if fn is None:
        return f"Unknown operation: {operation}. Available: {', '.join(ops.keys())}"
    return fn(text)
```
