from __future__ import annotations

import json

from src.skills.builtin import register_builtin_skills
from src.skills.registry import get_registry
from src.tools.executor import ToolExecutor
from src.tools.client import GemmaClient, LlamaServerManager
from src.config import MODEL_FILE, MMPROJ_FILE, LLAMA_SERVER


def test_skills_registry():
    print("=" * 50)
    print("Test 1: Skills Registry")
    print("=" * 50)

    register_builtin_skills()
    registry = get_registry()

    skills = registry.list_skills()
    print(f"Registered skills: {skills}")
    assert len(skills) == 7, f"Expected 7 skills, got {len(skills)}"
    print("PASS: 7 skills registered\n")


def test_tool_executor():
    print("=" * 50)
    print("Test 2: Tool Executor")
    print("=" * 50)

    register_builtin_skills()
    executor = ToolExecutor(get_registry())

    tool_call = {
        "id": "call_001",
        "type": "function",
        "function": {
            "name": "calculator",
            "arguments": '{"expression": "2 + 3 * 4"}',
        },
    }
    result = executor.execute_tool_call(tool_call)
    print(f"Calculator: 2 + 3 * 4 = {result['content']}")
    assert result["content"] == "14", f"Expected 14, got {result['content']}"

    tool_call2 = {
        "id": "call_002",
        "type": "function",
        "function": {
            "name": "datetime",
            "arguments": '{"action": "format", "format_str": "%Y-%m-%d"}',
        },
    }
    result2 = executor.execute_tool_call(tool_call2)
    print(f"DateTime: {result2['content']}")
    assert len(result2["content"]) == 10

    tool_call3 = {
        "id": "call_003",
        "type": "function",
        "function": {
            "name": "nonexistent_tool",
            "arguments": "{}",
        },
    }
    result3 = executor.execute_tool_call(tool_call3)
    print(f"Nonexistent tool: {result3['content']}")
    assert "Error" in result3["content"]

    print("PASS: Tool executor works correctly\n")


def test_openai_tool_schema():
    print("=" * 50)
    print("Test 3: OpenAI Tool Schema")
    print("=" * 50)

    register_builtin_skills()
    registry = get_registry()
    tools = registry.get_openai_tools()

    for tool in tools:
        func = tool["function"]
        assert tool["type"] == "function"
        assert "name" in func
        assert "description" in func
        assert "parameters" in func
        print(f"  {func['name']}: schema OK")

    print("PASS: All tool schemas valid\n")


def test_model_loading():
    print("=" * 50)
    print("Test 4: Model Loading & Inference")
    print("=" * 50)

    if not MODEL_FILE.exists():
        print(f"SKIP: Model file not found at {MODEL_FILE}")
        print("Run 'python start.py download' first.\n")
        return None

    if not LLAMA_SERVER.exists():
        print(f"SKIP: llama-server not found at {LLAMA_SERVER}")
        print("Run 'python start.py download' first.\n")
        return None

    register_builtin_skills()
    server = LlamaServerManager()

    try:
        print("Starting llama-server...")
        server.start()

        client = GemmaClient(registry=get_registry(), server_manager=server)

        print("\nTesting simple chat (no tools)...")
        response = client.simple_chat("Hello! What is 2+2? Answer briefly.")
        print(f"Response: {response[:200]}")

        print("\nTesting chat with tools...")
        messages = [{"role": "user", "content": "Use the calculator tool to compute 123 * 456"}]
        response = client.chat(messages)
        content = response["choices"][0]["message"].get("content", "")
        print(f"Response: {content[:200]}")

        print("\nPASS: Model loading and inference work\n")
        return client
    finally:
        server.stop()


def main():
    test_skills_registry()
    test_tool_executor()
    test_openai_tool_schema()
    client = test_model_loading()

    if client:
        print("=" * 50)
        print("All tests passed!")
        print("=" * 50)
    else:
        print("=" * 50)
        print("Basic tests passed (model tests skipped)")
        print("=" * 50)


if __name__ == "__main__":
    main()
