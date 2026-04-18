from __future__ import annotations

import json

from src.tools.client import GemmaClient, LlamaServerManager
from src.skills.builtin import register_builtin_skills
from src.skills.registry import get_registry


def main():
    print("=" * 60)
    print("  Gemma-4-E4B Interactive Chat with Tool Calling")
    print("=" * 60)

    print("\nRegistering skills...")
    register_builtin_skills()
    registry = get_registry()
    print(f"Available skills: {', '.join(registry.list_skills())}")

    print("\nStarting llama-server (this may take a moment)...")
    server = LlamaServerManager()

    try:
        server.start()
        client = GemmaClient(registry=registry, server_manager=server)
        print("Ready!\n")

        messages: list[dict] = []

        print("Type your message and press Enter. Commands:")
        print("  /tools  - List available tools")
        print("  /clear  - Clear conversation history")
        print("  /quit   - Exit")
        print("  /raw    - Toggle raw tool call display")
        print()

        show_raw = False

        while True:
            try:
                user_input = input("\n👤 You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if not user_input:
                continue

            if user_input == "/quit":
                print("Goodbye!")
                break
            elif user_input == "/clear":
                messages.clear()
                print("Conversation cleared.")
                continue
            elif user_input == "/tools":
                tools = client.executor.get_tools_schema()
                for t in tools:
                    func = t["function"]
                    print(f"  - {func['name']}: {func['description'][:80]}")
                continue
            elif user_input == "/raw":
                show_raw = not show_raw
                print(f"Raw tool calls: {'ON' if show_raw else 'OFF'}")
                continue

            messages.append({"role": "user", "content": user_input})

            print("\n🤖 Assistant: ", end="", flush=True)

            full_response = ""
            for chunk in client.chat_stream(messages):
                if chunk.startswith('{"type": "tool_calls"'):
                    if show_raw:
                        print(f"\n[TOOL CALL] {chunk.strip()}", end="", flush=True)
                    else:
                        try:
                            data = json.loads(chunk.strip())
                            for call in data.get("calls", []):
                                print(f"\n  🔧 Calling {call['name']}({call['arguments']})...", end="", flush=True)
                        except json.JSONDecodeError:
                            pass
                else:
                    print(chunk, end="", flush=True)
                    full_response += chunk

            print()

            if full_response:
                messages.append({"role": "assistant", "content": full_response})

    finally:
        server.stop()


if __name__ == "__main__":
    main()
