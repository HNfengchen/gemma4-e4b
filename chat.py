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
    print(f"Built-in skills: {', '.join(registry.list_skills())}")

    print("\nStarting llama-server...")
    server = LlamaServerManager()

    try:
        server.start()
        client = GemmaClient(registry=registry, server_manager=server)
        print(f"All skills: {', '.join(client.registry.list_skills())}")
        print("Ready!\n")

        messages: list[dict] = []

        print("Type your message and press Enter. Commands:")
        print("  /tools    - List available tools")
        print("  /clear    - Clear conversation history")
        print("  /quit     - Exit")
        print("  /raw      - Toggle raw tool call display")
        print("  /reload   - Reload skill.md files from skills/ directory")
        print("  /mcp      - Connect to MCP server (usage: /mcp <name> <command>)")
        print("  /agents   - Show current AGENTS.md system prompt")
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
            elif user_input == "/reload":
                loaded = client.reload_skills()
                print(f"Reloaded skills: {', '.join(loaded) if loaded else 'none'}")
                print(f"All skills: {', '.join(client.registry.list_skills())}")
                continue
            elif user_input.startswith("/mcp "):
                parts = user_input[5:].strip().split(None, 1)
                if len(parts) < 2:
                    print("Usage: /mcp <name> <command_or_url>")
                    continue
                name = parts[0]
                cmd_or_url = parts[1]
                try:
                    if cmd_or_url.startswith("http"):
                        loaded = client.connect_mcp_http(name, cmd_or_url)
                    else:
                        loaded = client.connect_mcp_stdio(name, cmd_or_url.split())
                    print(f"MCP '{name}' connected. Tools: {', '.join(loaded) if loaded else 'none'}")
                except Exception as e:
                    print(f"MCP connection failed: {e}")
                continue
            elif user_input == "/agents":
                print(f"System prompt ({len(client._system_prompt)} chars):")
                print(client._system_prompt[:500] + ("..." if len(client._system_prompt) > 500 else ""))
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
        client.close()
        server.stop()


if __name__ == "__main__":
    main()
