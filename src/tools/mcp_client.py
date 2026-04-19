from __future__ import annotations

import json
import subprocess
import sys
import threading
from typing import Any

from src.skills.base import Skill


class MCPToolSkill(Skill):
    def __init__(self, name: str, description: str, parameters: dict[str, Any], mcp_client: MCPClient):
        self.name = name
        self.description = description
        self.parameters = parameters
        self._mcp_client = mcp_client

    def execute(self, **kwargs) -> str:
        result = self._mcp_client.call_tool(self.name, kwargs)
        if result.get("isError"):
            texts = [c.get("text", "") for c in result.get("content", []) if c.get("type") == "text"]
            return f"Error: {'; '.join(texts)}"
        texts = [c.get("text", "") for c in result.get("content", []) if c.get("type") == "text"]
        return "\n".join(texts) if texts else json.dumps(result, ensure_ascii=False)


class MCPClient:
    def __init__(self, server_command: list[str] | None = None, server_url: str | None = None):
        self._server_command = server_command
        self._server_url = server_url
        self._process: subprocess.Popen | None = None
        self._request_id = 0
        self._lock = threading.Lock()
        self._tools: list[dict[str, Any]] = []
        self._initialized = False

    def start(self) -> None:
        if self._server_command:
            self._start_stdio()
        elif self._server_url:
            self._start_http()
        else:
            raise ValueError("Either server_command or server_url must be provided")

    def _start_stdio(self) -> None:
        self._process = subprocess.Popen(
            self._server_command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self._initialize()
        self._tools = self._list_tools()
        self._initialized = True

    def _start_http(self) -> None:
        import requests
        self._initialize_http()
        self._tools = self._list_tools_http()
        self._initialized = True

    def stop(self) -> None:
        if self._process and self._process.poll() is None:
            self._send_request("shutdown", {})
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None
        self._initialized = False

    def _next_id(self) -> int:
        with self._lock:
            self._request_id += 1
            return self._request_id

    def _send_request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        if not self._process or not self._process.stdin or not self._process.stdout:
            raise RuntimeError("MCP server process not running")

        msg = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": method,
            "params": params,
        }
        data = json.dumps(msg) + "\n"
        self._process.stdin.write(data.encode("utf-8"))
        self._process.stdin.flush()

        response_line = self._process.stdout.readline()
        if not response_line:
            raise RuntimeError("MCP server closed connection")
        response = json.loads(response_line.decode("utf-8"))

        if "error" in response:
            raise RuntimeError(f"MCP error: {response['error']}")
        return response.get("result", {})

    def _send_http_request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        import requests as req

        msg = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": method,
            "params": params,
        }
        resp = req.post(
            f"{self._server_url}/mcp",
            json=msg,
            timeout=30,
        )
        resp.raise_for_status()
        response = resp.json()
        if "error" in response:
            raise RuntimeError(f"MCP error: {response['error']}")
        return response.get("result", {})

    def _initialize(self) -> dict[str, Any]:
        return self._send_request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": "gemma4-e4b-client",
                "version": "1.0.0",
            },
        })

    def _initialize_http(self) -> dict[str, Any]:
        return self._send_http_request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": "gemma4-e4b-client",
                "version": "1.0.0",
            },
        })

    def _list_tools(self) -> list[dict[str, Any]]:
        result = self._send_request("tools/list", {})
        return result.get("tools", [])

    def _list_tools_http(self) -> list[dict[str, Any]]:
        result = self._send_http_request("tools/list", {})
        return result.get("tools", [])

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if self._server_command and self._process:
            return self._send_request("tools/call", {"name": name, "arguments": arguments})
        elif self._server_url:
            return self._send_http_request("tools/call", {"name": name, "arguments": arguments})
        raise RuntimeError("MCP client not connected")

    def get_tools(self) -> list[dict[str, Any]]:
        return self._tools

    def get_tool_skills(self) -> list[MCPToolSkill]:
        skills: list[MCPToolSkill] = []
        for tool in self._tools:
            name = tool.get("name", "")
            description = tool.get("description", "")
            input_schema = tool.get("inputSchema", {
                "type": "object",
                "properties": {},
            })
            skills.append(MCPToolSkill(
                name=name,
                description=description,
                parameters=input_schema,
                mcp_client=self,
            ))
        return skills

    @property
    def is_initialized(self) -> bool:
        return self._initialized


class MCPManager:
    def __init__(self):
        self._clients: dict[str, MCPClient] = {}

    def add_stdio_server(self, name: str, command: list[str]) -> MCPClient:
        client = MCPClient(server_command=command)
        client.start()
        self._clients[name] = client
        return client

    def add_http_server(self, name: str, url: str) -> MCPClient:
        client = MCPClient(server_url=url)
        client.start()
        self._clients[name] = client
        return client

    def remove_server(self, name: str) -> None:
        client = self._clients.pop(name, None)
        if client:
            client.stop()

    def get_all_tool_skills(self) -> list[MCPToolSkill]:
        skills: list[MCPToolSkill] = []
        for client in self._clients.values():
            skills.extend(client.get_tool_skills())
        return skills

    def stop_all(self) -> None:
        for client in self._clients.values():
            client.stop()
        self._clients.clear()

    def list_servers(self) -> list[str]:
        return list(self._clients.keys())


def load_mcp_config(config_path: str) -> dict[str, Any]:
    from pathlib import Path
    p = Path(config_path)
    if not p.exists():
        return {}
    content = p.read_text(encoding="utf-8")
    if p.suffix in (".yaml", ".yml"):
        try:
            import yaml
            return yaml.safe_load(content) or {}
        except ImportError:
            pass
    elif p.suffix == ".json":
        return json.loads(content)
    return {}
