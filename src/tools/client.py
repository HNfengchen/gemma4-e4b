from __future__ import annotations

import json
import requests
import subprocess
import sys
import time
from typing import Any, Generator

from src.config import (
    LLAMA_SERVER,
    MODEL_FILE,
    MMPROJ_FILE,
    SERVER_HOST,
    SERVER_PORT,
    N_CTX,
    N_BATCH,
    N_PARALLEL,
    TEMPERATURE,
    TOP_P,
    TOP_K,
    TOOL_CALL_MAX_ITERATIONS,
)
from src.tools.executor import ToolExecutor
from src.skills.registry import SkillRegistry, get_registry


class LlamaServerManager:
    def __init__(
        self,
        host: str = SERVER_HOST,
        port: int = SERVER_PORT,
        n_ctx: int = N_CTX,
        n_batch: int = N_BATCH,
    ):
        self.host = host
        self.port = port
        self.n_ctx = n_ctx
        self.n_batch = n_batch
        self._process: subprocess.Popen | None = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def api_url(self) -> str:
        return f"{self.base_url}/v1"

    def start(self, timeout: int = 120) -> None:
        if not LLAMA_SERVER.exists():
            raise FileNotFoundError(f"llama-server not found: {LLAMA_SERVER}")
        if not MODEL_FILE.exists():
            raise FileNotFoundError(f"Model not found: {MODEL_FILE}")

        cmd = [
            str(LLAMA_SERVER),
            "-m", str(MODEL_FILE),
            "--mmproj", str(MMPROJ_FILE),
            "--host", self.host,
            "--port", str(self.port),
            "-c", str(self.n_ctx),
            "-b", str(self.n_batch),
            "-np", str(N_PARALLEL),
            "--jinja",
            "-ngl", "99",
        ]

        self._process = subprocess.Popen(
            cmd,
            cwd=str(LLAMA_SERVER.parent),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                resp = requests.get(f"{self.base_url}/v1/models", timeout=2)
                if resp.status_code == 200:
                    print(f"llama-server started at {self.base_url}")
                    return
            except (requests.ConnectionError, requests.Timeout):
                pass
            time.sleep(2)

        self.stop()
        raise TimeoutError(f"llama-server failed to start within {timeout}s")

    def stop(self) -> None:
        if self._process and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None

    def is_running(self) -> bool:
        if self._process is None:
            return False
        return self._process.poll() is None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()


class GemmaClient:
    def __init__(
        self,
        registry: SkillRegistry | None = None,
        server_manager: LlamaServerManager | None = None,
    ):
        self.executor = ToolExecutor(registry or get_registry())
        self.server = server_manager or LlamaServerManager()
        self._system_prompt = (
            "You are a helpful AI assistant with access to tools. "
            "When you need to perform calculations, access files, execute code, "
            "or fetch web content, use the available tools. "
            "Always respond in the user's language."
        )
        self._session = requests.Session()

    def _chat_completion(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = TEMPERATURE,
        top_p: float = TOP_P,
        top_k: int = TOP_K,
        max_tokens: int = 2048,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = tools

        resp = self._session.post(
            f"{self.server.api_url}/chat/completions",
            json=payload,
            timeout=300,
        )
        resp.raise_for_status()
        return resp.json()

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = TEMPERATURE,
        top_p: float = TOP_P,
        top_k: int = TOP_K,
        max_tokens: int = 2048,
    ) -> dict[str, Any]:
        if tools is None:
            tools = self.executor.get_tools_schema()

        all_messages = [{"role": "system", "content": self._system_prompt}] + messages

        for _ in range(TOOL_CALL_MAX_ITERATIONS):
            response = self._chat_completion(
                all_messages, tools, temperature, top_p, top_k, max_tokens
            )

            choice = response["choices"][0]
            message = choice["message"]

            if message.get("tool_calls"):
                all_messages.append(message)
                tool_results = self.executor.process_tool_calls(message)
                all_messages.extend(tool_results)
            else:
                return response

        return response

    def chat_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = TEMPERATURE,
        top_p: float = TOP_P,
        top_k: int = TOP_K,
        max_tokens: int = 2048,
    ) -> Generator[str, None, None]:
        if tools is None:
            tools = self.executor.get_tools_schema()

        all_messages = [{"role": "system", "content": self._system_prompt}] + messages

        for _ in range(TOOL_CALL_MAX_ITERATIONS):
            response = self._chat_completion(
                all_messages, tools, temperature, top_p, top_k, max_tokens
            )

            choice = response["choices"][0]
            message = choice["message"]

            if message.get("tool_calls"):
                all_messages.append(message)
                tool_results = self.executor.process_tool_calls(message)
                all_messages.extend(tool_results)
                yield json.dumps(
                    {"type": "tool_calls", "calls": [
                        {"name": tc["function"]["name"], "arguments": tc["function"]["arguments"]}
                        for tc in message["tool_calls"]
                    ]},
                    ensure_ascii=False,
                ) + "\n"
            else:
                content = message.get("content", "")
                yield content
                return

        content = message.get("content", "")
        yield content

    def simple_chat(self, user_message: str, **kwargs) -> str:
        messages = [{"role": "user", "content": user_message}]
        response = self.chat(messages, **kwargs)
        return response["choices"][0]["message"].get("content", "")

    def close(self):
        self._session.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
