from __future__ import annotations

import json
import time
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

from src.tools.client import GemmaClient, LlamaServerManager, MCPManager
from src.skills.builtin import register_builtin_skills
from src.skills.registry import get_registry
from src.config import SERVER_HOST, SERVER_PORT


app = FastAPI(title="Gemma-4-E4B API Server", version="2.0.0")

client: GemmaClient | None = None
server_manager: LlamaServerManager | None = None


class ChatMessage(BaseModel):
    role: str
    content: str | None = None
    name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None


class ToolDef(BaseModel):
    type: str = "function"
    function: dict[str, Any]


class ChatCompletionRequest(BaseModel):
    model: str = "gemma-4-e4b"
    messages: list[ChatMessage]
    tools: list[ToolDef] | None = None
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 64
    max_tokens: int = 2048
    stream: bool = False


class MCPServerConfig(BaseModel):
    name: str
    command: list[str] | None = None
    url: str | None = None


@app.on_event("startup")
async def startup():
    global client, server_manager
    register_builtin_skills()
    server_manager = LlamaServerManager()
    server_manager.start()
    client = GemmaClient(registry=get_registry(), server_manager=server_manager)


@app.on_event("shutdown")
async def shutdown():
    if client:
        client.close()
    if server_manager:
        server_manager.stop()


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "gemma-4-e4b",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local",
            }
        ],
    }


@app.get("/v1/models/{model_id}")
async def get_model(model_id: str):
    return {
        "id": model_id,
        "object": "model",
        "created": int(time.time()),
        "owned_by": "local",
    }


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    if client is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    messages = [m.model_dump(exclude_none=True) for m in req.messages]
    tools = [t.model_dump() for t in req.tools] if req.tools else None

    if req.stream:
        return StreamingResponse(
            _stream_response(messages, tools, req),
            media_type="text/event-stream",
        )

    response = client.chat(
        messages=messages,
        tools=tools,
        temperature=req.temperature,
        top_p=req.top_p,
        top_k=req.top_k,
        max_tokens=req.max_tokens,
    )

    return JSONResponse(content=response)


def _stream_response(
    messages: list[dict],
    tools: list[dict] | None,
    req: ChatCompletionRequest,
):
    for chunk in client.chat_stream(
        messages=messages,
        tools=tools,
        temperature=req.temperature,
        top_p=req.top_p,
        top_k=req.top_k,
        max_tokens=req.max_tokens,
    ):
        data = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": req.model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": chunk},
                    "finish_reason": None,
                }
            ],
        }
        yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

    yield "data: [DONE]\n\n"


@app.get("/v1/tools")
async def list_tools():
    if client is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"tools": client.executor.get_tools_schema()}


@app.post("/v1/mcp/connect")
async def connect_mcp(config: MCPServerConfig):
    if client is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        if config.command:
            loaded = client.connect_mcp_stdio(config.name, config.command)
        elif config.url:
            loaded = client.connect_mcp_http(config.name, config.url)
        else:
            raise HTTPException(status_code=400, detail="Must provide 'command' or 'url'")
        return {"status": "ok", "tools_loaded": loaded}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/skills/reload")
async def reload_skills():
    if client is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    loaded = client.reload_skills()
    return {"status": "ok", "skills_loaded": loaded}


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": client is not None}


def run_server(host: str = SERVER_HOST, port: int = SERVER_PORT):
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
