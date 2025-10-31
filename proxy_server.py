import json
import os
import re
import time
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

import httpx
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse


app = FastAPI(title="OpenAI-Compatible Proxy -> Responses API")

# Default CORS for local OpenAI-compatible clients
cors_origins_env = os.getenv("PROXY_CORS_ORIGINS")
if cors_origins_env:
    origins = [o.strip() for o in cors_origins_env.split(",") if o.strip()]
else:
    origins = [
        "http://127.0.0.1:8000",
        "http://localhost:8000",
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins if origins else ["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)


def _read_file_text(path: str) -> Optional[str]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return None


def load_config() -> Dict[str, Any]:
    cfg: Dict[str, Any] = {
        "base_url": os.getenv("PROXY_UPSTREAM_BASE_URL"),
        "wire_api": os.getenv("PROXY_WIRE_API", "responses"),
        "default_model": os.getenv("PROXY_DEFAULT_MODEL"),
        "api_key": os.getenv("UPSTREAM_API_KEY") or os.getenv("OPENAI_API_KEY"),
    }

    # Fallbacks from local files when env vars are not present
    if not cfg["base_url"]:
        toml_text = _read_file_text("config.toml")
        if toml_text:
            # naive extract: the line base_url = "..."
            m = re.search(r"^\s*base_url\s*=\s*\"([^\"]+)\"", toml_text, re.MULTILINE)
            if m:
                cfg["base_url"] = m.group(1)
            # model if present
            mm = re.search(r"^\s*model\s*=\s*\"([^\"]+)\"", toml_text, re.MULTILINE)
            if mm:
                cfg["default_model"] = mm.group(1)
            wm = re.search(r"^\s*wire_api\s*=\s*\"([^\"]+)\"", toml_text, re.MULTILINE)
            if wm:
                cfg["wire_api"] = wm.group(1)

    if not cfg["api_key"]:
        auth_text = _read_file_text("auth.json")
        if auth_text:
            try:
                data = json.loads(auth_text)
                # prefer OPENAI_API_KEY key if present
                for k in ("UPSTREAM_API_KEY", "OPENAI_API_KEY", "API_KEY"):
                    if k in data and isinstance(data[k], str) and data[k].strip():
                        cfg["api_key"] = data[k].strip()
                        break
            except Exception:
                pass

    # Hard default if still not present
    if not cfg["base_url"]:
        cfg["base_url"] = "https://your-upstream.example.com/openai"
    if not cfg["default_model"]:
        cfg["default_model"] = "gpt-5"

    return cfg


CFG = load_config()


def upstream_endpoint(path: str) -> str:
    base = CFG["base_url"].rstrip("/")
    if not path.startswith("/"):
        path = "/" + path
    return base + path


def responses_api_path() -> str:
    # Allow override via env; default to '/responses' for this provider
    return os.getenv("PROXY_RESPONSES_PATH", "/responses")


def messages_to_responses_input(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert Chat Completions messages into Responses API input format.

    We map each message to { role, content: [{ type: "text", text }] }.
    Non-text content is ignored for simplicity.
    """
    converted: List[Dict[str, Any]] = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        parts: List[Dict[str, str]] = []

        # Determine upstream content type based on role:
        # - user/system/... => 'input_text'
        # - assistant => 'output_text' (represents prior assistant outputs in history)
        dest_type = "output_text" if role == "assistant" else "input_text"

        if isinstance(content, str):
            if content:
                parts.append({"type": dest_type, "text": content})
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    t = item.get("type")
                    if t in ("text", "input_text", "output_text") and isinstance(item.get("text"), str):
                        parts.append({"type": dest_type, "text": item["text"]})
                    # Ignore images and other types for now
                elif isinstance(item, str):
                    parts.append({"type": dest_type, "text": item})

        converted.append({"role": role, "content": parts})

    return converted


def chat_to_responses_payload(body: Dict[str, Any]) -> Tuple[Dict[str, Any], bool, str]:
    """Build a Responses API payload from a Chat Completions payload.

    Returns (payload, stream, model).
    """
    model = body.get("model") or CFG["default_model"]
    messages = body.get("messages") or []
    stream = bool(body.get("stream", False))

    payload: Dict[str, Any] = {
        "model": model,
        "input": messages_to_responses_input(messages),
    }

    if stream:
        payload["stream"] = True

    # Common parameter mappings
    if "temperature" in body:
        payload["temperature"] = body["temperature"]
    if "top_p" in body:
        payload["top_p"] = body["top_p"]
    if "max_tokens" in body:
        payload["max_output_tokens"] = body["max_tokens"]
    if "stop" in body:
        payload["stop_sequences"] = body["stop"]

    # Do NOT forward OpenAI penalties; upstream rejects them with 400
    # (e.g., {"detail":"Unsupported parameter: frequency_penalty"})
    # Some clients may still send them; we safely ignore here.

    return payload, stream, model


def responses_to_chat(resp: Dict[str, Any], model: str) -> Dict[str, Any]:
    """Translate a Responses API non-stream response to Chat Completions shape."""
    created = int(time.time())
    rid = resp.get("id") or f"proxy-resp-{int(time.time()*1000)}"

    text = ""
    finish_reason = "stop"

    # Attempt to extract text from multiple possible shapes
    if isinstance(resp.get("output"), list):
        # Prefer message blocks with content array
        for block in resp["output"]:
            if isinstance(block, dict):
                content = block.get("content")
                if isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict):
                            # common keys: text or output_text
                            if isinstance(part.get("text"), str):
                                text += part["text"]
                            elif isinstance(part.get("output_text"), str):
                                text += part["output_text"]
    elif isinstance(resp.get("content"), list):
        # Some providers may put content at top level
        for part in resp["content"]:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                text += part["text"]
    elif isinstance(resp.get("message"), dict):
        # Fallback to message.content
        msg = resp["message"]
        c = msg.get("content")
        if isinstance(c, str):
            text = c
        elif isinstance(c, list):
            for part in c:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    text += part["text"]
    elif isinstance(resp.get("choices"), list):
        # If upstream is already chat-completions-like
        ch0 = resp["choices"][0] if resp["choices"] else {}
        msg = ch0.get("message", {}) if isinstance(ch0, dict) else {}
        if isinstance(msg.get("content"), str):
            text = msg["content"]
        finish_reason = ch0.get("finish_reason", finish_reason)

    usage = resp.get("usage", {}) if isinstance(resp.get("usage"), dict) else {}
    prompt_tokens = usage.get("input_tokens") or usage.get("prompt_tokens")
    completion_tokens = usage.get("output_tokens") or usage.get("completion_tokens")
    total_tokens = usage.get("total_tokens") or (
        (prompt_tokens or 0) + (completion_tokens or 0) if (prompt_tokens is not None and completion_tokens is not None) else None
    )

    out: Dict[str, Any] = {
        "id": f"chatcmpl_{rid}",
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": finish_reason,
            }
        ],
    }

    if prompt_tokens is not None or completion_tokens is not None or total_tokens is not None:
        out["usage"] = {
            "prompt_tokens": prompt_tokens or 0,
            "completion_tokens": completion_tokens or 0,
            "total_tokens": total_tokens or (prompt_tokens or 0) + (completion_tokens or 0),
        }

    return out


def _sse_pack(data: Any) -> bytes:
    return ("data: " + (data if isinstance(data, str) else json.dumps(data)) + "\n\n").encode("utf-8")


def _chunk_for_text_delta(delta_text: str, model: str, cid: str, created: int, with_role: bool) -> Dict[str, Any]:
    delta_obj: Dict[str, Any] = {"content": delta_text}
    if with_role:
        delta_obj["role"] = "assistant"
    return {
        "id": cid,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": delta_obj,
                "finish_reason": None,
            }
        ],
    }


async def _bridge_responses_stream(resp: httpx.Response, model: str) -> AsyncGenerator[bytes, None]:
    created = int(time.time())
    cid = f"chatcmpl-{int(time.time()*1000)}"
    first = True
    event_name: Optional[str] = None
    data_lines: List[str] = []

    # If upstream isn't SSE, read whole body and emit as a single chunk
    ctype = resp.headers.get("content-type", "")
    if "text/event-stream" not in ctype:
        try:
            raw = await resp.aread()
            text_out = ""
            try:
                obj = json.loads(raw)
                # If upstream already returns chat.completion
                if isinstance(obj, dict) and isinstance(obj.get("choices"), list):
                    ch0 = obj["choices"][0] if obj["choices"] else {}
                    msg = ch0.get("message", {}) if isinstance(ch0, dict) else {}
                    if isinstance(msg.get("content"), str):
                        text_out = msg["content"]
                    else:
                        text_out = json.dumps(obj, ensure_ascii=False)
                else:
                    # Convert Responses-like JSON to text
                    res = responses_to_chat(obj, model=model)
                    text_out = res["choices"][0]["message"]["content"]
            except Exception:
                text_out = raw.decode("utf-8", "replace")

            # Emit one or two chunks and finish
            if first:
                yield _sse_pack(
                    {
                        "id": cid,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                    }
                )
                first = False
            yield _sse_pack(_chunk_for_text_delta(text_out, model, cid, created, with_role=False))
            yield _sse_pack(
                {
                    "id": cid,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
            )
            yield _sse_pack("[DONE]")
            return
        except Exception:
            # fall through to line iteration as a last resort
            pass

    async for raw_line in resp.aiter_lines():
        if raw_line is None:
            continue
        line = raw_line.rstrip("\r\n")
        if not line:
            # dispatch event
            if not data_lines:
                event_name = None
                continue
            payload_str = "\n".join(data_lines).strip()
            data_lines = []
            try:
                if payload_str == "[DONE]":
                    # final sentinel from upstream
                    # mirror OpenAI style
                    yield _sse_pack("[DONE]")
                    break

                obj = json.loads(payload_str)
            except Exception:
                # if not JSON, ignore
                event_name = None
                continue

            # Some providers include a top-level type or event
            typ = None
            if isinstance(obj, dict):
                typ = obj.get("type") or obj.get("event") or event_name

            # Directly pass through OpenAI chat.completion.chunk
            if isinstance(obj, dict) and "choices" in obj and obj.get("object") == "chat.completion.chunk":
                first = False
                yield _sse_pack(obj)
            elif isinstance(obj, dict) and (
                (isinstance(typ, str) and typ.endswith(".delta") and isinstance(obj.get("delta"), str))
                or (isinstance(obj.get("text"), str) and isinstance(typ, str) and ("output_text" in typ or "refusal" in typ))
            ):
                delta_text = obj.get("delta") if isinstance(obj.get("delta"), str) else obj.get("text") or ""
                if delta_text:
                    chunk = _chunk_for_text_delta(delta_text, model, cid, created, with_role=first)
                    first = False
                    yield _sse_pack(chunk)
            elif isinstance(obj, dict) and (typ == "response.completed" or typ == "response.stop" or obj.get("done") is True):
                # emit a final finish chunk then [DONE]
                final_chunk = {
                    "id": cid,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop",
                        }
                    ],
                }
                yield _sse_pack(final_chunk)
                yield _sse_pack("[DONE]")
                break
            elif isinstance(obj, dict) and "error" in obj:
                # surface upstream error in a simple way
                err = obj.get("error")
                if isinstance(err, dict):
                    msg = err.get("message") or str(err)
                else:
                    msg = str(err)
                # emit as a comment to not break clients
                yield (": error: " + msg + "\n\n").encode("utf-8")
            else:
                # Unknown event; ignore
                pass

            event_name = None
            continue

        if line.startswith(":"):
            # SSE comment, ignore
            continue
        if line.startswith("event:"):
            event_name = line[len("event:") :].strip()
            continue
        if line.startswith("data:"):
            data_lines.append(line[len("data:") :].lstrip())
            continue

    # safety: ensure end signal
    yield _sse_pack(
        {
            "id": cid,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
    )
    yield _sse_pack("[DONE]")


async def _collect_responses_stream_text(resp: httpx.Response, model: str) -> str:
    """Consume an upstream Responses SSE (or JSON) and return the final text."""
    ctype = resp.headers.get("content-type", "")
    text_parts: List[str] = []

    # If not SSE, parse as JSON or text
    if "text/event-stream" not in ctype:
        try:
            raw = await resp.aread()
            try:
                obj = json.loads(raw)
                res = responses_to_chat(obj, model=model)
                return res["choices"][0]["message"]["content"] or ""
            except Exception:
                return raw.decode("utf-8", "replace")
        except Exception:
            return ""

    # SSE parsing
    event_name: Optional[str] = None
    data_lines: List[str] = []
    async for raw_line in resp.aiter_lines():
        if raw_line is None:
            continue
        line = raw_line.rstrip("\r\n")
        if not line:
            if not data_lines:
                event_name = None
                continue
            payload_str = "\n".join(data_lines).strip()
            data_lines = []
            try:
                if payload_str == "[DONE]":
                    break
                obj = json.loads(payload_str)
            except Exception:
                event_name = None
                continue

            typ = None
            if isinstance(obj, dict):
                typ = obj.get("type") or obj.get("event") or event_name

            # If upstream already yields OpenAI chunks
            if isinstance(obj, dict) and obj.get("object") == "chat.completion.chunk" and isinstance(obj.get("choices"), list):
                ch0 = obj["choices"][0] if obj["choices"] else {}
                delta = ch0.get("delta", {}) if isinstance(ch0, dict) else {}
                if isinstance(delta.get("content"), str):
                    text_parts.append(delta["content"]) 
            elif isinstance(obj, dict) and (
                (isinstance(typ, str) and typ.endswith(".delta") and isinstance(obj.get("delta"), str))
                or (isinstance(obj.get("text"), str) and isinstance(typ, str) and ("output_text" in typ or "refusal" in typ))
            ):
                delta_text = obj.get("delta") if isinstance(obj.get("delta"), str) else obj.get("text") or ""
                if isinstance(delta_text, str) and delta_text:
                    text_parts.append(delta_text)
            elif isinstance(obj, dict) and (typ == "response.completed" or typ == "response.stop" or obj.get("done") is True):
                break
            elif isinstance(obj, dict) and "error" in obj:
                # capture error content too
                err = obj.get("error")
                msg = err.get("message") if isinstance(err, dict) else str(err)
                if msg:
                    text_parts.append(f"[upstream-error] {msg}")
            event_name = None
            continue

        if line.startswith(":"):
            continue
        if line.startswith("event:"):
            event_name = line[len("event:") :].strip()
            continue
        if line.startswith("data:"):
            data_lines.append(line[len("data:") :].lstrip())
            continue

    return "".join(text_parts)


def _resolve_upstream_key(authorization_header: Optional[str]) -> Optional[str]:
    # Prefer bearer token from client if present
    if authorization_header and authorization_header.lower().startswith("bearer "):
        token = authorization_header[7:].strip()
        if token:
            return token
    return CFG.get("api_key")


@app.get("/v1/models")
async def list_models(authorization: Optional[str] = Header(default=None, convert_underscores=False)):
    key = _resolve_upstream_key(authorization)
    headers = {"Accept": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"

    # Try to proxy upstream if it exists; otherwise, return a local stub
    url = upstream_endpoint("/v1/models")
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.get(url, headers=headers)
            if r.status_code == 200 and r.headers.get("content-type", "").startswith("application/json"):
                data = r.json()
                # Some providers return plain array; normalize to OpenAI shape
                if isinstance(data, dict) and "data" in data:
                    return JSONResponse(data)
                elif isinstance(data, list):
                    return JSONResponse({"object": "list", "data": data})
    except Exception:
        pass

    # Fallback stub with configured model
    model_id = CFG.get("default_model", "gpt-5")
    return JSONResponse(
        {
            "object": "list",
            "data": [
                {
                    "id": model_id,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "proxy",
                }
            ],
        }
    )


@app.post("/v1/chat/completions")
async def chat_completions(request: Request, authorization: Optional[str] = Header(default=None, convert_underscores=False)):
    try:
        body: Dict[str, Any] = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    payload, stream, model = chat_to_responses_payload(body)

    key = _resolve_upstream_key(authorization)
    if not key:
        # Allow missing key to proceed if upstream doesn't require; but warn
        pass

    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream" if stream else "application/json",
    }
    if key:
        headers["Authorization"] = f"Bearer {key}"

    # Some providers host Responses API at /responses (no /v1)
    url = upstream_endpoint(responses_api_path())

    if stream:
        # Upstream requires stream=true; always stream upstream and bridge
        payload["stream"] = True
        headers["Accept"] = "text/event-stream"
        async def streamer() -> AsyncGenerator[bytes, None]:
            created = int(time.time())
            cid = f"chatcmpl-{int(time.time()*1000)}"
            async with httpx.AsyncClient(timeout=None) as client:
                try:
                    upstream_resp = await client.post(url, headers=headers, json=payload)
                except Exception as e:
                    yield (f": proxy-error: {e}\n\n").encode("utf-8")
                    yield _sse_pack("[DONE]")
                    return

            if upstream_resp.status_code >= 400:
                try:
                    raw = upstream_resp.text
                except Exception:
                    raw = ""
                # Emit a readable error as chunks
                yield _sse_pack({
                    "id": cid,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                })
                yield _sse_pack({
                    "id": cid,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{"index": 0, "delta": {"content": f"Upstream error {upstream_resp.status_code}: {raw}"}, "finish_reason": None}],
                })
                yield _sse_pack({
                    "id": cid,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                })
                yield _sse_pack("[DONE]")
                return

            # Pass-through bridge of upstream SSE
            async for chunk in _bridge_responses_stream(upstream_resp, model=model):
                yield chunk

        return StreamingResponse(streamer(), media_type="text/event-stream")
    else:
        # Client asked non-stream, but upstream requires streaming; consume SSE and return full text
        payload["stream"] = True
        headers["Accept"] = "text/event-stream"
        async with httpx.AsyncClient(timeout=None) as client:
            try:
                upstream_resp = await client.post(url, headers=headers, json=payload)
            except Exception as e:
                raise HTTPException(status_code=502, detail=f"Upstream error: {e}")

            if upstream_resp.status_code >= 400:
                try:
                    detail = upstream_resp.json()
                except Exception:
                    detail = {"error": upstream_resp.text}
                raise HTTPException(status_code=upstream_resp.status_code, detail=detail)

            try:
                text = await _collect_responses_stream_text(upstream_resp, model=model)
            except Exception as e:
                raise HTTPException(status_code=502, detail=f"Upstream stream parse error: {e}")

            created = int(time.time())
            out = {
                "id": f"chatcmpl_{int(time.time()*1000)}",
                "object": "chat.completion",
                "created": created,
                "model": model,
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
                ],
            }
            return JSONResponse(out)


@app.get("/health")
async def health():
    return {"status": "ok", "upstream": CFG.get("base_url"), "wire_api": CFG.get("wire_api")}


@app.get("/")
async def root():
    return {
        "message": "OpenAI-compatible proxy is running",
        "endpoints": ["/v1/models", "/v1/chat/completions", "/health"],
    }


def _main():
    import uvicorn

    host = os.getenv("PROXY_HOST", "127.0.0.1")
    # Lock to 8010 by default; no auto fallback
    port = int(os.getenv("PROXY_PORT", "8010"))

    print(f"[proxy] Starting on http://{host}:{port}")
    uvicorn.run("proxy_server:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    _main()
