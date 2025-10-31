# ResponseBridge
An agent for converting OpenAl-Response type APIs to OpenAI-compatible formats.

## 项目简介
- **OpenAI 兼容入口**：通过 FastAPI 暴露 `/v1/chat/completions` 与 `/v1/models` 等端点，保持对 OpenAI SDK / 客户端的兼容。
- **Responses API 转换器**：将 Chat Completions 的消息与参数转换为 Responses API 需要的结构，并对返回内容进行反向映射。
- **流式桥接**：在 `stream=true` 时与上游保持 SSE 连接，按 OpenAI 的 `chat.completion.chunk` 规范回放数据。
- **弹性配置**：环境变量优先，其次读取 `config.toml` 和 `auth.json`，方便在不同部署环境下切换。

## 快速开始
1. **准备环境**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate                     # Windows
   # source .venv/bin/activate               # macOS / Linux
   pip install -r requirements.txt
   ```
2. **配置上游信息**
   - 设置环境变量（推荐），或编辑 `config.toml`/`auth.json`。
   - 至少需要 `PROXY_UPSTREAM_BASE_URL` 与上游可用的 API Key。
3. **启动服务**
   ```bash
   python proxy_server.py
   # 或者
   uvicorn proxy_server:app --host 0.0.0.0 --port 8010
   ```
4. **健康检查**
   - 打开浏览器访问 `http://127.0.0.1:8010/health`，确认返回 `{"status": "ok", ...}`。

## 配置说明
- **环境变量优先级最高**，用于快速调试与容器部署：
  - `PROXY_UPSTREAM_BASE_URL`：Responses API 或其他 OpenAI 兼容服务基础地址。
  - `PROXY_WIRE_API`：上游协议名称，默认 `responses`。
  - `PROXY_DEFAULT_MODEL`：未显式传入时默认的 `model`。
  - `UPSTREAM_API_KEY` / `OPENAI_API_KEY`：上游鉴权密钥；客户端未带 `Authorization` 时使用。
  - `PROXY_RESPONSES_PATH`：Responses API 的路径前缀，默认 `/responses`。
  - `PROXY_CORS_ORIGINS`：逗号分隔的来源列表，用于浏览器场景。
  - `PROXY_HOST` / `PROXY_PORT`：内置启动脚本的监听地址与端口（默认 `127.0.0.1:8010`）。
- **`config.toml`**：在缺省环境变量时提供后备值，例如：
  ```toml
  model = "gpt-5"
  base_url = "https://your-upstream.example.com/openai"
  wire_api = "responses"
  ```
- **`auth.json`**：同样作为 API Key 的后备来源：
  ```json
  { "OPENAI_API_KEY": "sk-..." }
  ```
  当文件中存在 `UPSTREAM_API_KEY` / `OPENAI_API_KEY` / `API_KEY` 键时会被读取。

## 认证与鉴权
- 若客户端请求包含 `Authorization: Bearer <token>`，代理直接转发该 token。
- 否则使用配置的上游 API Key；若为空则尝试匿名访问，部分上游可能拒绝请求。

## API 接口
- `GET /`：返回服务欢迎信息与可用端点列表。
- `GET /health`：健康检查，包含当前上游地址与 wire API。
- `GET /v1/models`：优先透传上游响应；失败时返回根据本地默认模型生成的占位列表。
- `POST /v1/chat/completions`：接受 OpenAI Chat Completions 格式，转换并转发至 Responses API。
  - **流式模式**：传入 `stream=true` 时保持 SSE 并回放 `chat.completion.chunk`。
  - **非流式模式**：即便客户端不请求流式，上游若只支持 SSE 亦会被收集并整合为一次性响应。

## 请求示例
```bash
curl -X POST http://127.0.0.1:8010/v1/chat/completions ^
  -H "Authorization: Bearer <client-token>" ^
  -H "Content-Type: application/json" ^
  -d "{\"model\":\"gpt-5\",\"messages\":[{\"role\":\"user\",\"content\":\"你好，今天的天气如何？\"}]}"
```

流式模式可使用：
```bash
curl -N -X POST http://127.0.0.1:8010/v1/chat/completions ^
  -H "Authorization: Bearer <client-token>" ^
  -H "Content-Type: application/json" ^
  -d "{\"model\":\"gpt-5\",\"stream\":true,\"messages\":[{\"role\":\"user\",\"content\":\"写一首小诗\"}]}"
```

## 开发提示
- 项目依赖 FastAPI + httpx，可自由扩展更多端点或协议转换逻辑。
- 若需要调试 CORS，可临时设置 `PROXY_CORS_ORIGINS="http://localhost:3000"` 等值。
- 目前仓库未包含测试套件；建议在引入新功能时补充集成测试（如使用 `pytest-asyncio`）。
- 当上游返回非标准结构时，`_bridge_responses_stream` 与 `responses_to_chat` 提供了多重兜底逻辑，可在此基础上做适配。

## 故障排查
- **502 / proxy-error**：通常为上游不可达或认证失败，检查基础地址与 API Key。
- **空响应或文本缺失**：确认上游返回的事件类型是否含 `delta` 或 `text` 字段，必要时在转换函数中增加适配。
- **CORS 拒绝**：为浏览器客户端显式设置 `PROXY_CORS_ORIGINS`。

