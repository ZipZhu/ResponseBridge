SillyTavern 的 OpenAI 兼容代理（后端为 Responses API）

**项目简介**
- 将 SillyTavern 的 OpenAI 兼容客户端，桥接到仅支持 Responses API 的第三方服务（如 `https://codex.imyaichat.com/openai`）。
- 对外暴露标准的 OpenAI Chat Completions 接口，同时在内部与上游的 Responses API 通信。
- 支持流式与非流式两种用法：
  - 流式：把上游 SSE 事件转换为 OpenAI `chat.completion.chunk` 风格的增量分块。
  - 非流式：消费上游 SSE，聚合为一次性 Chat Completions JSON 响应。

**功能特性**
- OpenAI 兼容端点：`/v1/chat/completions`、`/v1/models`、`/health`、`/`。
- 消息/参数映射：将 Chat Completions 的消息和采样参数映射为 Responses API 的入参。
- 自动适配上游“必须流式”的要求：即使下游非流式，也会对上游强制 `stream: true`，并自动聚合返回。
- 错误透明：上游 4xx/5xx 会以清晰的文本方式在流式模式中回显到前端；非流式下也会整理成清晰的错误信息。
- CORS：默认允许来自 `http://127.0.0.1:8000` 与 `http://localhost:8000` 的跨域（适配 SillyTavern）。
- 配置与密钥：可从 `config.toml` 和 `auth.json` 读取，也支持环境变量覆盖。

**架构与实现**
- 相关技术：`FastAPI`（Web 框架）、`httpx`（HTTP 客户端，含 SSE 流处理）、`Uvicorn`（ASGI 服务器）。
- 主要端点：
  - `GET /`：简单状态提示。
  - `GET /health`：返回代理状态、上游地址、wire_api 等信息。
  - `GET /v1/models`：若上游有此接口则透传；否则返回包含默认模型的简易列表。
  - `POST /v1/chat/completions`：OpenAI Chat Completions 兼容接口。
- 上游路径：
  - Responses API 路径默认使用 `PROXY_RESPONSES_PATH=/responses`，拼接到 `base_url`（例如 `https://codex.imyaichat.com/openai/responses`）。
  - 如你的上游不同，可通过环境变量覆盖。
- 消息映射（Chat → Responses）：
  - 将 Chat messages 映射为 Responses 的 `input` 列表。
  - 角色到内容块类型：
    - `assistant`（历史助手回复）→ `type: "output_text"`
    - 其他（如 `user`、`system`）→ `type: "input_text"`
  - 目前仅桥接文本内容，图片/工具等类型未映射。
- 参数映射：
  - `temperature`、`top_p` 直传；
  - `max_tokens` → `max_output_tokens`；
  - `stop` → `stop_sequences`；
  - 忽略不被上游支持的参数（如 `frequency_penalty`、`presence_penalty`），以避免 400 报错。
- 流式桥接：
  - 对上游强制 `stream: true` 并设置 `Accept: text/event-stream`；
  - 解析上游 SSE：支持常见的 `response.output_text.delta`、`response.completed` 等事件，也能透传已是 OpenAI 风格的分块；
  - 非流式下，消费整段 SSE 内容并拼装为一次性 Chat Completions 响应。
- 认证与 CORS：
  - 优先使用下游请求头 `Authorization: Bearer ...`；若无则读取本地 `auth.json` 或环境变量 `OPENAI_API_KEY`/`UPSTREAM_API_KEY`；
  - 默认允许 SillyTavern 的本地 UI 跨域，支持通过 `PROXY_CORS_ORIGINS` 配置。
- 错误处理：
  - 将上游错误以清晰格式展示（如：`Upstream error 400: {...}`），避免“无声失败”。
  - 处理“必须流式”的要求与不支持的参数，避免常见 400。

**安装与运行**
- 前置要求：Python 3.10+。
- 安装依赖：
  - `pip install -r requirements.txt`
- 运行代理：
  - `python proxy_server.py`
  - 默认监听：`http://127.0.0.1:8010`（启动时会打印 `[proxy] Starting on http://127.0.0.1:8010`）。
- 健康检查：
  - 浏览器打开 `http://127.0.0.1:8010/health`。

**在 SillyTavern 中使用**
- Base URL：`http://127.0.0.1:8010/v1`
- API Key：使用上游服务的密钥（可放在 SillyTavern，也可提前写入 `auth.json`/环境变量）。
- Model：`gpt-5`（或你的上游模型名）。
- 建议开启“流式输出”。

**自测示例（PowerShell）**
- 非流式：
  - `$body = @{ model='gpt-5'; messages=@(@{ role='user'; content='你好' }) }`
  - `$json = $body | ConvertTo-Json -Depth 6`
  - `Invoke-RestMethod -Uri 'http://127.0.0.1:8010/v1/chat/completions' -Method Post -ContentType 'application/json' -Body $json`
- 流式（SSE）：
  - `curl.exe -N -X POST http://127.0.0.1:8010/v1/chat/completions -H "Content-Type: application/json" --data "{\"model\":\"gpt-5\",\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}],\"stream\":true}"`

**可配置项（环境变量）**
- `PROXY_UPSTREAM_BASE_URL`：上游基础地址（可覆盖 `config.toml` 中的 `base_url`）。
- `PROXY_RESPONSES_PATH`：Responses API 路径（默认 `/responses`）。
- `PROXY_DEFAULT_MODEL`：默认模型（可覆盖 `config.toml` 的 `model`）。
- `OPENAI_API_KEY` / `UPSTREAM_API_KEY`：上游鉴权用的密钥（若请求未携带 Authorization 时使用）。
- `PROXY_HOST` / `PROXY_PORT`：监听地址与端口（默认 `127.0.0.1:8010`）。
- `PROXY_CORS_ORIGINS`：允许的跨域来源，逗号分隔。

**常见问题**
- 端口冲突：SillyTavern 占用 `8000`，本代理使用 `8010`。如仍冲突，可临时用环境变量切换：`$env:PROXY_PORT=8011; python .\proxy_server.py`。
- 报错 `Stream 必须设置为 true`：代理已对上游强制流式；若仍出现，重启代理后再试。
- 报错 `Unsupported parameter: frequency_penalty`：代理已忽略不支持的参数；如再遇到类似参数名，请反馈。
- 报错 `Invalid value: 'text'` 或 `Invalid value: 'input_text'`：
  - 代理按角色自动选择 `input_text`（用户/系统）或 `output_text`（助手历史）。若仍报错，请确认已重启代理，并提供完整错误我来适配。
- `/v1/models` 404：上游无该接口时，代理会返回包含默认模型的简易列表，不影响使用。

**文件说明**
- `proxy_server.py:1`：FastAPI 应用，路由、消息/参数映射、SSE 桥接与错误处理。
- `requirements.txt:1`：依赖列表（FastAPI、httpx、uvicorn）。
- `config.toml:1`：读取 `base_url`、`model`、`wire_api`（可被环境变量覆盖）。
- `auth.json:1`：读取 `OPENAI_API_KEY`（或用 `UPSTREAM_API_KEY`）。
- `README_PROXY.md:1`：本文档。

