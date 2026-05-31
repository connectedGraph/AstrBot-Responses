# AstrBot-Responses

## 中文

这是一个个人维护的 [AstrBotDevs/AstrBot](https://github.com/AstrBotDevs/AstrBot) fork，基于上游 `v4.3.1`。

这个 fork 主要面向 OpenAI-compatible `/v1/responses` 接口、模型内置联网搜索，以及 AstrBot WebUI 中可见的工具/思考摘要时间线。上游 README、完整部署文档和通用项目介绍不在这里重复。原始上游 README 见：

[上游 AstrBot README](https://github.com/AstrBotDevs/AstrBot/blob/v4.3.1/README.md)

### Fork 改动

1. 新增 `openai_responses` provider，支持 OpenAI-compatible `/v1/responses`。
2. 通过兼容的 `/v1/models` 获取模型列表。
3. 请求中始终带官方内置联网搜索工具 `{"type": "web_search"}`。
4. 将 AstrBot 和 MCP 工具转换为 Responses function tools。
5. 将 Responses `function_call` 输出转换回 AstrBot 现有工具循环格式。
6. 将 Responses 原生 `web_search_call` 事件映射为 WebUI 的 `tool_call` / `tool_call_result` 卡片。
7. 默认请求 `reasoning: {"summary": "auto"}`，并且只展示 API 明确返回的 reasoning summary。
8. 默认请求 `include: ["web_search_call.action.sources"]`，上游支持时可在工具结果中显示搜索来源。
9. 如果上游拒绝 `include` 或 `reasoning.summary`，自动去掉这些可选 trace 字段重试一次。
10. 增强 WebUI 聊天和历史记录展示，工具卡可显示 name、args、result、description、schema、status。

### 不做什么

- 不展示隐藏的 raw chain-of-thought。
- 不替换 AstrBot 现有工具循环。
- 不面向上游提交 PR。
- README 只保留这个 fork 的 Responses 改动说明。

### Provider 配置

可以在 WebUI 中使用 `OpenAI Responses` 模板新增 provider，也可以手动配置：

```json
{
  "id": "openai_responses",
  "provider": "openai",
  "type": "openai_responses",
  "provider_type": "chat_completion",
  "enable": true,
  "key": ["YOUR_API_KEY"],
  "api_base": "https://api.openai.com/v1",
  "timeout": 120,
  "model_config": {
    "model": "gpt-4.1-mini",
    "temperature": 0.4
  },
  "custom_extra_body": {},
  "custom_headers": {},
  "modalities": ["text", "image", "tool_use"]
}
```

provider 会请求：

- `POST {api_base}/responses`
- `GET {api_base}/models`

`api_base` 需要包含 `/v1`，例如 `https://api.openai.com/v1`。

### 内置联网搜索

这里的 `web_search` 是 Responses API 的模型内置联网搜索，不同于 AstrBot 外部搜索插件，例如 Bing、Tavily 或其他由插件自行执行的搜索服务。

这个 fork 会把联网搜索工具直接传给模型：

```json
{"type": "web_search"}
```

当上游返回 `web_search_call` 事件时，AstrBot WebUI 会显示为可展开的工具卡。

### Trace 展示

这个 fork 复用 AstrBot 现有 message-chain 链路，不新增一套前端协议。

Responses 事件映射如下：

| Responses 事件 | WebUI 展示 |
| --- | --- |
| `web_search_call` | `tool_call` / `tool_call_result` 卡片 |
| `function_call` | AstrBot 工具循环执行 |
| `reasoning.summary` | 思考摘要块 |
| `output_text.delta` | 普通流式正文 |

工具卡可以显示：

- name
- args
- result
- description
- schema
- status

### 缩写说明

- `MCP`: Model Context Protocol，AstrBot 可用它接入外部工具。
- `CoT`: chain of thought，思维链。本 fork 只展示 API 返回的摘要，不展示隐藏 raw reasoning。
- `SSE`: Server-Sent Events，Responses 和 WebChat streaming 使用的 HTTP 流式格式。
- `Responses`: OpenAI `/v1/responses` API 形态，用于多模态、工具调用等模型响应。
- `tool_call`: 模型请求调用工具。
- `tool_call_result`: 工具执行结果，返回给模型或展示在 UI 中。

### 开发说明

主要改动文件：

- `astrbot/core/provider/sources/openai_responses_source.py`
- `astrbot/core/provider/entities.py`
- `astrbot/core/agent/runners/tool_loop_agent_runner.py`
- `astrbot/core/pipeline/process_stage/method/llm_request.py`
- `astrbot/core/platform/sources/webchat/webchat_event.py`
- `astrbot/dashboard/routes/chat.py`
- `dashboard/src/components/chat/Chat.vue`
- `dashboard/src/components/chat/MessageList.vue`
- `dashboard/src/views/ConversationPage.vue`

常用检查：

```bash
python -m py_compile astrbot/core/provider/sources/openai_responses_source.py
cd dashboard
npm run build
```

### 上游

本 fork 派生自 AstrBot。原项目介绍、安装指南、平台支持矩阵和上游贡献规则请看：

[https://github.com/AstrBotDevs/AstrBot](https://github.com/AstrBotDevs/AstrBot)

---

## English

This is a personal fork of [AstrBotDevs/AstrBot](https://github.com/AstrBotDevs/AstrBot), based on upstream tag `v4.3.1`.

This fork focuses on OpenAI-compatible `/v1/responses` support, model-native web search, and visible tool/reasoning traces in AstrBot WebUI. The upstream README, full deployment docs, and general project introduction are intentionally not duplicated here. Read the original upstream README here:

[Upstream AstrBot README](https://github.com/AstrBotDevs/AstrBot/blob/v4.3.1/README.md)

### Fork Changes

1. Adds an `openai_responses` provider for OpenAI-compatible `/v1/responses` endpoints.
2. Lists models through the compatible `/v1/models` endpoint.
3. Always includes the official built-in web search tool as `{"type": "web_search"}`.
4. Converts AstrBot and MCP tools into Responses function tools.
5. Converts Responses `function_call` output back into AstrBot's existing tool loop shape.
6. Maps Responses native `web_search_call` events into WebUI `tool_call` and `tool_call_result` cards.
7. Requests reasoning summaries with `reasoning: {"summary": "auto"}` and displays only returned summaries.
8. Requests `include: ["web_search_call.action.sources"]` so search sources can appear in tool results when the upstream supports them.
9. Retries once without optional trace fields if an upstream rejects `include` or `reasoning.summary`.
10. Extends WebUI chat/history rendering so tool cards can show name, args, result, description, schema, and status.

### What This Fork Does Not Do

- It does not expose hidden raw chain-of-thought.
- It does not replace AstrBot's existing tool loop.
- It does not open or target an upstream pull request.
- It keeps this README focused on this fork's Responses changes.

### Provider Setup

Add a provider using the WebUI template named `OpenAI Responses`, or configure one manually:

```json
{
  "id": "openai_responses",
  "provider": "openai",
  "type": "openai_responses",
  "provider_type": "chat_completion",
  "enable": true,
  "key": ["YOUR_API_KEY"],
  "api_base": "https://api.openai.com/v1",
  "timeout": 120,
  "model_config": {
    "model": "gpt-4.1-mini",
    "temperature": 0.4
  },
  "custom_extra_body": {},
  "custom_headers": {},
  "modalities": ["text", "image", "tool_use"]
}
```

The provider sends requests to:

- `POST {api_base}/responses`
- `GET {api_base}/models`

`api_base` should include `/v1`, for example `https://api.openai.com/v1`.

### Built-In Web Search

The `web_search` tool used here is model-native Responses API web search. It is different from external AstrBot search plugins such as Bing, Tavily, or other plugin-executed search providers.

This fork sends the web search tool directly to the model:

```json
{"type": "web_search"}
```

When the upstream returns `web_search_call` events, AstrBot WebUI shows them as expandable tool cards.

### Trace Display

The fork reuses AstrBot's message-chain flow instead of inventing a separate frontend protocol.

Responses events are mapped as follows:

| Responses event | WebUI display |
| --- | --- |
| `web_search_call` | `tool_call` / `tool_call_result` card |
| `function_call` | AstrBot tool loop execution |
| `reasoning.summary` | thinking summary block |
| `output_text.delta` | normal streamed assistant text |

Tool cards can display:

- name
- args
- result
- description
- schema
- status

### Abbreviations

- `MCP`: Model Context Protocol. AstrBot can use it to attach external tools.
- `CoT`: chain of thought. This fork only displays API-returned summaries, not hidden raw reasoning.
- `SSE`: Server-Sent Events, the HTTP streaming format used by Responses and WebChat streaming.
- `Responses`: OpenAI's `/v1/responses` API shape for multimodal, tool-using model responses.
- `tool_call`: the model requested a tool.
- `tool_call_result`: the tool execution result returned to the model or displayed in the UI.

### Development Notes

Main modified areas:

- `astrbot/core/provider/sources/openai_responses_source.py`
- `astrbot/core/provider/entities.py`
- `astrbot/core/agent/runners/tool_loop_agent_runner.py`
- `astrbot/core/pipeline/process_stage/method/llm_request.py`
- `astrbot/core/platform/sources/webchat/webchat_event.py`
- `astrbot/dashboard/routes/chat.py`
- `dashboard/src/components/chat/Chat.vue`
- `dashboard/src/components/chat/MessageList.vue`
- `dashboard/src/views/ConversationPage.vue`

Useful checks:

```bash
python -m py_compile astrbot/core/provider/sources/openai_responses_source.py
cd dashboard
npm run build
```

### Upstream

This fork remains derived from AstrBot. For the original project description, installation guide, platform matrix, and upstream contribution policy, use:

[https://github.com/AstrBotDevs/AstrBot](https://github.com/AstrBotDevs/AstrBot)
