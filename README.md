# AstrBot-Responses

This is a personal fork of [AstrBotDevs/AstrBot](https://github.com/AstrBotDevs/AstrBot), based on upstream tag `v4.3.1`.

This fork focuses on OpenAI-compatible `/v1/responses` support, model-native web search, and visible tool/reasoning traces in AstrBot WebUI. The upstream project README, full deployment docs, and general project introduction are intentionally not duplicated here. Read the original upstream README here:

[Upstream AstrBot README](https://github.com/AstrBotDevs/AstrBot/blob/v4.3.1/README.md)

## Fork Changes

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

## What This Fork Does Not Do

- It does not expose hidden raw chain-of-thought.
- It does not replace AstrBot's existing tool loop.
- It does not open or target an upstream pull request.
- It keeps this README focused on this fork's Responses changes.

## Provider Setup

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

## Built-In Web Search

The `web_search` tool used here is model-native Responses API web search. It is different from external AstrBot search plugins such as Bing, Tavily, or other plugin-executed search providers.

This fork sends the web search tool directly to the model:

```json
{"type": "web_search"}
```

When the upstream returns `web_search_call` events, AstrBot WebUI shows them as expandable tool cards.

## Trace Display

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

## Abbreviations

- `MCP`: Model Context Protocol. AstrBot can use it to attach external tools.
- `CoT`: chain of thought. This fork only displays API-returned summaries, not hidden raw reasoning.
- `SSE`: Server-Sent Events, the HTTP streaming format used by Responses and WebChat streaming.
- `Responses`: OpenAI's `/v1/responses` API shape for multimodal, tool-using model responses.
- `tool_call`: the model requested a tool.
- `tool_call_result`: the tool execution result returned to the model or displayed in the UI.

## Development Notes

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

## Upstream

This fork remains derived from AstrBot. For the original project description, installation guide, platform matrix, and upstream contribution policy, use:

[https://github.com/AstrBotDevs/AstrBot](https://github.com/AstrBotDevs/AstrBot)
