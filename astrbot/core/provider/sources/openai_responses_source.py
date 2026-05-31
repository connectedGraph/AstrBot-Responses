import copy
import json
import os
import random
import time
from collections.abc import AsyncGenerator
from typing import Any, Literal

import httpx

import astrbot.core.message.components as Comp
from astrbot import logger
from astrbot.core.agent.tool import ToolSet
from astrbot.core.message.message_event_result import MessageChain
from astrbot.core.provider.entities import LLMResponse, ToolCallsResult

from ..register import register_provider_adapter
from .openai_source import ProviderOpenAIOfficial


@register_provider_adapter(
    "openai_responses",
    "OpenAI Responses API 提供商适配器",
    default_config_tmpl={
        "id": "openai_responses",
        "provider": "openai",
        "type": "openai_responses",
        "provider_type": "chat_completion",
        "enable": True,
        "key": [],
        "api_base": "https://api.openai.com/v1",
        "timeout": 120,
        "model_config": {"model": "gpt-4.1-mini", "temperature": 0.4},
        "custom_extra_body": {},
        "custom_headers": {},
        "modalities": ["text", "image", "tool_use"],
        "hint": "OpenAI-compatible /v1/responses provider with built-in web_search and AstrBot/MCP tools.",
    },
    provider_display_name="OpenAI Responses",
)
class ProviderOpenAIResponses(ProviderOpenAIOfficial):
    """OpenAI-compatible Responses API adapter.

    AstrBot's agent loop is built around Chat Completions-style function calls.
    This adapter sends Responses API payloads, keeps native web_search visible in
    the UI, and converts Responses function_call output back into AstrBot tool
    calls so existing local and MCP tools keep working.
    """

    WEB_SEARCH_INCLUDE = "web_search_call.action.sources"
    WEB_SEARCH_DESCRIPTION = "OpenAI built-in web search"
    WEB_SEARCH_SCHEMA: dict[str, Any] = {
        "type": "object",
        "properties": {
            "type": {
                "type": "string",
                "description": "Observed web search action type.",
            },
            "queries": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Search queries submitted by the model.",
            },
            "domains": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional domain filters used by the search action.",
            },
            "sources": {
                "type": "array",
                "items": {"type": "object"},
                "description": "Sources returned by the provider when available.",
            },
        },
        "additionalProperties": True,
    }

    def __init__(
        self,
        provider_config,
        provider_settings,
        default_persona=None,
    ) -> None:
        super().__init__(provider_config, provider_settings, default_persona)
        api_base = str(provider_config.get("api_base", "") or "").rstrip("/")
        self.responses_url = f"{api_base}/responses"
        self.models_url = f"{api_base}/models"
        self.headers = self._build_headers(provider_config)
        self.http_client = httpx.AsyncClient(
            proxy=provider_config.get("proxy") or os.environ.get("http_proxy") or None,
            headers=self.headers,
            timeout=self.timeout,
        )

    @staticmethod
    def _build_headers(provider_config: dict) -> dict[str, str]:
        headers: dict[str, str] = {
            "User-Agent": "AstrBot/ResponsesAdapter",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        custom_headers = provider_config.get("custom_headers", {})
        if isinstance(custom_headers, dict):
            headers.update({str(k): str(v) for k, v in custom_headers.items()})
        return headers

    def set_key(self, key):
        self.chosen_api_key = key
        self.client.api_key = key

    def get_current_key(self) -> str:
        return str(self.chosen_api_key or self.client.api_key or "")

    def _headers_with_auth(self, api_key: str) -> dict[str, str]:
        return {**self.headers, "Authorization": f"Bearer {api_key}"}

    async def get_models(self):
        try:
            resp = await self.http_client.get(
                self.models_url,
                headers=self._headers_with_auth(self.get_current_key()),
            )
            resp.raise_for_status()
            payload = resp.json()
            return sorted(
                str(item["id"])
                for item in payload.get("data", [])
                if isinstance(item, dict) and item.get("id")
            )
        except httpx.HTTPStatusError as e:
            raise Exception(f"获取模型列表失败：{e.response.text}") from e
        except Exception as e:
            raise Exception(f"获取模型列表失败：{e}") from e

    async def _prepare_responses_payload(
        self,
        prompt: str | None,
        image_urls: list[str] | None = None,
        contexts: list[dict] | None = None,
        system_prompt: str | None = None,
        tool_calls_result: ToolCallsResult | list[ToolCallsResult] | None = None,
        model: str | None = None,
        func_tool: ToolSet | None = None,
        tool_choice: Literal["auto", "required"] = "auto",
        **kwargs,
    ) -> tuple[dict, list[dict]]:
        payloads, context_query = await self._prepare_chat_payload(
            prompt,
            image_urls,
            contexts,
            system_prompt,
            tool_calls_result,
            model=model,
            **kwargs,
        )
        payload = self._chat_payload_to_responses_payload(
            payloads,
            func_tool=func_tool,
            tool_choice=tool_choice,
        )
        return payload, context_query

    def _chat_payload_to_responses_payload(
        self,
        payloads: dict,
        *,
        func_tool: ToolSet | None = None,
        tool_choice: Literal["auto", "required"] = "auto",
    ) -> dict:
        payload = {"model": payloads.get("model") or self.get_model()}
        payload["input"] = self._messages_to_responses_input(
            payloads.get("messages", []),
        )

        for key, value in payloads.items():
            if key in {"messages", "model", "tools", "tool_choice", "stream_options"}:
                continue
            payload[key] = value

        custom_extra_body = self.provider_config.get("custom_extra_body", {})
        if isinstance(custom_extra_body, dict):
            for key, value in custom_extra_body.items():
                if key in {"tools", "tool_choice", "stream", "stream_options"}:
                    continue
                if key == "max_tokens":
                    payload.setdefault("max_output_tokens", value)
                    continue
                payload[key] = value

        payload["tools"] = self._responses_tools(custom_extra_body, func_tool)
        if func_tool and not func_tool.empty():
            payload["tool_choice"] = tool_choice
        payload["include"] = self._responses_include(payload.get("include"))
        payload["reasoning"] = self._responses_reasoning(payload.get("reasoning"))
        return self._drop_none_values(payload)

    @staticmethod
    def _drop_none_values(payload: dict) -> dict:
        return {key: value for key, value in payload.items() if value is not None}

    @classmethod
    def _responses_tools(
        cls,
        custom_extra_body: Any,
        func_tool: ToolSet | None = None,
    ) -> list[dict]:
        tools: list[dict] = []
        raw_tools = None
        if isinstance(custom_extra_body, dict):
            raw_tools = custom_extra_body.get("responses_tools")

        if isinstance(raw_tools, str):
            try:
                raw_tools = json.loads(raw_tools)
            except json.JSONDecodeError:
                raw_tools = None
        if isinstance(raw_tools, list):
            tools.extend(item for item in raw_tools if isinstance(item, dict))

        if func_tool and not func_tool.empty():
            tools.extend(cls._responses_function_tools(func_tool))

        if not any(tool.get("type") == "web_search" for tool in tools):
            tools.insert(0, {"type": "web_search"})
        return tools

    @staticmethod
    def _responses_function_tools(func_tool: ToolSet) -> list[dict]:
        converted: list[dict] = []
        for tool in func_tool.openai_schema():
            if not isinstance(tool, dict) or tool.get("type") != "function":
                continue
            function = tool.get("function")
            if not isinstance(function, dict) or not function.get("name"):
                continue
            responses_tool: dict[str, Any] = {
                "type": "function",
                "name": function["name"],
            }
            for key in ("description", "parameters", "strict"):
                if key in function:
                    responses_tool[key] = function[key]
            converted.append(responses_tool)
        return converted

    @classmethod
    def _responses_include(cls, current_include: Any) -> list[str]:
        include: list[str] = []
        if isinstance(current_include, str):
            include.append(current_include)
        elif isinstance(current_include, list):
            include.extend(str(item) for item in current_include if item)
        if cls.WEB_SEARCH_INCLUDE not in include:
            include.append(cls.WEB_SEARCH_INCLUDE)
        return include

    @staticmethod
    def _responses_reasoning(current_reasoning: Any) -> dict:
        if isinstance(current_reasoning, dict):
            reasoning = copy.deepcopy(current_reasoning)
        else:
            reasoning = {}
        reasoning.setdefault("summary", "auto")
        return reasoning

    @staticmethod
    def _is_unsupported_responses_option_error(exc: httpx.HTTPStatusError) -> bool:
        if exc.response.status_code not in {400, 422}:
            return False
        body = exc.response.text.lower()
        unsupported_markers = (
            "unknown parameter",
            "unsupported parameter",
            "unrecognized request argument",
            "invalid include",
            "reasoning.summary",
            "web_search_call.action.sources",
        )
        option_markers = (
            "include",
            "reasoning",
            "summary",
            "web_search_call.action.sources",
        )
        return (
            any(marker in body for marker in unsupported_markers)
            and any(marker in body for marker in option_markers)
        ) or (
            any(marker in body for marker in ("invalid", "unsupported", "unknown"))
            and any(marker in body for marker in option_markers)
        )

    @staticmethod
    def _fallback_payload_without_optional_trace_fields(payload: dict) -> dict:
        fallback = copy.deepcopy(payload)
        fallback.pop("include", None)
        reasoning = fallback.get("reasoning")
        if isinstance(reasoning, dict):
            reasoning = copy.deepcopy(reasoning)
            reasoning.pop("summary", None)
            if reasoning:
                fallback["reasoning"] = reasoning
            else:
                fallback.pop("reasoning", None)
        return fallback

    def _messages_to_responses_input(self, messages: list[Any]) -> list[dict]:
        responses_messages: list[dict] = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role") or "user")
            if role == "tool":
                tool_output = self._tool_message_to_function_call_output(message)
                if tool_output:
                    responses_messages.append(tool_output)
                    continue
                content = message.get("content", "")
                responses_messages.append(
                    {
                        "role": "user",
                        "content": self._parts_to_input_content(
                            f"[Tool result]\n{content}",
                            target="input",
                        ),
                    },
                )
                continue
            if role == "assistant" and isinstance(message.get("tool_calls"), list):
                responses_messages.extend(
                    self._assistant_tool_calls_to_function_call_items(
                        message.get("tool_calls", []),
                    )
                )
                content = self._parts_to_input_content(
                    message.get("content", ""),
                    target="output",
                )
                if content:
                    responses_messages.append({"role": "assistant", "content": content})
                continue
            if role == "system":
                role = "developer"
            if role not in {"user", "assistant", "developer"}:
                role = "user"
            content_target = "output" if role == "assistant" else "input"
            content = self._parts_to_input_content(
                message.get("content", ""),
                target=content_target,
            )
            if content:
                responses_messages.append({"role": role, "content": content})
        return responses_messages

    @staticmethod
    def _assistant_tool_calls_to_function_call_items(tool_calls: list[Any]) -> list[dict]:
        items: list[dict] = []
        for tool_call in tool_calls:
            if hasattr(tool_call, "model_dump"):
                tool_call = tool_call.model_dump()
            if not isinstance(tool_call, dict) or tool_call.get("type") != "function":
                continue
            function = tool_call.get("function")
            if not isinstance(function, dict):
                continue
            name = function.get("name")
            if not name:
                continue
            arguments = function.get("arguments")
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments or {}, ensure_ascii=False)
            items.append(
                {
                    "type": "function_call",
                    "call_id": str(tool_call.get("id") or ""),
                    "name": str(name),
                    "arguments": arguments,
                }
            )
        return items

    @staticmethod
    def _tool_message_to_function_call_output(message: dict) -> dict | None:
        call_id = message.get("tool_call_id")
        if not call_id:
            return None
        content = message.get("content", "")
        if isinstance(content, list):
            output = json.dumps(content, ensure_ascii=False)
        elif isinstance(content, str):
            output = content
        else:
            output = json.dumps(content, ensure_ascii=False)
        return {
            "type": "function_call_output",
            "call_id": str(call_id),
            "output": output,
        }

    def _parts_to_input_content(self, content: Any, *, target: str) -> list[dict]:
        text_type = "output_text" if target == "output" else "input_text"
        if content is None:
            return []
        if isinstance(content, str):
            return [{"type": text_type, "text": content}] if content else []
        if isinstance(content, list):
            converted: list[dict] = []
            for part in content:
                if not isinstance(part, dict):
                    if part:
                        converted.append({"type": text_type, "text": str(part)})
                    continue
                part_type = part.get("type")
                if part_type in {"input_text", "output_text"}:
                    converted.append(part)
                elif part_type == "text":
                    converted.append(
                        {"type": text_type, "text": str(part.get("text", ""))},
                    )
                elif part_type == "image_url" and target == "input":
                    image_url = part.get("image_url", {})
                    if isinstance(image_url, dict) and image_url.get("url"):
                        image_part: dict[str, Any] = {
                            "type": "input_image",
                            "image_url": image_url["url"],
                        }
                        if image_url.get("detail"):
                            image_part["detail"] = image_url["detail"]
                        converted.append(image_part)
                elif part_type == "input_image" and target == "input":
                    converted.append(part)
                elif part.get("text"):
                    converted.append(
                        {"type": text_type, "text": str(part.get("text", ""))},
                    )
            return converted
        return [{"type": text_type, "text": str(content)}]

    def _trace_chain_for_tool_call(self, tool_call: dict) -> MessageChain:
        return MessageChain(
            type="tool_call",
            chain=[
                Comp.Json(data=tool_call),
                Comp.Plain(f"🔎 {tool_call.get('name', 'tool')}"),
            ],
        )

    def _trace_chain_for_tool_call_result(
        self,
        tool_call_id: str,
        result: dict,
        *,
        ts: float | None = None,
    ) -> MessageChain:
        return MessageChain(
            type="tool_call_result",
            chain=[
                Comp.Json(
                    data={
                        "id": tool_call_id,
                        "name": result.get("name") or result.get("tool_name"),
                        "status": result.get("status"),
                        "ts": ts or time.time(),
                        "result": json.dumps(result, ensure_ascii=False),
                        "description": result.get("description"),
                        "schema": result.get("schema"),
                    },
                ),
                Comp.Plain(json.dumps(result, ensure_ascii=False)),
            ],
        )

    @staticmethod
    def _trace_chain_for_reasoning(text: str) -> MessageChain | None:
        if not text:
            return None
        return MessageChain(type="reasoning").message(text)

    def _web_search_tool_payload(
        self,
        item: dict,
        *,
        status: str | None = None,
        ts: float | None = None,
    ) -> dict:
        action = item.get("action")
        if not isinstance(action, dict):
            action = {}
        return {
            "id": str(item.get("id") or item.get("call_id") or "web_search"),
            "name": "web_search",
            "args": action,
            "ts": ts or time.time(),
            "status": status or item.get("status") or "in_progress",
            "description": self.WEB_SEARCH_DESCRIPTION,
            "schema": self.WEB_SEARCH_SCHEMA,
        }

    @staticmethod
    def _extract_web_search_sources(item: dict) -> list:
        action = item.get("action")
        if isinstance(action, dict) and isinstance(action.get("sources"), list):
            return action["sources"]
        if isinstance(item.get("sources"), list):
            return item["sources"]
        return []

    def _web_search_result_payload(
        self,
        item: dict,
        *,
        response: dict | None = None,
    ) -> dict:
        action = item.get("action")
        if not isinstance(action, dict):
            action = {}
        result: dict[str, Any] = {
            "name": "web_search",
            "status": item.get("status") or "completed",
            "action": action,
            "sources": self._extract_web_search_sources(item),
            "description": self.WEB_SEARCH_DESCRIPTION,
            "schema": self.WEB_SEARCH_SCHEMA,
        }
        if isinstance(response, dict):
            web_search = response.get("web_search")
            if isinstance(web_search, dict) and "num_requests" in web_search:
                result["num_requests"] = web_search.get("num_requests")
        return result

    def _extract_reasoning_summary_texts(self, value: Any) -> list[str]:
        texts: list[str] = []
        if isinstance(value, str):
            if value:
                texts.append(value)
            return texts
        if isinstance(value, list):
            for item in value:
                texts.extend(self._extract_reasoning_summary_texts(item))
            return texts
        if isinstance(value, dict):
            for key in ("text", "summary_text"):
                text = value.get(key)
                if isinstance(text, str) and text:
                    texts.append(text)
            if "summary" in value:
                texts.extend(self._extract_reasoning_summary_texts(value["summary"]))
            if "content" in value:
                texts.extend(self._extract_reasoning_summary_texts(value["content"]))
        return texts

    def _trace_chains_from_response_output(self, response: dict) -> list[MessageChain]:
        output = response.get("output", [])
        if not isinstance(output, list):
            return []

        trace_chains: list[MessageChain] = []
        for item in output:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type == "reasoning":
                for text in self._extract_reasoning_summary_texts(item.get("summary")):
                    chain = self._trace_chain_for_reasoning(text)
                    if chain:
                        trace_chains.append(chain)
                continue
            if item_type == "web_search_call":
                tool_call = self._web_search_tool_payload(
                    item,
                    status=item.get("status") or "completed",
                )
                trace_chains.append(self._trace_chain_for_tool_call(tool_call))
                trace_chains.append(
                    self._trace_chain_for_tool_call_result(
                        tool_call["id"],
                        self._web_search_result_payload(item, response=response),
                    )
                )
        return trace_chains

    @staticmethod
    def _stream_event_item(event: dict) -> dict:
        item = event.get("item")
        return item if isinstance(item, dict) else {}

    @staticmethod
    def _stream_event_item_id(event: dict, item: dict | None = None) -> str:
        source = item if item is not None else {}
        for key in ("id", "item_id", "output_item_id", "call_id"):
            value = source.get(key) or event.get(key)
            if value:
                return str(value)
        return "web_search"

    @staticmethod
    def _reasoning_summary_event_key(event: dict) -> str:
        parts: list[str] = []
        for key in ("item_id", "output_index", "summary_index"):
            if event.get(key) is not None:
                parts.append(str(event[key]))
        return ":".join(parts) or "default"

    def _trace_chains_from_stream_event(
        self,
        event: dict,
        web_search_calls: dict[str, dict],
        emitted_completed_web_search_ids: set[str],
        reasoning_summary_keys_with_delta: set[str],
    ) -> list[MessageChain]:
        event_type = str(event.get("type") or "")
        trace_chains: list[MessageChain] = []

        if event_type == "response.reasoning_summary_text.delta":
            delta = event.get("delta")
            if isinstance(delta, str) and delta:
                reasoning_summary_keys_with_delta.add(
                    self._reasoning_summary_event_key(event)
                )
                chain = self._trace_chain_for_reasoning(delta)
                if chain:
                    trace_chains.append(chain)
            return trace_chains

        if event_type == "response.reasoning_summary_text.done":
            summary_key = self._reasoning_summary_event_key(event)
            if summary_key in reasoning_summary_keys_with_delta:
                return trace_chains
            for key in ("text", "summary_text"):
                text = event.get(key)
                if isinstance(text, str) and text:
                    chain = self._trace_chain_for_reasoning(text)
                    if chain:
                        trace_chains.append(chain)
                    break
            return trace_chains

        item = self._stream_event_item(event)
        item_type = item.get("type")
        if event_type == "response.output_item.done" and item_type == "reasoning":
            if reasoning_summary_keys_with_delta:
                return trace_chains
            for text in self._extract_reasoning_summary_texts(item.get("summary")):
                chain = self._trace_chain_for_reasoning(text)
                if chain:
                    trace_chains.append(chain)
            return trace_chains

        if event_type == "response.output_item.added" and item_type == "web_search_call":
            tool_call = self._web_search_tool_payload(
                item,
                status=item.get("status") or "in_progress",
            )
            web_search_calls[tool_call["id"]] = tool_call
            trace_chains.append(self._trace_chain_for_tool_call(tool_call))
            return trace_chains

        if event_type in {
            "response.web_search_call.in_progress",
            "response.web_search_call.searching",
            "response.web_search_call.completed",
        }:
            tool_call_id = self._stream_event_item_id(event)
            existing = web_search_calls.get(tool_call_id)
            if not existing:
                existing = self._web_search_tool_payload(
                    {"id": tool_call_id, "type": "web_search_call"},
                )
            existing = copy.deepcopy(existing)
            existing["status"] = event_type.removeprefix("response.web_search_call.")
            existing["ts"] = existing.get("ts") or time.time()
            web_search_calls[tool_call_id] = existing
            trace_chains.append(self._trace_chain_for_tool_call(existing))
            return trace_chains

        if event_type == "response.output_item.done" and item_type == "web_search_call":
            tool_call = self._web_search_tool_payload(
                item,
                status=item.get("status") or "completed",
            )
            if tool_call["id"] in web_search_calls:
                tool_call["ts"] = web_search_calls[tool_call["id"]].get(
                    "ts",
                    tool_call["ts"],
                )
            web_search_calls[tool_call["id"]] = tool_call
            trace_chains.append(self._trace_chain_for_tool_call(tool_call))
            trace_chains.append(
                self._trace_chain_for_tool_call_result(
                    tool_call["id"],
                    self._web_search_result_payload(item),
                )
            )
            emitted_completed_web_search_ids.add(tool_call["id"])
            return trace_chains

        return trace_chains

    @staticmethod
    def _trace_chain_tool_call_id(chain: MessageChain) -> str | None:
        if chain.type not in {"tool_call", "tool_call_result"}:
            return None
        for comp in chain.chain:
            if isinstance(comp, Comp.Json):
                data = comp.data
                if isinstance(data, str):
                    try:
                        data = json.loads(data)
                    except json.JSONDecodeError:
                        return None
                if isinstance(data, dict):
                    tool_call_id = data.get("id")
                    return str(tool_call_id) if tool_call_id else None
        return None

    @staticmethod
    def _payload_has_optional_trace_fields(payload: dict) -> bool:
        reasoning = payload.get("reasoning")
        return bool(
            payload.get("include")
            or (isinstance(reasoning, dict) and "summary" in reasoning)
        )

    async def _post_responses(
        self,
        payload: dict,
        *,
        stream: bool,
        api_key: str,
    ) -> dict | AsyncGenerator[dict, None]:
        request_payload = copy.deepcopy(payload)
        request_payload["stream"] = stream
        start = time.perf_counter()
        tools = request_payload.get("tools")
        tool_names = []
        if isinstance(tools, list):
            for tool in tools:
                if isinstance(tool, dict):
                    tool_names.append(str(tool.get("name") or tool.get("type") or "?"))
        logger.info(
            "[OpenAI Responses] POST %s model=%s stream=%s tools=%s tool_names=%s",
            self.responses_url,
            request_payload.get("model"),
            stream,
            len(tools) if isinstance(tools, list) else tools,
            tool_names,
        )

        headers = self._headers_with_auth(api_key)
        if not stream:
            resp = await self.http_client.post(
                self.responses_url,
                headers=headers,
                json=request_payload,
                timeout=self.timeout,
            )
            elapsed = time.perf_counter() - start
            logger.info(
                "[OpenAI Responses] status=%s elapsed=%.2fs",
                resp.status_code,
                elapsed,
            )
            if resp.status_code >= 400:
                logger.error("[OpenAI Responses] error body: %s", resp.text[:4096])
            resp.raise_for_status()
            return resp.json()

        async def _iter_events() -> AsyncGenerator[dict, None]:
            async with self.http_client.stream(
                "POST",
                self.responses_url,
                headers=headers,
                json=request_payload,
                timeout=self.timeout,
            ) as resp:
                elapsed = time.perf_counter() - start
                logger.info(
                    "[OpenAI Responses] stream status=%s connected=%.2fs",
                    resp.status_code,
                    elapsed,
                )
                if resp.status_code >= 400:
                    body = await resp.aread()
                    logger.error(
                        "[OpenAI Responses] stream error body: %s",
                        body.decode("utf-8", errors="replace")[:4096],
                    )
                    resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data = line.removeprefix("data:").strip()
                    if data == "[DONE]":
                        break
                    try:
                        yield json.loads(data)
                    except json.JSONDecodeError:
                        logger.debug("[OpenAI Responses] skipped SSE line: %s", data)

        return _iter_events()

    async def _query_responses(self, payload: dict, api_key: str) -> LLMResponse:
        try:
            raw = await self._post_responses(payload, stream=False, api_key=api_key)
        except httpx.HTTPStatusError as e:
            if not self._is_unsupported_responses_option_error(e):
                raise
            fallback_payload = self._fallback_payload_without_optional_trace_fields(
                payload
            )
            logger.warning(
                "[OpenAI Responses] include/reasoning.summary unsupported; retrying once without optional trace fields."
            )
            raw = await self._post_responses(
                fallback_payload,
                stream=False,
                api_key=api_key,
            )
        assert isinstance(raw, dict)
        return self._parse_responses_response(raw)

    async def _iter_stream_events_with_optional_trace_fallback(
        self,
        payload: dict,
        api_key: str,
    ) -> AsyncGenerator[dict, None]:
        event_iter = await self._post_responses(payload, stream=True, api_key=api_key)
        assert not isinstance(event_iter, dict)
        try:
            async for event in event_iter:
                yield event
        except httpx.HTTPStatusError as e:
            if not (
                self._payload_has_optional_trace_fields(payload)
                and self._is_unsupported_responses_option_error(e)
            ):
                raise
            fallback_payload = self._fallback_payload_without_optional_trace_fields(
                payload
            )
            logger.warning(
                "[OpenAI Responses] include/reasoning.summary unsupported; retrying stream once without optional trace fields."
            )
            fallback_event_iter = await self._post_responses(
                fallback_payload,
                stream=True,
                api_key=api_key,
            )
            assert not isinstance(fallback_event_iter, dict)
            async for event in fallback_event_iter:
                yield event

    async def _query_responses_stream(
        self,
        payload: dict,
        api_key: str,
    ) -> AsyncGenerator[LLMResponse, None]:
        final_response: dict | None = None
        full_text_parts: list[str] = []
        web_search_calls: dict[str, dict] = {}
        emitted_completed_web_search_ids: set[str] = set()
        reasoning_summary_keys_with_delta: set[str] = set()
        final_trace_chains: list[MessageChain] = []
        stream_output_items: list[dict] = []

        async for event in self._iter_stream_events_with_optional_trace_fallback(
            payload,
            api_key,
        ):
            event_type = event.get("type")
            item = event.get("item")
            if (
                event_type == "response.output_item.done"
                and isinstance(item, dict)
                and item.get("type") == "function_call"
            ):
                stream_output_items.append(item)

            trace_chains = self._trace_chains_from_stream_event(
                event,
                web_search_calls,
                emitted_completed_web_search_ids,
                reasoning_summary_keys_with_delta,
            )
            if trace_chains:
                yield LLMResponse(
                    "assistant",
                    trace_chains=trace_chains,
                    is_chunk=True,
                )

            delta = self._extract_stream_delta(event)
            if delta:
                full_text_parts.append(delta)
                yield LLMResponse(
                    "assistant",
                    result_chain=MessageChain(chain=[Comp.Plain(delta)]),
                    is_chunk=True,
                )

            if event_type == "response.completed" and isinstance(
                event.get("response"),
                dict,
            ):
                final_response = event["response"]
                if stream_output_items and not final_response.get("output"):
                    final_response["output"] = stream_output_items
                final_trace_chains = self._trace_chains_from_response_output(
                    final_response
                )

        if emitted_completed_web_search_ids and final_trace_chains:
            final_trace_chains = [
                chain
                for chain in final_trace_chains
                if self._trace_chain_tool_call_id(chain)
                not in emitted_completed_web_search_ids
            ]

        if final_response is not None:
            final_text = self._extract_response_text(final_response)
            if self._extract_function_calls(final_response):
                final_llm_response = self._parse_responses_response(
                    final_response,
                    include_trace_chains=False,
                )
                if full_text_parts:
                    final_llm_response.result_chain = None
                final_llm_response.trace_chains = final_trace_chains
                yield final_llm_response
                return
            if final_text and not full_text_parts:
                final_llm_response = self._parse_responses_response(
                    final_response,
                    include_trace_chains=False,
                )
                final_llm_response.trace_chains = final_trace_chains
                yield final_llm_response
                return
            text = "".join(full_text_parts)
            if text:
                yield LLMResponse(
                    "assistant",
                    result_chain=MessageChain(chain=[Comp.Plain(text)]),
                    trace_chains=final_trace_chains,
                    raw_completion=final_response,  # type: ignore[arg-type]
                )
                return
            final_llm_response = self._parse_responses_response(
                final_response,
                include_trace_chains=False,
            )
            final_llm_response.trace_chains = final_trace_chains
            yield final_llm_response
            return

        text = "".join(full_text_parts)
        if not text:
            raise Exception("OpenAI Responses stream has no usable output.")
        yield LLMResponse(
            "assistant",
            result_chain=MessageChain(chain=[Comp.Plain(text)]),
        )

    @staticmethod
    def _extract_stream_delta(event: dict) -> str:
        event_type = str(event.get("type") or "")
        if event_type == "response.output_text.delta":
            delta = event.get("delta")
            if isinstance(delta, str):
                return delta
        return ""

    def _parse_responses_response(
        self,
        response: dict,
        *,
        include_trace_chains: bool = True,
    ) -> LLMResponse:
        function_calls = self._extract_function_calls(response)
        if function_calls:
            text = self._extract_response_text(response)
            result_chain = MessageChain(chain=[Comp.Plain(text)]) if text else None
            return LLMResponse(
                "tool",
                result_chain=result_chain,
                trace_chains=self._trace_chains_from_response_output(response)
                if include_trace_chains
                else [],
                tools_call_args=[call["arguments"] for call in function_calls],
                tools_call_name=[call["name"] for call in function_calls],
                tools_call_ids=[call["call_id"] for call in function_calls],
                raw_completion=response,  # type: ignore[arg-type]
            )

        text = self._extract_response_text(response)
        if not text:
            logger.error("OpenAI Responses returned no usable output: %s", response)
            raise Exception(
                f"OpenAI Responses returned no usable output. response_id={response.get('id')}",
            )
        return LLMResponse(
            "assistant",
            result_chain=MessageChain(chain=[Comp.Plain(text)]),
            trace_chains=self._trace_chains_from_response_output(response)
            if include_trace_chains
            else [],
            raw_completion=response,  # type: ignore[arg-type]
        )

    def _extract_function_calls(self, response: dict) -> list[dict[str, Any]]:
        output = response.get("output", [])
        if not isinstance(output, list):
            return []

        function_calls: list[dict[str, Any]] = []
        for item in output:
            if not isinstance(item, dict) or item.get("type") != "function_call":
                continue
            name = item.get("name")
            if not name:
                continue
            raw_arguments = item.get("arguments")
            if isinstance(raw_arguments, str):
                try:
                    arguments = json.loads(raw_arguments or "{}")
                except json.JSONDecodeError as e:
                    logger.error(
                        "[OpenAI Responses] failed to parse function arguments: %s", e
                    )
                    arguments = {}
            elif isinstance(raw_arguments, dict):
                arguments = raw_arguments
            else:
                arguments = {}
            function_calls.append(
                {
                    "name": str(name),
                    "arguments": arguments or {},
                    "call_id": str(
                        item.get("call_id")
                        or item.get("id")
                        or f"call_{len(function_calls)}"
                    ),
                }
            )
        return function_calls

    def _extract_response_text(self, response: dict) -> str:
        output_text = response.get("output_text")
        if isinstance(output_text, str) and output_text:
            return output_text

        parts: list[str] = []
        output = response.get("output", [])
        if not isinstance(output, list):
            return ""
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content", [])
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") in {"output_text", "text", "input_text"}:
                    text = part.get("text")
                    if isinstance(text, str):
                        parts.append(text)
        return "".join(parts)

    async def text_chat(
        self,
        prompt,
        session_id=None,
        image_urls=None,
        func_tool=None,
        contexts=None,
        system_prompt=None,
        tool_calls_result=None,
        model=None,
        tool_choice: Literal["auto", "required"] = "auto",
        **kwargs,
    ) -> LLMResponse:
        payloads, _ = await self._prepare_responses_payload(
            prompt,
            image_urls,
            contexts,
            system_prompt,
            tool_calls_result,
            model=model,
            func_tool=func_tool,
            tool_choice=tool_choice,
            **kwargs,
        )

        max_retries = 3
        available_api_keys = self.api_keys.copy()
        if not available_api_keys:
            raise Exception("OpenAI Responses provider missing API key.")
        last_exception = None
        for retry_cnt in range(max_retries):
            chosen_key = random.choice(available_api_keys)
            try:
                self.set_key(chosen_key)
                return await self._query_responses(payloads, chosen_key)
            except httpx.HTTPStatusError as e:
                last_exception = e
                if e.response.status_code == 429 and len(available_api_keys) > 1:
                    available_api_keys.remove(chosen_key)
                    continue
                raise
            except Exception as e:
                last_exception = e
                if retry_cnt == max_retries - 1:
                    raise
        raise last_exception or Exception("未知错误")

    async def text_chat_stream(
        self,
        prompt,
        session_id=None,
        image_urls=None,
        func_tool=None,
        contexts=None,
        system_prompt=None,
        tool_calls_result=None,
        model=None,
        tool_choice: Literal["auto", "required"] = "auto",
        **kwargs,
    ) -> AsyncGenerator[LLMResponse, None]:
        payloads, _ = await self._prepare_responses_payload(
            prompt,
            image_urls,
            contexts,
            system_prompt,
            tool_calls_result,
            model=model,
            func_tool=func_tool,
            tool_choice=tool_choice,
            **kwargs,
        )

        max_retries = 3
        available_api_keys = self.api_keys.copy()
        if not available_api_keys:
            raise Exception("OpenAI Responses provider missing API key.")
        last_exception = None
        for retry_cnt in range(max_retries):
            chosen_key = random.choice(available_api_keys)
            try:
                self.set_key(chosen_key)
                async for response in self._query_responses_stream(
                    payloads,
                    chosen_key,
                ):
                    yield response
                return
            except httpx.HTTPStatusError as e:
                last_exception = e
                if e.response.status_code == 429 and len(available_api_keys) > 1:
                    available_api_keys.remove(chosen_key)
                    continue
                raise
            except Exception as e:
                last_exception = e
                if retry_cnt == max_retries - 1:
                    raise
        raise last_exception or Exception("未知错误")

    async def terminate(self):
        await self.http_client.aclose()
