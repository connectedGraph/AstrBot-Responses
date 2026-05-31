import sys
import traceback
import typing as T
import time
from .base import BaseAgentRunner, AgentResponse, AgentState
from ..hooks import BaseAgentRunHooks
from ..tool_executor import BaseFunctionToolExecutor
from ..run_context import ContextWrapper, TContext
from ..response import AgentResponseData
from astrbot.core.provider.provider import Provider
from astrbot.core.message.message_event_result import (
    MessageChain,
)
from astrbot.core.message.components import Json, Plain
from astrbot.core.provider.entities import (
    ProviderRequest,
    LLMResponse,
    ToolCallMessageSegment,
    AssistantMessageSegment,
    ToolCallsResult,
)
from mcp.types import (
    TextContent,
    ImageContent,
    EmbeddedResource,
    TextResourceContents,
    BlobResourceContents,
    CallToolResult,
)
from astrbot import logger

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override


class ToolLoopAgentRunner(BaseAgentRunner[TContext]):
    @override
    async def reset(
        self,
        provider: Provider,
        request: ProviderRequest,
        run_context: ContextWrapper[TContext],
        tool_executor: BaseFunctionToolExecutor[TContext],
        agent_hooks: BaseAgentRunHooks[TContext],
        **kwargs: T.Any,
    ) -> None:
        self.req = request
        self.streaming = kwargs.get("streaming", False)
        self.provider = provider
        self.final_llm_resp = None
        self._state = AgentState.IDLE
        self.tool_executor = tool_executor
        self.agent_hooks = agent_hooks
        self.run_context = run_context

    def _transition_state(self, new_state: AgentState) -> None:
        """转换 Agent 状态"""
        if self._state != new_state:
            logger.debug(f"Agent state transition: {self._state} -> {new_state}")
            self._state = new_state

    @staticmethod
    def _trace_chain_response_type(chain: MessageChain) -> str:
        if chain.type == "tool_direct_result":
            return "tool_call_result"
        return chain.type or "streaming_delta"

    def _agent_responses_from_trace_chains(
        self, llm_response: LLMResponse
    ) -> T.Iterator[AgentResponse]:
        for chain in getattr(llm_response, "trace_chains", []) or []:
            if chain is None:
                continue
            yield AgentResponse(
                type=self._trace_chain_response_type(chain),
                data=AgentResponseData(chain=chain),
            )

    @staticmethod
    def _tool_schema_for(
        req: ProviderRequest, tool_name: str
    ) -> tuple[str | None, dict | None]:
        if not req.func_tool:
            return None, None
        func_tool = req.func_tool.get_func(tool_name)
        if not func_tool:
            return None, None
        return func_tool.description, func_tool.parameters

    def _tool_call_chain(
        self,
        req: ProviderRequest,
        tool_name: str,
        tool_args: dict,
        tool_call_id: str,
        status: str = "in_progress",
    ) -> MessageChain:
        description, schema = self._tool_schema_for(req, tool_name)
        return MessageChain(
            type="tool_call",
            chain=[
                Json(
                    data={
                        "id": tool_call_id,
                        "name": tool_name,
                        "args": tool_args,
                        "status": status,
                        "ts": time.time(),
                        "description": description,
                        "schema": schema,
                    }
                ),
                Plain(f"🔨 调用工具: {tool_name}"),
            ],
        )

    def _tool_result_chain(
        self,
        req: ProviderRequest,
        tool_name: str,
        tool_call_id: str,
        content: str,
        status: str = "completed",
    ) -> MessageChain:
        description, schema = self._tool_schema_for(req, tool_name)
        return MessageChain(
            type="tool_call_result",
            chain=[
                Json(
                    data={
                        "id": tool_call_id,
                        "name": tool_name,
                        "status": status,
                        "ts": time.time(),
                        "result": content,
                        "description": description,
                        "schema": schema,
                    }
                ),
                Plain(content),
            ],
        )

    async def _iter_llm_responses(self) -> T.AsyncGenerator[LLMResponse, None]:
        """Yields chunks *and* a final LLMResponse."""
        if self.streaming:
            stream = self.provider.text_chat_stream(**self.req.__dict__)
            async for resp in stream:  # type: ignore
                yield resp
        else:
            yield await self.provider.text_chat(**self.req.__dict__)

    @override
    async def step(self):
        """
        Process a single step of the agent.
        This method should return the result of the step.
        """
        if not self.req:
            raise ValueError("Request is not set. Please call reset() first.")

        if self._state == AgentState.IDLE:
            try:
                await self.agent_hooks.on_agent_begin(self.run_context)
            except Exception as e:
                logger.error(f"Error in on_agent_begin hook: {e}", exc_info=True)

        # 开始处理，转换到运行状态
        self._transition_state(AgentState.RUNNING)
        llm_resp_result = None

        async for llm_response in self._iter_llm_responses():
            assert isinstance(llm_response, LLMResponse)
            for trace_resp in self._agent_responses_from_trace_chains(llm_response):
                yield trace_resp
            if llm_response.is_chunk:
                if (
                    getattr(llm_response, "trace_chains", None)
                    and not llm_response.result_chain
                    and not llm_response.completion_text
                ):
                    continue
                if llm_response.result_chain:
                    yield AgentResponse(
                        type="streaming_delta",
                        data=AgentResponseData(chain=llm_response.result_chain),
                    )
                else:
                    yield AgentResponse(
                        type="streaming_delta",
                        data=AgentResponseData(
                            chain=MessageChain().message(llm_response.completion_text)
                        ),
                    )
                continue
            llm_resp_result = llm_response
            break  # got final response

        if not llm_resp_result:
            return

        # 处理 LLM 响应
        llm_resp = llm_resp_result

        if llm_resp.role == "err":
            # 如果 LLM 响应错误，转换到错误状态
            self.final_llm_resp = llm_resp
            self._transition_state(AgentState.ERROR)
            yield AgentResponse(
                type="err",
                data=AgentResponseData(
                    chain=MessageChain().message(
                        f"LLM 响应错误: {llm_resp.completion_text or '未知错误'}"
                    )
                ),
            )

        if not llm_resp.tools_call_name:
            # 如果没有工具调用，转换到完成状态
            self.final_llm_resp = llm_resp
            self._transition_state(AgentState.DONE)
            try:
                await self.agent_hooks.on_agent_done(self.run_context, llm_resp)
            except Exception as e:
                logger.error(f"Error in on_agent_done hook: {e}", exc_info=True)

        # 返回 LLM 结果
        if llm_resp.result_chain:
            yield AgentResponse(
                type="llm_result",
                data=AgentResponseData(chain=llm_resp.result_chain),
            )
        elif llm_resp.completion_text:
            yield AgentResponse(
                type="llm_result",
                data=AgentResponseData(
                    chain=MessageChain().message(llm_resp.completion_text)
                ),
            )

        # 如果有工具调用，还需处理工具调用
        if llm_resp.tools_call_name:
            tool_call_result_blocks = []
            for tool_call_name, tool_call_args, tool_call_id in zip(
                llm_resp.tools_call_name,
                llm_resp.tools_call_args,
                llm_resp.tools_call_ids,
            ):
                yield AgentResponse(
                    type="tool_call",
                    data=AgentResponseData(
                        chain=self._tool_call_chain(
                            self.req, tool_call_name, tool_call_args, tool_call_id
                        )
                    ),
                )
            async for result in self._handle_function_tools(self.req, llm_resp):
                if isinstance(result, list):
                    tool_call_result_blocks = result
                elif isinstance(result, MessageChain):
                    yield AgentResponse(
                        type="tool_call_result",
                        data=AgentResponseData(chain=result),
                    )
            # 将结果添加到上下文中
            tool_calls_result = ToolCallsResult(
                tool_calls_info=AssistantMessageSegment(
                    role="assistant",
                    tool_calls=llm_resp.to_openai_tool_calls(),
                    content=llm_resp.completion_text,
                ),
                tool_calls_result=tool_call_result_blocks,
            )
            self.req.append_tool_calls_result(tool_calls_result)

    async def _handle_function_tools(
        self,
        req: ProviderRequest,
        llm_response: LLMResponse,
    ) -> T.AsyncGenerator[MessageChain | list[ToolCallMessageSegment], None]:
        """处理函数工具调用。"""
        tool_call_result_blocks: list[ToolCallMessageSegment] = []
        logger.info(f"Agent 使用工具: {llm_response.tools_call_name}")

        # 执行函数调用
        for func_tool_name, func_tool_args, func_tool_id in zip(
            llm_response.tools_call_name,
            llm_response.tools_call_args,
            llm_response.tools_call_ids,
        ):
            try:
                if not req.func_tool:
                    return
                func_tool = req.func_tool.get_func(func_tool_name)
                logger.info(f"使用工具：{func_tool_name}，参数：{func_tool_args}")

                try:
                    await self.agent_hooks.on_tool_start(
                        self.run_context, func_tool, func_tool_args
                    )
                except Exception as e:
                    logger.error(f"Error in on_tool_start hook: {e}", exc_info=True)

                executor = self.tool_executor.execute(
                    tool=func_tool,
                    run_context=self.run_context,
                    **func_tool_args,
                )
                async for resp in executor:
                    if isinstance(resp, CallToolResult):
                        res = resp
                        if isinstance(res.content[0], TextContent):
                            tool_call_result_blocks.append(
                                ToolCallMessageSegment(
                                    role="tool",
                                    tool_call_id=func_tool_id,
                                    content=res.content[0].text,
                                )
                            )
                            yield self._tool_result_chain(
                                req, func_tool_name, func_tool_id, res.content[0].text
                            )
                        elif isinstance(res.content[0], ImageContent):
                            tool_call_result_blocks.append(
                                ToolCallMessageSegment(
                                    role="tool",
                                    tool_call_id=func_tool_id,
                                    content="返回了图片(已直接发送给用户)",
                                )
                            )
                            yield MessageChain(type="tool_direct_result").base64_image(
                                res.content[0].data
                            )
                        elif isinstance(res.content[0], EmbeddedResource):
                            resource = res.content[0].resource
                            if isinstance(resource, TextResourceContents):
                                tool_call_result_blocks.append(
                                    ToolCallMessageSegment(
                                        role="tool",
                                        tool_call_id=func_tool_id,
                                        content=resource.text,
                                    )
                                )
                                yield self._tool_result_chain(
                                    req, func_tool_name, func_tool_id, resource.text
                                )
                            elif (
                                isinstance(resource, BlobResourceContents)
                                and resource.mimeType
                                and resource.mimeType.startswith("image/")
                            ):
                                tool_call_result_blocks.append(
                                    ToolCallMessageSegment(
                                        role="tool",
                                        tool_call_id=func_tool_id,
                                        content="返回了图片(已直接发送给用户)",
                                    )
                                )
                                yield MessageChain(
                                    type="tool_direct_result"
                                ).base64_image(resource.blob)
                            else:
                                tool_call_result_blocks.append(
                                    ToolCallMessageSegment(
                                        role="tool",
                                        tool_call_id=func_tool_id,
                                        content="返回的数据类型不受支持",
                                    )
                                )
                                yield self._tool_result_chain(
                                    req,
                                    func_tool_name,
                                    func_tool_id,
                                    "返回的数据类型不受支持。",
                                    status="error",
                                )

                    elif resp is None:
                        # Tool 直接请求发送消息给用户
                        # 这里我们将直接结束 Agent Loop。
                        self._transition_state(AgentState.DONE)
                        if res := self.run_context.event.get_result():
                            if res.chain:
                                yield MessageChain(
                                    chain=res.chain, type="tool_direct_result"
                                )
                    else:
                        logger.warning(
                            f"Tool 返回了不支持的类型: {type(resp)}，将忽略。"
                        )

                try:
                    await self.agent_hooks.on_tool_end(
                        self.run_context, func_tool, func_tool_args, None
                    )
                except Exception as e:
                    logger.error(f"Error in on_tool_end hook: {e}", exc_info=True)

                self.run_context.event.clear_result()
            except Exception as e:
                logger.warning(traceback.format_exc())
                tool_call_result_blocks.append(
                    ToolCallMessageSegment(
                        role="tool",
                        tool_call_id=func_tool_id,
                        content=f"error: {str(e)}",
                    )
                )
                yield self._tool_result_chain(
                    req,
                    func_tool_name,
                    func_tool_id,
                    f"error: {str(e)}",
                    status="error",
                )

        # 处理函数调用响应
        if tool_call_result_blocks:
            yield tool_call_result_blocks

    def done(self) -> bool:
        """检查 Agent 是否已完成工作"""
        return self._state in (AgentState.DONE, AgentState.ERROR)

    def get_final_llm_resp(self) -> LLMResponse | None:
        return self.final_llm_resp
