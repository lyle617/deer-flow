"""Task tool for delegating work to subagents."""

import asyncio
import logging
import time
import uuid
from dataclasses import replace
from typing import Annotated, Any

from langchain.tools import InjectedToolCallId, ToolRuntime
from langchain_core.tools import StructuredTool
from langgraph.config import get_stream_writer
from langgraph.typing import ContextT
from pydantic import BaseModel, Field

from deerflow.agents.lead_agent.prompt import get_skills_prompt_section
from deerflow.agents.thread_state import ThreadState
from deerflow.sandbox.security import LOCAL_BASH_SUBAGENT_DISABLED_MESSAGE, is_host_bash_allowed
from deerflow.subagents import SubagentExecutor, get_available_subagent_names, get_subagent_config
from deerflow.subagents.executor import (
    SubagentStatus,
    cleanup_background_task,
    get_background_task_result,
    request_cancel_background_task,
)

logger = logging.getLogger(__name__)
_GENERAL_PURPOSE_MIN_MAX_TURNS = 60


class _TaskToolInput(BaseModel):
    description: str = Field(
        description="A short (3-5 word) description of the task for logging/display. ALWAYS PROVIDE THIS PARAMETER FIRST."
    )
    prompt: str = Field(
        description="The task description for the subagent. Be specific and clear about what needs to be done. ALWAYS PROVIDE THIS PARAMETER SECOND."
    )
    subagent_type: str = Field(
        description="The type of subagent to use. ALWAYS PROVIDE THIS PARAMETER THIRD."
    )
    max_turns: int | None = Field(
        default=None,
        description="Optional maximum number of agent turns. Defaults to subagent's configured max.",
    )


_TASK_TOOL_DESCRIPTION = """Delegate a task to a specialized subagent that runs in its own context.

Subagents help you:
- Preserve context by keeping exploration and implementation separate
- Handle complex multi-step tasks autonomously
- Execute commands or operations in isolated contexts

Available subagent types depend on the active sandbox configuration:
- **general-purpose**: A capable agent for complex, multi-step tasks that require
  both exploration and action. Use when the task requires complex reasoning,
  multiple dependent steps, or would benefit from isolated context.
- **bash**: Command execution specialist for running bash commands. This is only
  available when host bash is explicitly allowed or when using an isolated shell
  sandbox such as `AioSandboxProvider`.

When to use this tool:
- Complex tasks requiring multiple steps or tools
- Tasks that produce verbose output
- When you want to isolate context from the main conversation
- Parallel research or exploration tasks

When NOT to use this tool:
- Simple, single-step operations (use tools directly)
- Tasks requiring user interaction or clarification
"""


class _TaskToolExecution:
    """Materialized task-tool execution context for sync/async polling."""

    def __init__(
        self,
        *,
        config: Any,
        description: str,
        task_id: str,
        trace_id: str | None,
        writer: Any,
        max_poll_count: int,
    ) -> None:
        self.config = config
        self.description = description
        self.task_id = task_id
        self.trace_id = trace_id
        self.writer = writer
        self.max_poll_count = max_poll_count


def _tomato_parent_tool_groups(
    runtime: ToolRuntime[ContextT, ThreadState] | None,
) -> list[str] | None:
    if runtime is None or not runtime.context:
        return None
    raw_groups = runtime.context.get("tomato_tool_groups")
    if not isinstance(raw_groups, list):
        return None
    groups = [group for group in raw_groups if isinstance(group, str) and group.strip()]
    return groups or None


def _tomato_disable_mcp_for_subagents(
    runtime: ToolRuntime[ContextT, ThreadState] | None,
) -> bool:
    if runtime is None or not runtime.context:
        return False
    return runtime.context.get("tomato_disable_mcp_for_subagents") is True


def _task_failure_kind(error: str | None) -> str:
    if not error:
        return ""
    lowered = error.lower()
    if "graphrecursionerror" in lowered or "recursion limit" in lowered:
        return "recoverable_subagent_recursion_limit"
    return ""


def _task_failed_message(*, error: str | None, failure_kind: str) -> str:
    error_text = error or "unknown subagent failure"
    if failure_kind == "recoverable_subagent_recursion_limit":
        return (
            f"Task failed. {failure_kind}: {error_text}. "
            "Parent agent should continue without retrying the same subagent prompt; "
            "summarize available evidence or retry with a narrower delegated prompt."
        )
    return f"Task failed. Error: {error_text}"


def _prepare_task_execution(
    *,
    runtime: ToolRuntime[ContextT, ThreadState] | None,
    description: str,
    prompt: str,
    subagent_type: str,
    tool_call_id: str,
    max_turns: int | None,
) -> str | _TaskToolExecution:
    available_subagent_names = get_available_subagent_names()

    config = get_subagent_config(subagent_type)
    if config is None:
        available = ", ".join(available_subagent_names)
        return f"Error: Unknown subagent type '{subagent_type}'. Available: {available}"
    if subagent_type == "bash" and not is_host_bash_allowed():
        return f"Error: {LOCAL_BASH_SUBAGENT_DISABLED_MESSAGE}"

    overrides: dict[str, Any] = {}
    skills_section = get_skills_prompt_section()
    if skills_section:
        overrides["system_prompt"] = config.system_prompt + "\n\n" + skills_section
    if max_turns is not None:
        overrides["max_turns"] = max_turns
    if overrides:
        config = replace(config, **overrides)
    if subagent_type == "general-purpose" and config.max_turns < _GENERAL_PURPOSE_MIN_MAX_TURNS:
        logger.warning(
            "Clamping general-purpose subagent max_turns from %s to %s",
            config.max_turns,
            _GENERAL_PURPOSE_MIN_MAX_TURNS,
        )
        config = replace(config, max_turns=_GENERAL_PURPOSE_MIN_MAX_TURNS)

    sandbox_state = None
    thread_data = None
    thread_id = None
    parent_model = None
    trace_id = None

    if runtime is not None:
        sandbox_state = runtime.state.get("sandbox")
        thread_data = runtime.state.get("thread_data")
        thread_id = runtime.context.get("thread_id") if runtime.context else None
        if thread_id is None:
            thread_id = runtime.config.get("configurable", {}).get("thread_id")

        metadata = runtime.config.get("metadata", {})
        parent_model = metadata.get("model_name")
        trace_id = metadata.get("trace_id") or str(uuid.uuid4())[:8]

    from deerflow.tools import get_available_tools

    tool_kwargs: dict[str, Any] = {
        "model_name": parent_model,
        "subagent_enabled": False,
    }
    parent_tool_groups = _tomato_parent_tool_groups(runtime)
    if parent_tool_groups is not None:
        tool_kwargs["groups"] = parent_tool_groups
    if _tomato_disable_mcp_for_subagents(runtime):
        tool_kwargs["include_mcp"] = False
    tools = get_available_tools(**tool_kwargs)
    executor = SubagentExecutor(
        config=config,
        tools=tools,
        parent_model=parent_model,
        sandbox_state=sandbox_state,
        thread_data=thread_data,
        thread_id=thread_id,
        trace_id=trace_id,
    )
    task_id = executor.execute_async(prompt, task_id=tool_call_id)
    max_poll_count = (config.timeout_seconds + 60) // 5

    logger.info(
        "[trace=%s] Started background task %s (subagent=%s, timeout=%ss, polling_limit=%s polls)",
        trace_id,
        task_id,
        subagent_type,
        config.timeout_seconds,
        max_poll_count,
    )

    writer = get_stream_writer()
    writer({"type": "task_started", "task_id": task_id, "description": description})
    return _TaskToolExecution(
        config=config,
        description=description,
        task_id=task_id,
        trace_id=trace_id,
        writer=writer,
        max_poll_count=max_poll_count,
    )


def _poll_task_once(
    *,
    execution: _TaskToolExecution,
    poll_count: int,
    last_status: Any,
    last_message_count: int,
) -> tuple[str | None, Any, int]:
    result = get_background_task_result(execution.task_id)

    if result is None:
        logger.error(
            "[trace=%s] Task %s not found in background tasks",
            execution.trace_id,
            execution.task_id,
        )
        execution.writer(
            {
                "type": "task_failed",
                "task_id": execution.task_id,
                "error": "Task disappeared from background tasks",
            }
        )
        cleanup_background_task(execution.task_id)
        return (
            f"Error: Task {execution.task_id} disappeared from background tasks",
            last_status,
            last_message_count,
        )

    if result.status != last_status:
        logger.info(
            "[trace=%s] Task %s status: %s",
            execution.trace_id,
            execution.task_id,
            result.status.value,
        )
        last_status = result.status

    current_message_count = len(result.ai_messages)
    if current_message_count > last_message_count:
        for i in range(last_message_count, current_message_count):
            message = result.ai_messages[i]
            execution.writer(
                {
                    "type": "task_running",
                    "task_id": execution.task_id,
                    "message": message,
                    "message_index": i + 1,
                    "total_messages": current_message_count,
                }
            )
            logger.info(
                "[trace=%s] Task %s sent message #%s/%s",
                execution.trace_id,
                execution.task_id,
                i + 1,
                current_message_count,
            )
        last_message_count = current_message_count

    if result.status == SubagentStatus.COMPLETED:
        execution.writer(
            {"type": "task_completed", "task_id": execution.task_id, "result": result.result}
        )
        logger.info(
            "[trace=%s] Task %s completed after %s polls",
            execution.trace_id,
            execution.task_id,
            poll_count,
        )
        cleanup_background_task(execution.task_id)
        return f"Task Succeeded. Result: {result.result}", last_status, last_message_count
    if result.status == SubagentStatus.FAILED:
        failure_kind = _task_failure_kind(result.error)
        event_payload: dict[str, Any] = {
            "type": "task_failed",
            "task_id": execution.task_id,
            "error": result.error,
        }
        if failure_kind:
            event_payload["failure_kind"] = failure_kind
            event_payload["recoverable"] = True
        execution.writer(event_payload)
        logger.error(
            "[trace=%s] Task %s failed: %s",
            execution.trace_id,
            execution.task_id,
            result.error,
        )
        cleanup_background_task(execution.task_id)
        return (
            _task_failed_message(error=result.error, failure_kind=failure_kind),
            last_status,
            last_message_count,
        )
    if result.status == SubagentStatus.CANCELLED:
        execution.writer(
            {"type": "task_cancelled", "task_id": execution.task_id, "error": result.error}
        )
        logger.info(
            "[trace=%s] Task %s cancelled: %s",
            execution.trace_id,
            execution.task_id,
            result.error,
        )
        cleanup_background_task(execution.task_id)
        return "Task cancelled by user.", last_status, last_message_count
    if result.status == SubagentStatus.TIMED_OUT:
        execution.writer(
            {"type": "task_timed_out", "task_id": execution.task_id, "error": result.error}
        )
        logger.warning(
            "[trace=%s] Task %s timed out: %s",
            execution.trace_id,
            execution.task_id,
            result.error,
        )
        cleanup_background_task(execution.task_id)
        return f"Task timed out. Error: {result.error}", last_status, last_message_count

    if poll_count > execution.max_poll_count:
        timeout_minutes = execution.config.timeout_seconds // 60
        logger.error(
            "[trace=%s] Task %s polling timed out after %s polls (should have been caught by thread pool timeout)",
            execution.trace_id,
            execution.task_id,
            poll_count,
        )
        execution.writer({"type": "task_timed_out", "task_id": execution.task_id})
        return (
            f"Task polling timed out after {timeout_minutes} minutes. "
            f"This may indicate the background task is stuck. Status: {result.status.value}",
            last_status,
            last_message_count,
        )

    return None, last_status, last_message_count


async def _task_tool_async(
    runtime: ToolRuntime[ContextT, ThreadState],
    description: str,
    prompt: str,
    subagent_type: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    max_turns: int | None = None,
) -> str:
    execution = _prepare_task_execution(
        runtime=runtime,
        description=description,
        prompt=prompt,
        subagent_type=subagent_type,
        tool_call_id=tool_call_id,
        max_turns=max_turns,
    )
    if isinstance(execution, str):
        return execution

    poll_count = 0
    last_status = None
    last_message_count = 0

    try:
        while True:
            terminal, last_status, last_message_count = _poll_task_once(
                execution=execution,
                poll_count=poll_count,
                last_status=last_status,
                last_message_count=last_message_count,
            )
            if terminal is not None:
                return terminal
            await asyncio.sleep(5)
            poll_count += 1
    except asyncio.CancelledError:
        request_cancel_background_task(execution.task_id)

        async def cleanup_when_done() -> None:
            cleanup_poll_count = 0
            while True:
                result = get_background_task_result(execution.task_id)
                if result is None:
                    return
                if result.status in {
                    SubagentStatus.COMPLETED,
                    SubagentStatus.FAILED,
                    SubagentStatus.CANCELLED,
                    SubagentStatus.TIMED_OUT,
                } or getattr(result, "completed_at", None) is not None:
                    cleanup_background_task(execution.task_id)
                    return
                if cleanup_poll_count > execution.max_poll_count:
                    logger.warning(
                        "[trace=%s] Deferred cleanup for task %s timed out after %s polls",
                        execution.trace_id,
                        execution.task_id,
                        cleanup_poll_count,
                    )
                    return
                await asyncio.sleep(5)
                cleanup_poll_count += 1

        def log_cleanup_failure(cleanup_task: asyncio.Task[None]) -> None:
            if cleanup_task.cancelled():
                return
            exc = cleanup_task.exception()
            if exc is not None:
                logger.error(
                    "[trace=%s] Deferred cleanup failed for task %s: %s",
                    execution.trace_id,
                    execution.task_id,
                    exc,
                )

        logger.debug(
            "[trace=%s] Scheduling deferred cleanup for cancelled task %s",
            execution.trace_id,
            execution.task_id,
        )
        asyncio.create_task(cleanup_when_done()).add_done_callback(log_cleanup_failure)
        raise


def _task_tool_sync(
    runtime: ToolRuntime[ContextT, ThreadState],
    description: str,
    prompt: str,
    subagent_type: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    max_turns: int | None = None,
) -> str:
    execution = _prepare_task_execution(
        runtime=runtime,
        description=description,
        prompt=prompt,
        subagent_type=subagent_type,
        tool_call_id=tool_call_id,
        max_turns=max_turns,
    )
    if isinstance(execution, str):
        return execution

    poll_count = 0
    last_status = None
    last_message_count = 0

    while True:
        terminal, last_status, last_message_count = _poll_task_once(
            execution=execution,
            poll_count=poll_count,
            last_status=last_status,
            last_message_count=last_message_count,
        )
        if terminal is not None:
            return terminal
        time.sleep(5)
        poll_count += 1


task_tool = StructuredTool.from_function(
    func=_task_tool_sync,
    coroutine=_task_tool_async,
    name="task",
    description=_TASK_TOOL_DESCRIPTION,
    args_schema=_TaskToolInput,
)
