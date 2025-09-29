"""Handler for REST API call to provide answer to chat queries with conversation context."""

import ast
import json
import re
import logging
from typing import Annotated, Any, AsyncIterator, Iterator, cast

from llama_stack_client import APIConnectionError
from llama_stack_client import AsyncLlamaStackClient  # type: ignore
from llama_stack_client.types import UserMessage  # type: ignore

from llama_stack_client.lib.agents.event_logger import interleaved_content_as_str
from llama_stack_client.types.agents.agent_turn_response_stream_chunk import (
    AgentTurnResponseStreamChunk,
)
from llama_stack_client.types.shared import ToolCall
from llama_stack_client.types.shared.interleaved_content_item import TextContentItem

from fastapi import APIRouter, HTTPException, Request, Depends, status
from fastapi.responses import StreamingResponse

from auth import get_auth_dependency
from auth.interface import AuthTuple
from authorization.middleware import authorize
from client import AsyncLlamaStackClientHolder
from configuration import configuration
import metrics
from models.config import Action
from models.requests import ChatRequest
from models.database.conversations import UserConversation
from utils.endpoints import check_configuration_loaded, get_system_prompt
from utils.mcp_headers import mcp_headers_dependency, handle_mcp_headers_with_toolgroups
from utils.transcripts import store_transcript
from utils.types import TurnSummary
from utils.endpoints import validate_model_provider_override

# Import helper functions from streaming_query
from app.endpoints.streaming_query import (
    format_stream_data,
    stream_start_event,
    stream_end_event,
    stream_build_event,
)

from app.endpoints.query import (
    get_rag_toolgroups,
    is_input_shield,
    is_output_shield,
    is_transcripts_enabled,
    select_model_and_provider_id,
    validate_attachments_metadata,
    evaluate_model_hints,
)

logger = logging.getLogger("app.endpoints.handlers")
router = APIRouter(tags=["chat"])
auth_dependency = get_auth_dependency()


@router.post("/chat")
@authorize(Action.STREAMING_QUERY)
async def chat_endpoint_handler(  # pylint: disable=too-many-locals
    request: Request,
    chat_request: ChatRequest,
    auth: Annotated[AuthTuple, Depends(auth_dependency)],
    mcp_headers: dict[str, dict[str, str]] = Depends(mcp_headers_dependency),
) -> StreamingResponse:
    """
    Handle request to the /chat endpoint.

    This endpoint receives a chat request with conversation context, authenticates the user,
    selects the appropriate model and provider, and streams incremental response events from
    the Llama Stack backend to the client. Events include start, token updates, tool calls,
    turn completions, errors, and end-of-stream metadata. Unlike streaming_query, this endpoint
    maintains conversation context and creates a new agent for each request with session
    persistence disabled.

    Returns:
        StreamingResponse: An HTTP streaming response yielding SSE-formatted events for the
        chat lifecycle.

    Raises:
        HTTPException: Returns HTTP 500 if unable to connect to the Llama Stack server.
    """
    # Nothing interesting in the request
    _ = request

    check_configuration_loaded(configuration)

    # Enforce RBAC: optionally disallow overriding model/provider in requests
    validate_model_provider_override(chat_request, request.state.authorized_actions)

    # log Llama Stack configuration
    logger.info("Llama stack config: %s", configuration.llama_stack_configuration)

    user_id, _user_name, token = auth

    try:
        # try to get Llama Stack client
        client = AsyncLlamaStackClientHolder().get_client()
        llama_stack_model_id, model_id, provider_id = select_model_and_provider_id(
            await client.models.list(),
            *evaluate_model_hints(user_conversation=None, query_request=chat_request),
        )
        response, conversation_id = await retrieve_chat_response(
            client,
            llama_stack_model_id,
            chat_request,
            token,
            mcp_headers=mcp_headers,
        )
        metadata_map: dict[str, dict[str, Any]] = {}

        async def response_generator(
            turn_response: AsyncIterator[AgentTurnResponseStreamChunk],
        ) -> AsyncIterator[str]:
            """
            Generate SSE formatted streaming response for chat.

            Asynchronously generates a stream of Server-Sent Events (SSE) representing
            incremental responses from a language model turn.

            Yields start, token, tool call, turn completion, and end events as
            SSE-formatted strings. Collects the complete response for transcript
            storage if enabled.
            """
            chunk_id = 0
            summary = TurnSummary(
                llm_response="No response from the model", tool_calls=[]
            )

            # Send start event
            yield stream_start_event(conversation_id)

            async for chunk in turn_response:
                p = chunk.event.payload
                if p.event_type == "turn_complete":
                    summary.llm_response = interleaved_content_as_str(
                        p.turn.output_message.content
                    )
                elif p.event_type == "step_complete":
                    if p.step_details.step_type == "tool_execution":
                        summary.append_tool_calls_from_llama(p.step_details)

                for event in stream_build_event(chunk, chunk_id, metadata_map):
                    chunk_id += 1
                    yield event

            yield stream_end_event(metadata_map)

            if not is_transcripts_enabled():
                logger.debug("Transcript collection is disabled in the configuration")
            else:
                store_transcript(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    model_id=model_id,
                    provider_id=provider_id,
                    query_is_valid=True,  # TODO(lucasagomes): implement as part of query validation
                    query=chat_request.query,
                    query_request=chat_request,
                    summary=summary,
                    rag_chunks=[],  # TODO(lucasagomes): implement rag_chunks
                    truncated=False,  # TODO(lucasagomes): implement truncation as part
                    # of quota work
                    attachments=chat_request.attachments or [],
                )

        # Update metrics for the LLM call
        metrics.llm_calls_total.labels(provider_id, model_id).inc()

        return StreamingResponse(response_generator(response))
    # connection to Llama Stack server
    except APIConnectionError as e:
        # Update metrics for the LLM call failure
        metrics.llm_calls_failures_total.inc()
        logger.error("Unable to connect to Llama Stack: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "response": "Unable to connect to Llama Stack",
                "cause": str(e),
            },
        ) from e


async def retrieve_chat_response(
    client: AsyncLlamaStackClient,
    model_id: str,
    chat_request: ChatRequest,
    token: str,
    mcp_headers: dict[str, dict[str, str]] | None = None,
) -> tuple[AsyncIterator[AgentTurnResponseStreamChunk], str]:
    """
    Retrieve response from LLMs and agents for chat requests.

    Asynchronously retrieves a streaming response and conversation ID from the
    Llama Stack agent for a given chat request with conversation context.

    This function configures input/output shields, system prompt, and tool usage
    based on the request and environment. It prepares the agent with appropriate
    headers and toolgroups, validates attachments if present, and initiates a
    streaming turn with the conversation context messages and any provided documents.

    Unlike the regular streaming query, this creates a new agent for each request
    with session persistence disabled.

    Parameters:
        model_id (str): Identifier of the model to use for the query.
        chat_request (ChatRequest): The chat request with conversation context.
        token (str): Authentication token for downstream services.
        mcp_headers (dict[str, dict[str, str]], optional): Multi-cluster proxy
        headers for tool integrations.

    Returns:
        tuple: A tuple containing the streaming response object and the conversation ID.
    """
    available_input_shields = [
        shield.identifier
        for shield in filter(is_input_shield, await client.shields.list())
    ]
    available_output_shields = [
        shield.identifier
        for shield in filter(is_output_shield, await client.shields.list())
    ]
    if not available_input_shields and not available_output_shields:
        logger.info("No available shields. Disabling safety")
    else:
        logger.info(
            "Available input shields: %s, output shields: %s",
            available_input_shields,
            available_output_shields,
        )
    # use system prompt from request or default one
    system_prompt = get_system_prompt(chat_request, configuration)
    logger.debug("Using system prompt: %s", system_prompt)

    # TODO(lucasagomes): redact attachments content before sending to LLM
    # if attachments are provided, validate them
    if chat_request.attachments:
        validate_attachments_metadata(chat_request.attachments)

    agent, conversation_id, session_id = await get_chat_agent(
        client,
        model_id,
        system_prompt,
        available_input_shields,
        available_output_shields,
        chat_request.no_tools or False,
    )

    logger.debug("Conversation ID: %s, session ID: %s", conversation_id, session_id)
    # bypass tools and MCP servers if no_tools is True
    if chat_request.no_tools:
        mcp_headers = {}
        agent.extra_headers = {}
        toolgroups = None
    else:
        # preserve compatibility when mcp_headers is not provided
        if mcp_headers is None:
            mcp_headers = {}

        mcp_headers = handle_mcp_headers_with_toolgroups(mcp_headers, configuration)

        if not mcp_headers and token:
            for mcp_server in configuration.mcp_servers:
                mcp_headers[mcp_server.url] = {
                    "Authorization": f"Bearer {token}",
                }

        agent.extra_headers = {
            "X-LlamaStack-Provider-Data": json.dumps(
                {
                    "mcp_headers": mcp_headers,
                }
            ),
        }

        vector_db_ids = [
            vector_db.identifier for vector_db in await client.vector_dbs.list()
        ]
        toolgroups = (get_rag_toolgroups(vector_db_ids) or []) + [
            mcp_server.name for mcp_server in configuration.mcp_servers
        ]
        # Convert empty list to None for consistency with existing behavior
        if not toolgroups:
            toolgroups = None

    # Get the conversation messages including context and current query
    messages = chat_request.get_messages()

    response = await agent.create_turn(
        messages=messages,
        session_id=session_id,
        documents=chat_request.get_documents(),
        stream=True,
        toolgroups=toolgroups,
    )
    response = cast(AsyncIterator[AgentTurnResponseStreamChunk], response)

    return response, conversation_id


async def get_chat_agent(
    client: AsyncLlamaStackClient,
    model_id: str,
    system_prompt: str,
    available_input_shields: list[str],
    available_output_shields: list[str],
    no_tools: bool = False,
) -> tuple[any, str, str]:
    """Get a new agent for chat with session persistence disabled."""
    from llama_stack_client.lib.agents.agent import AsyncAgent
    from utils.types import GraniteToolParser
    from utils.suid import get_suid

    logger.debug("Creating new chat agent with session persistence disabled")
    agent = AsyncAgent(
        client,  # type: ignore[arg-type]
        model=model_id,
        instructions=system_prompt,
        input_shields=available_input_shields if available_input_shields else [],
        output_shields=available_output_shields if available_output_shields else [],
        tool_parser=None if no_tools else GraniteToolParser.get_parser(model_id),
        enable_session_persistence=False,  # Disable session persistence for chat
    )
    await agent.initialize()

    conversation_id = agent.agent_id
    session_id = await agent.create_session(get_suid())

    return agent, conversation_id, session_id
