"""Common utilities for the project."""

from typing import Any, cast
from logging import Logger


from client import get_llama_stack_client
from llama_stack.distribution.library_client import LlamaStackAsLibraryClient
from models.config import Configuration


# TODO(lucasagomes): implement this function to retrieve user ID from auth
def retrieve_user_id(auth: Any) -> str:  # pylint: disable=unused-argument
    """Retrieve the user ID from the authentication handler.

    Args:
        auth: The Authentication handler (FastAPI Depends) that will
            handle authentication Logic.

    Returns:
        str: The user ID.
    """
    return "user_id_placeholder"


async def _register_mcp_toolgroups(
    client, mcp_servers, logger: Logger, is_async: bool = False
) -> None:
    """Common logic for registering MCP toolgroups, works for both sync and async clients."""
    # Get registered tools - handle both sync and async clients
    if is_async:
        registered_tools = await client.tools.list()
    else:
        registered_tools = client.tools.list()

    registered_toolgroups = [tool.toolgroup_id for tool in registered_tools]
    logger.debug("Registered toolgroups: %s", set(registered_toolgroups))

    # Register toolgroups for MCP servers if not already registered
    for mcp in mcp_servers:
        if mcp.name not in registered_toolgroups:
            logger.debug("Registering MCP server: %s, %s", mcp.name, mcp.url)

            registration_params = {
                "toolgroup_id": mcp.name,
                "provider_id": mcp.provider_id,
                "mcp_endpoint": {"uri": mcp.url},
            }

            if is_async:
                await client.toolgroups.register(**registration_params)
            else:
                client.toolgroups.register(**registration_params)

            logger.debug("MCP server %s registered successfully", mcp.name)


async def register_mcp_servers_async(
    logger: Logger, configuration: Configuration
) -> None:
    """Register Model Context Protocol (MCP) servers with the LlamaStack client."""
    if configuration.llama_stack.use_as_library_client:
        # Library client - use async interface
        # config.py validation ensures library_client_config_path is not None when use_as_library_client is True
        config_path = cast(str, configuration.llama_stack.library_client_config_path)
        client = LlamaStackAsLibraryClient(config_path)
        await client.async_client.initialize()

        await _register_mcp_toolgroups(
            client.async_client, configuration.mcp_servers, logger, is_async=True
        )
    else:
        # Service client - use sync interface
        client = get_llama_stack_client(configuration.llama_stack)

        await _register_mcp_toolgroups(
            client, configuration.mcp_servers, logger, is_async=False
        )
