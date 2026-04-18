# SPDX-License-Identifier: Apache-2.0
"""
MCP — Model Context Protocol integration.

Client, server registry, transport, and response models for connecting the
governance pipeline to MCP servers. Responses from MCP servers flow through
the sanitizer before reaching agent context (see ``kitelogik.tether`` for
the sanitization step).

Public API re-exports live here — import from ``kitelogik.mcp`` rather than
the submodules so import paths stay stable across point releases.
"""

from kitelogik.mcp.client import MCPCallError, MCPClient, MCPSupplyChainError
from kitelogik.mcp.models import MCPContent, MCPServer, MCPToolResult, MCPVerificationStatus
from kitelogik.mcp.registry import ServerRegistry
from kitelogik.mcp.stdio_transport import StdioTransport, StdioTransportError

__all__ = [
    "MCPCallError",
    "MCPClient",
    "MCPContent",
    "MCPServer",
    "MCPSupplyChainError",
    "MCPToolResult",
    "MCPVerificationStatus",
    "ServerRegistry",
    "StdioTransport",
    "StdioTransportError",
]
