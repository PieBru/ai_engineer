---
description: "Rules and available tools for network operations."
author: "AI Engineer Team"
version: "1.0"
---
## Network Operations (via function calls):

The following tools are available for network interactions:
- `connect_local_mcp_stream`: Connects to a local (localhost or 127.0.0.1) MCP server endpoint that provides a streaming HTTP response. Returns the aggregated data from the stream.
  Example: Fetching logs or real-time metrics from a local development server.
  The `endpoint_url` must start with `http://localhost` or `http://127.0.0.1`.
- `connect_remote_mcp_sse`: Connects to a remote MCP server endpoint using Server-Sent Events (SSE) over HTTP/HTTPS. Returns a summary of received events.
  Example: Monitoring status updates or notifications from a remote service.
  The `endpoint_url` must be a valid HTTP or HTTPS URL.

**Network Operation Specific Guidelines:**
- Clearly state the purpose of connecting to an endpoint.
- Use `connect_local_mcp_stream` only for `http://localhost...` or `http://127.0.0.1...` URLs.
- Be mindful of potential timeouts or if the service is not running when using network tools.
- The data returned will be a text summary or aggregation.
- When `connect_local_mcp_stream` returns data, if it appears to be structured (e.g., JSON lines, logs), try to parse and summarize it meaningfully. If it's unstructured text, summarize its main content.
- After `connect_remote_mcp_sse` provides a summary of events, analyze these events in the context of the user's original request. For example, if the user asked about a service's status, try to infer the status from the events.
