# /home/piero/Piero/AI/AI-Engineer/src/network_utils.py
import httpx

def handle_local_mcp_stream(endpoint_url: str, timeout_seconds: int, max_data_chars: int) -> str:
    """
    Connects to a local MCP server endpoint that provides a streaming response.
    Reads the stream and returns the aggregated data.
    """
    try:
        parsed_url = httpx.URL(endpoint_url)
        if not (parsed_url.host.lower() in ("localhost", "127.0.0.1") and parsed_url.scheme.lower() in ("http", "https")):
             return f"Error: For connect_local_mcp_stream, endpoint_url must be for localhost (http or https). Provided: {endpoint_url}"
    except httpx.UnsupportedProtocol:
         return f"Error: Invalid or unsupported URL scheme for local MCP stream: {endpoint_url}"
    except Exception as e_val: # Catch other potential parsing errors
        return f"Error validating local MCP stream URL '{endpoint_url}': {str(e_val)}"

    data_buffer = []
    chars_read = 0
    timeout_config = httpx.Timeout(timeout_seconds, read=timeout_seconds)

    try:
        with httpx.Client(timeout=timeout_config) as client:
            with client.stream("GET", endpoint_url) as response:
                response.raise_for_status()
                for chunk in response.iter_text():
                    if chars_read + len(chunk) > max_data_chars:
                        remaining_len = max_data_chars - chars_read
                        data_buffer.append(chunk[:remaining_len])
                        data_buffer.append("... (data truncated)")
                        chars_read = max_data_chars
                        break
                    data_buffer.append(chunk)
                    chars_read += len(chunk)
        return "".join(data_buffer)
    except httpx.HTTPStatusError as e:
        error_detail = f"HTTP {e.response.status_code}"
        try:
            e.response.read()
            error_detail += f" - {e.response.text[:200]}"
        except httpx.ResponseNotRead:
            error_detail += " - (Error response body could not be read for details)"
        except Exception:
            error_detail += " - (Failed to retrieve error response body details)"
        return f"Error connecting to local MCP stream '{endpoint_url}': {error_detail}"
    except httpx.RequestError as e:
        return f"Error connecting to local MCP stream '{endpoint_url}': {str(e)}"
    except Exception as e:
        return f"An unexpected error occurred with local MCP stream '{endpoint_url}': {str(e)}"

def handle_remote_mcp_sse(endpoint_url: str, max_events: int, listen_timeout_seconds: int) -> str:
    """
    Connects to a remote MCP server endpoint using Server-Sent Events (SSE).
    Listens for events and returns a summary of received events.
    """
    try:
        parsed_url = httpx.URL(endpoint_url)
        if parsed_url.scheme.lower() not in ("http", "https"):
            return f"Error: For connect_remote_mcp_sse, endpoint_url must be a valid HTTP/HTTPS URL. Provided: {endpoint_url}"
    except httpx.UnsupportedProtocol:
        return f"Error: Invalid or unsupported URL scheme for remote MCP SSE: {endpoint_url}"
    except Exception as e_val:
        return f"Error validating remote MCP SSE URL '{endpoint_url}': {str(e_val)}"

    events_received = []
    timeout_config = httpx.Timeout(listen_timeout_seconds, read=listen_timeout_seconds)

    try:
        with httpx.Client(timeout=timeout_config) as client:
            with client.stream("GET", endpoint_url, headers={"Accept": "text/event-stream"}) as response:
                response.raise_for_status()
                if "text/event-stream" not in response.headers.get("Content-Type", "").lower():
                    return f"Error: Endpoint '{endpoint_url}' did not return 'text/event-stream' content type. Got: {response.headers.get('Content-Type')}"

                current_event_data = []
                for line in response.iter_lines():
                    if not line: # Message boundary
                        if current_event_data:
                            events_received.append("".join(current_event_data))
                            current_event_data = []
                            if len(events_received) >= max_events:
                                break
                    elif line.startswith("data:"):
                        current_event_data.append(line[5:].strip() + "\n")

                if current_event_data and len(events_received) < max_events: # Capture last event if loop ended before empty line
                    events_received.append("".join(current_event_data).strip())

        summary = f"Received {len(events_received)} SSE event(s) from '{endpoint_url}'.\n"
        summary += "Last few events:\n" + "\n---\n".join(events_received[-5:]) # Show last 5 events
        return summary
    except httpx.HTTPStatusError as e:
        error_detail = f"HTTP {e.response.status_code}"
        try:
            e.response.read()
            error_detail += f" - {e.response.text[:200]}"
        except httpx.ResponseNotRead:
            error_detail += " - (Error response body could not be read for details)"
        except Exception:
            error_detail += " - (Failed to retrieve error response body details)"
        return f"Error connecting to remote MCP SSE '{endpoint_url}': {error_detail}"
    except httpx.RequestError as e:
        return f"Error connecting to remote MCP SSE '{endpoint_url}': {str(e)}"
    except Exception as e:
        return f"An unexpected error occurred with remote MCP SSE '{endpoint_url}': {str(e)}"