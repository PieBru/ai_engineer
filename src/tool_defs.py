# /run/media/piero/NVMe-4TB/Piero/AI/AI-Engineer/tool_defs.py
RISKY_TOOLS = {"create_file", "create_multiple_files", "edit_file", "connect_remote_mcp_sse"}

tools = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the content of a single file from the filesystem",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The path to the file to read (relative or absolute)",
                    }
                },
                "required": ["file_path"]
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_multiple_files",
            "description": "Read the content of multiple files from the filesystem",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Array of file paths to read (relative or absolute)",
                    }
                },
                "required": ["file_paths"]
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_file",
            "description": "Create a new file or overwrite an existing file with the provided content",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The path where the file should be created",
                    },
                    "content": {
                        "type": "string",
                        "description": "The content to write to the file",
                    }
                },
                "required": ["file_path", "content"]
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_multiple_files",
            "description": "Create multiple files at once",
            "parameters": {
                "type": "object",
                "properties": {
                    "files": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string"},
                                "content": {"type": "string"}
                            },
                            "required": ["path", "content"]
                        },
                        "description": "Array of files to create with their paths and content",
                    }
                },
                "required": ["files"]
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Edit an existing file by replacing a specific snippet with new content",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The path to the file to edit",
                    },
                    "original_snippet": {
                        "type": "string",
                        "description": "The exact text snippet to find and replace",
                    },
                    "new_snippet": {
                        "type": "string",
                        "description": "The new text to replace the original snippet with",
                    }
                },
                "required": ["file_path", "original_snippet", "new_snippet"]
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "connect_local_mcp_stream",
            "description": "Connects to a local MCP server endpoint that provides a streaming response. Reads the stream and returns the aggregated data. Primarily for localhost or 127.0.0.1.",
            "parameters": {
                "type": "object",
                "properties": {
                    "endpoint_url": {
                        "type": "string",
                        "description": "The full URL of the local MCP streaming endpoint (e.g., http://localhost:8000/stream)."
                    },
                    "timeout_seconds": {
                        "type": "integer",
                        "description": "Timeout in seconds for the connection and for reading the entire stream (default: 30).",
                        "default": 30
                    },
                    "max_data_chars": {
                        "type": "integer",
                        "description": "Maximum number of characters to read from the stream before truncating (default: 10000).",
                        "default": 10000
                    }
                },
                "required": ["endpoint_url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "connect_remote_mcp_sse",
            "description": "Connects to a remote MCP server endpoint using Server-Sent Events (SSE). Listens for events and returns a summary of received events.",
            "parameters": {
                "type": "object",
                "properties": {
                    "endpoint_url": {
                        "type": "string",
                        "description": "The full URL of the remote MCP SSE endpoint."
                    },
                    "max_events": {
                        "type": "integer",
                        "description": "Maximum number of SSE events to process before closing the connection (default: 10).",
                        "default": 10
                    },
                    "listen_timeout_seconds": {
                        "type": "integer",
                        "description": "Timeout in seconds for the connection and for listening to events (default: 60).",
                        "default": 60
                    }
                },
                "required": ["endpoint_url"]
            }
        }
    }
]

