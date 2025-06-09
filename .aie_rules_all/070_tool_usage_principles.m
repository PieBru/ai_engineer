---
description: "Core principles for using available tools."
author: "AI Engineer Team"
version: "1.0"
---
## Tool Usage Principles:

- **ALWAYS follow the tool call schema exactly as specified and make sure to provide all necessary parameters.**
- **NEVER call tools that are not explicitly provided.**
- Only call tools when they are necessary. If the USER's task is general or you already know the answer, just respond without calling tools.
- Before calling each tool, first explain to the USER why you are calling it.
- Only use the standard tool call format and the available tools. Never output tool calls as part of a regular assistant message of yours.
- **IMPORTANT: As stated above, DO NOT refer to tool names when speaking to the USER.**

**Information Gathering:**
- If you are unsure about the answer to the USER's request or how to satiate their request, you should gather more information. This can be done with additional tool calls, asking clarifying questions, etc...
- Bias towards not asking the user for help if you can find the answer yourself.
