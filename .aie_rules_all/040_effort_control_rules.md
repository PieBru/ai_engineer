---
description: "Defines how the AI should manage its reasoning and reply effort."
author: "AI Engineer Team"
version: "1.0"
---
## Effort Control Settings (Instructions for AI):

For each of your turns, you will receive system instructions appended to the user's message, indicating the current `reasoning_effort` and `reply_effort` settings. You MUST adhere to these settings.
- `reasoning_effort` defines the depth of your internal thinking process:
  - 'low': Minimize or skip an explicit internal thinking phase. Aim for a direct answer. Your internal reasoning, if any is exposed (e.g. via <think> tags or similar mechanisms if you use them), should be very brief.
  - 'medium': Employ a standard, balanced thinking process.
  - 'high': Engage in a detailed and thorough internal monologue or thinking process. Your internal reasoning, if exposed, should be comprehensive.
- `reply_effort` defines the verbosity and detail of your final reply to the user:
  - 'low': Deliver a concise, summary-level answer, focusing on the key information. Be brief.
  - 'medium': Offer a standard level of detail in your reply, balancing conciseness with completeness. This is the default if not specified.
  - 'high': Provide a comprehensive, detailed, and expansive explanation in your final answer. Be thorough and elaborate.
