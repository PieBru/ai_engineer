---
description: "Comprehensive guidelines for code excellence, analysis, discussion, code citation, and AI interaction standards. Consolidates rules from 010_code_analysis_rules.md, 100_clean-code.md, and 105_codequality.md."
author: "AI Engineer Team"
version: "1.0"
---

# Code Excellence, Analysis, and Interaction Guidelines

This document consolidates and organizes guidelines related to writing high-quality code, performing code analysis, and adhering to specific interaction protocols.

## 1. Core Code Excellence Principles (Clean Code)

These principles are fundamental to writing clean, maintainable, and understandable code.

### 1.1. Constants Over Magic Numbers
- Replace hard-coded values with named constants.
- Use descriptive constant names that explain the value's purpose.
- Keep constants at the top of the file or in a dedicated constants file.

### 1.2. Meaningful Names
- Variables, functions, and classes should reveal their purpose.
- Names should explain why something exists and how it's used.
- Avoid abbreviations unless they're universally understood.

### 1.3. Smart Comments
- Don't comment on what the code does - make the code self-documenting.
- Use comments to explain *why* something is done a certain way.
- Document APIs, complex algorithms, and non-obvious side effects.

### 1.4. Single Responsibility
- Each function should do exactly one thing.
- Functions should be small and focused.
- If a function needs a comment to explain what it does, it should be split.

### 1.5. Function Arguments
- Limit the number of function arguments (ideally 0-2, 3 at most).
- Avoid boolean flags as parameters; consider splitting the function or using enums/objects.
- If multiple arguments are related, group them into an object/struct.

### 1.6. DRY (Don't Repeat Yourself)
- Extract repeated code into reusable functions.
- Share common logic through proper abstraction.
- Maintain single sources of truth.

### 1.7. Clean Structure
- Keep related code together.
- Organize code in a logical hierarchy.
- Use consistent file and folder naming conventions.
- Vertical Formatting: Keep lines short. Separate concepts with blank lines. Keep related code vertically dense.

### 1.8. Encapsulation
- Hide implementation details.
- Expose clear interfaces.
- Move nested conditionals into well-named functions.

### 1.9. Code Quality Maintenance
- Refactor continuously.
- Fix technical debt early.
- Leave code cleaner than you found it.

### 1.10. Error Handling
- Use exceptions rather than error codes where appropriate.
- Provide context with exceptions.
- Don't return null or pass null unless the API/language idiomatically supports it for that case.

### 1.11. Testing
- Write tests before fixing bugs.
- Keep tests readable and maintainable.
- Test edge cases and error conditions.
- Ensure tests are fast, independent, repeatable, self-validating, and timely (FIRST principles).

### 1.12. Version Control (General Principles)
- Write clear commit messages.
- Make small, focused commits.
- Use meaningful branch names.
(Note: Specific Gitflow rules are detailed in `600_gitflow.md`)

## 2. Code Analysis, Discussion, and Citation

### 2.1. Code Analysis & Discussion
- Analyze code with expert-level insight.
- Explain complex concepts clearly.
- Suggest optimizations and best practices.
- Debug issues with precision.

### 2.2. Code Citation Format
When citing code regions or blocks in your responses, you MUST use the following format (as defined in `001_default.md` for consistency):
```
startLine:endLine:filepath
// ... existing code ...
```
Example:
```
12:15:app/components/Todo.tsx
// ... existing code ...
```
- `startLine`: The starting line number (inclusive)
- `endLine`: The ending line number (inclusive)
- `filepath`: The complete path to the file
- The code block should be enclosed in triple backticks.
- Use `// ... existing code ...` to indicate omitted code sections.

## 3. AI Interaction and Output Guidelines

These guidelines govern how information is presented and how changes are made to ensure clarity, control, and adherence to user requests.

### 3.1. Information and Change Presentation
- **Verify Information:** Always verify information before presenting it. Do not make assumptions or speculate without clear evidence.
- **File-by-File Changes:** Make changes file by file to allow the user to review each one.
- **Preserve Existing Code:** Don't remove unrelated code or functionalities. Pay attention to preserving existing structures.
- **Single Chunk Edits (per file):** Provide all edits for a single file in one chunk/tool call, rather than multiple-step instructions or explanations for modifications to the same file.
- **Code Formatting Consistency:** When providing code, ensure it adheres to the established formatting standards of the project (e.g., Black for Python, Prettier for JS/TS). Do not introduce new formatting styles.

### 3.2. Communication and Scope
- **No Apologies:** Never use apologies.
- **No Understanding Feedback (in code/docs):** Avoid giving feedback about your understanding (e.g., "I understand you want...") in comments or documentation you generate.
- **No Summaries (of changes):** Don't summarize changes made unless specifically asked. (This refers to avoiding redundant summaries if the changes are already clear from the diff/tool output).
- **No Inventions:** Don't invent changes other than what's explicitly requested.
- **No Unnecessary Confirmations:** Don't ask for confirmation of information already provided in the context or that is part of your task.
- **Exactness:** Follow instructions precisely. Do not deviate or add extra information unless explicitly asked.
- **No Implementation Checks (to user):** Don't ask the user to verify implementations that are visible in the provided context and part of your analysis.
- **No Unnecessary Updates:** Don't suggest updates or changes to files when there are no actual modifications needed based on the request.
- **Focus on the Request:** Address only the specific task or question asked. Avoid tangential discussions or suggestions.
- **No Current Implementation (unless asked):** Don't show or discuss the current implementation unless specifically requested as part of the task.

### 3.3. Technical Details in Output
- **No Whitespace Suggestions:** Don't suggest whitespace changes unless they are part of a broader formatting task or fixing a syntax error.
- **Provide Real File Links/Paths:** When referring to files, always use their actual paths.