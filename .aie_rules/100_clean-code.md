---
description: "Core principles for writing clean, maintainable, and understandable code."
author: "AI Engineer Team"
version: "1.0"
---
# Clean Code Guidelines

## Constants Over Magic Numbers
- Replace hard-coded values with named constants
- Use descriptive constant names that explain the value's purpose
- Keep constants at the top of the file or in a dedicated constants file

## Meaningful Names
- Variables, functions, and classes should reveal their purpose
- Names should explain why something exists and how it's used
- Avoid abbreviations unless they're universally understood

## Smart Comments
- Don't comment on what the code does - make the code self-documenting
- Use comments to explain why something is done a certain way
- Document APIs, complex algorithms, and non-obvious side effects

## Single Responsibility
- Each function should do exactly one thing
- Functions should be small and focused
- If a function needs a comment to explain what it does, it should be split

## Function Arguments
- Limit the number of function arguments (ideally 0-2, 3 at most).
- Avoid boolean flags as parameters; consider splitting the function or using enums/objects.
- If multiple arguments are related, group them into an object/struct.

## DRY (Don't Repeat Yourself)
- Extract repeated code into reusable functions
- Share common logic through proper abstraction
- Maintain single sources of truth

## Clean Structure
- Keep related code together
- Organize code in a logical hierarchy
- Use consistent file and folder naming conventions
- Vertical Formatting: Keep lines short. Separate concepts with blank lines. Keep related code vertically dense.

## Encapsulation
- Hide implementation details
- Expose clear interfaces
- Move nested conditionals into well-named functions

## Code Quality Maintenance
- Refactor continuously
- Fix technical debt early
- Leave code cleaner than you found it

## Error Handling
- Use exceptions rather than error codes where appropriate.
- Provide context with exceptions.
- Don't return null or pass null unless the API/language idiomatically supports it for that case.

## Testing
- Write tests before fixing bugs
- Keep tests readable and maintainable
- Test edge cases and error conditions
- Ensure tests are fast, independent, repeatable, self-validating, and timely (FIRST principles).

## Version Control
- Write clear commit messages
- Make small, focused commits
- Use meaningful branch names 
