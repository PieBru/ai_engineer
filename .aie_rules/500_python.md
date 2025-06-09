---
description: "Comprehensive Python programming guidelines and best practices, including modern tooling and advanced concepts."
author: "AI Engineer Team"
version: "1.1"
---
# Python Best Practices

## Project Structure
- Use src-layout with `src/your_package_name/`
- Place tests in `tests/` directory parallel to `src/`
- Keep configuration in `config/` or as environment variables
- Store requirements in `requirements.txt` or `pyproject.toml`
- Place static files in `static/` directory
- Use a `.gitignore` file to exclude unnecessary files (e.g., `__pycache__/`, `*.pyc`, virtual environment directories).
- Use `templates/` for Jinja2 templates

## Code Style
- Follow Black code formatting
- Use isort for import sorting
- Follow PEP 8 style guidelines.
- Follow PEP 8 naming conventions:
  - snake_case for functions and variables
  - PascalCase for classes
  - UPPER_CASE for constants
- Maximum line length of 88 characters (Black default)
- Use f-strings for string formatting (Python 3.6+).
- Use absolute imports over relative imports

## Type Hints
- Use type hints for all function parameters and returns
- Import types from `typing` module
- Use `Optional[Type]` or `Type | None` (Python 3.10+) for optional values.
- Use `TypeVar` for generic types
- Use `Literal` for specific allowed string or number values.
- Consider using `TypedDict` for dictionary shapes.
- Define custom types in `types.py`
- Use `Protocol` for duck typing

## Flask Structure
- Use Flask factory pattern
- Organize routes using Blueprints
- Use Flask-SQLAlchemy for database
- Implement proper error handlers
- Use Flask-Login for authentication
- Structure views with proper separation of concerns

## Database
- Use SQLAlchemy ORM
- Implement database migrations with Alembic
- Use proper connection pooling
- Define models in separate modules
- Implement proper relationships
- Use proper indexing strategies
- Be mindful of N+1 query problems and use appropriate eager/lazy loading strategies.

## Authentication
- Use Flask-Login for session management
- Implement Google OAuth using Flask-OAuth
- Hash passwords with bcrypt
- Use proper session security
- Implement CSRF protection
- Use proper role-based access control

## API Design
- Use Flask-RESTful for REST APIs
- Implement proper request validation
- Use proper HTTP status codes
- Handle errors consistently
- Use proper response formats
- Implement proper rate limiting
- Version APIs (e.g., /api/v1/...).

## Testing
- Use pytest for testing
- Write tests for all routes
- Use pytest-cov for coverage
- Implement proper fixtures
- Use proper mocking with pytest-mock
- Strive for high test coverage, especially for critical paths.
- Test all error scenarios

## Security
- Use HTTPS in production
- Implement proper CORS
- Sanitize all user inputs
- Use proper session configuration
- Implement proper logging
- Regularly audit dependencies for vulnerabilities (e.g., using `pip-audit` or `safety`).
- Follow OWASP guidelines

## Performance
- Use proper caching with Flask-Caching
- Implement database query optimization
- Use proper connection pooling
- Implement proper pagination
- Use background tasks for heavy operations
- Profile code to identify bottlenecks before optimizing.
- Monitor application performance

## Error Handling
- Create custom exception classes
- Use proper try-except blocks
- Implement proper logging
- Return proper error responses
- Handle edge cases properly
- Provide clear and actionable error messages.
- Use proper error messages

## Documentation
- Use Google-style docstrings
- Document all public APIs
- Keep README.md updated
- Use proper inline comments
- Generate API documentation
- Document data models and schemas.
- Document environment setup

## Development Workflow
- Use virtual environments (venv)
- Implement pre-commit hooks
- Use linters like Flake8 or Pylint.
- Use proper Git workflow
- Follow semantic versioning
- Use proper CI/CD practices
- Implement proper logging
- Conduct code reviews.

## Dependencies
- Pin dependency versions
- Use requirements.txt for production
- Separate dev dependencies
- Use `pyproject.toml` for project metadata and dependency management (PEP 518, PEP 621).
- Use proper package versions
- Regularly update dependencies
- Check for security vulnerabilities
