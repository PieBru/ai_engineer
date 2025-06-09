---
description: "Comprehensive FastAPI best practices for building robust and scalable web APIs."
author: "AI Engineer Team"
version: "1.0"
---

# FastAPI Best Practices

## Project Structure
- Use proper directory structure
- Implement proper module organization
- Use proper dependency injection
- Keep routes organized by domain
- Implement proper middleware
- Use proper configuration management
- Utilize `Lifespan` events for application startup and shutdown logic.

## API Design
- Use proper HTTP methods
- Implement proper status codes
- Use proper request/response models
- Implement proper validation
- Use proper error handling
- Leverage FastAPI's automatic data validation and serialization.
- Document APIs with OpenAPI

## Models
- Use Pydantic models
- Implement proper validation
- Use proper type hints
- Keep models organized
- Use proper inheritance
- Use `Field` for additional validation and metadata in Pydantic models.
- Implement proper serialization

## Database
- Use proper ORM (SQLAlchemy)
- Implement proper migrations
- Use proper connection pooling
- Implement proper transactions
- Use proper query optimization
- Consider using asynchronous database drivers (e.g., `asyncpg` for PostgreSQL, `aiomysql` for MySQL) with async ORMs or query builders.
- Handle database errors properly

## Authentication
- Implement proper JWT authentication
- Use proper password hashing
- Implement proper role-based access
- Use proper session management
- Implement proper OAuth2
- Utilize FastAPI's `Security` utilities for dependency injection of security schemes.
- Handle authentication errors properly

## Security
- Implement proper CORS
- Use proper rate limiting
- Implement proper input validation
- Use proper security headers
- Handle security errors properly
- Regularly update dependencies to patch security vulnerabilities.
- Implement proper logging

## Performance
- Use proper caching
- Implement proper async operations
- Use proper background tasks
- Implement proper connection pooling
- Use proper query optimization
- Profile application performance to identify bottlenecks.
- Monitor performance metrics

## Testing
- Write proper unit tests
- Implement proper integration tests
- Use proper test fixtures
- Implement proper mocking
- Test error scenarios
- Utilize FastAPI's `TestClient` for testing API endpoints.
- Use proper test coverage

## Deployment
- Use proper Docker configuration
- Implement proper CI/CD
- Use proper environment variables
- Implement proper logging
- Use proper monitoring
- Consider using ASGI servers like Uvicorn or Hypercorn for production.
- Handle deployment errors properly

## Documentation
- Use proper docstrings
- Implement proper API documentation
- Use proper type hints
- Keep documentation updated
- Document error scenarios
- Use proper versioning

## Asynchronous Programming
- Use `async` and `await` for I/O-bound operations to improve concurrency.
- Ensure all dependencies (like database drivers, HTTP clients) used in async routes are also async-compatible.
- Be mindful of blocking calls in async code.
