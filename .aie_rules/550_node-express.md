---
description: "Comprehensive Node.js and Express.js best practices for building robust and scalable web applications and APIs."
author: "AI Engineer Team"
version: "1.0"
---

# Node.js and Express.js Best Practices

## Project Structure
- Use proper directory structure
-   (e.g., `src/` for source, `config/`, `controllers/`, `routes/`, `services/`, `models/`, `middleware/`, `utils/`, `tests/`)
- Implement proper module organization
- Use proper middleware organization
- Keep routes organized by domain
- Implement proper error handling
- Manage environment variables using `.env` files and a library like `dotenv`.
- Use proper configuration management

## Express Setup
- Use proper middleware setup
- Implement proper routing
- Use proper error handling
- Configure proper security middleware
- Implement proper validation
- Use template engines (like EJS, Pug, Handlebars) appropriately if serving HTML.
- Use proper static file serving

## API Design
- Use proper REST principles
- Implement proper versioning
- Use proper request validation
- Handle errors properly
- Implement proper response formats
- Use consistent error response structures.
- Document APIs properly

## Database Integration
- Use proper ORM/ODM
- Implement proper migrations
- Use proper connection pooling
- Sanitize inputs to prevent SQL/NoSQL injection.
- Implement proper transactions
- Use proper query optimization
- Handle database errors properly

## Authentication
- Implement proper JWT handling
- Use proper password hashing
- Implement proper session management
- Use proper OAuth integration
- Implement proper role-based access
- Handle auth errors properly

## Security
- Use proper CORS setup
- Implement proper rate limiting
- Use proper security headers
- Implement proper input validation
- Regularly update dependencies to patch security vulnerabilities.
- Use proper encryption
- Handle security vulnerabilities

## Performance
- Use proper caching
- Implement proper async operations
- Use proper connection pooling
- Implement proper logging
- Use proper monitoring
- Profile application to identify performance bottlenecks.
- Handle high traffic properly

## Testing
- Write proper unit tests
- Implement proper integration tests
- Use proper test runners
- Implement proper mocking
- Test error scenarios
- Utilize libraries like Supertest for API endpoint testing.
- Use proper test coverage

## Deployment
- Use proper Docker setup
- Implement proper CI/CD
- Use proper environment variables
- Configure proper logging
- Implement proper monitoring
- Use a process manager like PM2 in production.
- Handle deployment errors

## Best Practices
- Follow Node.js best practices
- Use proper async/await
- Implement proper error handling
- Use proper logging
- Handle process signals properly
- Document code properly

## TypeScript Integration (If Applicable)
- Use TypeScript for type safety and improved developer experience.
- Define interfaces or types for request/response bodies, parameters, and models.
- Utilize decorators for routes and middleware if using frameworks like NestJS or with tsyringe.

## Build and Tooling
- Use linters (e.g., ESLint) and formatters (e.g., Prettier) for code consistency.
- Use Nodemon for automatic server restarts during development.
- Manage dependencies with `npm` or `yarn` and keep `package.json` and lock files (`package-lock.json`, `yarn.lock`) in version control.
- Consider using a transpiler like Babel or `ts-node` if using newer JavaScript features or TypeScript.
