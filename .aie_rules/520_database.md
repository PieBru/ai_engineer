---
description: "Comprehensive database best practices, including Prisma, Supabase, and general design principles."
author: "AI Engineer Team"
version: "1.0"
---

# Database Best Practices

## Prisma Setup
- Use proper schema design
- Implement proper migrations
- Use proper relation definitions
- Configure proper connection
- Implement proper seeding
- Use proper client setup

## Prisma Models
- Use proper model naming
- Implement proper relations
- Use proper field types
- Define proper indexes
- Implement proper constraints
- Use `@default(now())` or similar for `createdAt` and `updatedAt` timestamps.
- Use proper enums

## Prisma Queries
- Use proper query optimization
- Implement proper filtering
- Use proper relations loading
- Handle transactions properly
- Implement proper pagination
- Use `select` or `include` wisely to fetch only necessary data.
- Use proper aggregations

## Supabase Setup
- Configure proper project setup
- Implement proper authentication
- Use proper database setup
- Configure proper storage
- Implement proper policies
- Use proper client setup

## Supabase Security
- Implement proper RLS policies
- Use proper authentication
- Configure proper permissions
- Handle sensitive data properly
- Implement proper backups
- Use proper encryption

## Supabase Queries
- Use proper query optimization
- Implement proper filtering
- Use proper joins
- Handle real-time properly
- Utilize database functions and triggers for complex logic where appropriate.
- Implement proper pagination
- Use proper functions

## Database Design
- Use proper normalization
- Implement proper indexing
- Use proper constraints
- Define proper relations
- Implement proper cascades
- Choose appropriate data types to optimize storage and performance.
- Use proper data types

## Performance
- Use proper connection pooling
- Implement proper caching
- Use proper query optimization
- Handle N+1 queries properly
- Implement proper batching
- Regularly analyze query performance (e.g., using `EXPLAIN`).
- Monitor performance metrics

## Security
- Use proper authentication
- Implement proper authorization
- Handle sensitive data properly
- Use proper encryption
- Implement proper backups
- Regularly review access controls and permissions.
- Monitor security issues

## Best Practices
- Follow database conventions
- Use proper migrations
- Implement proper versioning
- Handle errors properly
- Write clear and comprehensive documentation for the database schema and any complex queries or stored procedures.
- Document schema properly
- Monitor database health 

## Data Integrity
- Enforce data integrity using constraints (NOT NULL, UNIQUE, CHECK, FOREIGN KEY).
- Use transactions to ensure atomicity of operations.
- Implement validation at the application layer in addition to database constraints.
