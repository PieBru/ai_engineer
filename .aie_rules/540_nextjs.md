---
description: "Comprehensive Next.js best practices for building robust, scalable, and performant web applications using the App Router."
author: "AI Engineer Team"
version: "1.1" # Assuming this is an update
---

# Next.js Best Practices

## Project Structure
- Use the App Router directory structure
- Place components in `app` directory for route-specific components
- Place shared components in `components` directory
- Place utilities and helpers in `lib` directory
- Use lowercase with dashes for directories (e.g., `components/auth-wizard`)
- Use a `.gitignore` file to exclude build artifacts (`.next/`), node_modules, etc.
- Manage environment variables using `.env.local`, `.env.development`, `.env.production`.

## Components
- Use Server Components by default
- Mark client components explicitly with 'use client'
- Wrap client components in Suspense with fallback
- Use dynamic loading for non-critical components
- Implement proper error boundaries
- Use React Server Components (RSC) for rendering static or server-rendered parts.
- Place static content and interfaces at file end

## Performance
- Optimize images: Use WebP format, size data, lazy loading
- Minimize use of 'useEffect' and 'setState'
- Favor Server Components (RSC) where possible
- Use dynamic loading for non-critical components
- Optimize font loading using `next/font`.
- Implement proper caching strategies

## Data Fetching
- Use Server Components for data fetching when possible
- Implement proper error handling for data fetching
- Use appropriate caching strategies
- Handle loading and error states appropriately

## Routing
- Use the App Router conventions (`app/`) over the Pages Router (`pages/`).
- Use the App Router conventions
- Implement proper loading and error states for routes
- Use dynamic routes appropriately
- Handle parallel routes when needed

## Forms and Validation
- Use Zod for form validation
- Implement proper server-side validation
- Handle form errors appropriately
- Show loading states during form submission

## State Management
- Minimize client-side state
- Use React Context sparingly
- Prefer server state when possible
- Implement proper loading states 

## Middleware
- Use `middleware.ts` (or `.js`) for logic that runs before a request is completed.
- Use middleware for authentication, internationalization, or custom headers.
- Keep middleware logic lightweight and fast.
- Define the matcher config to specify which paths the middleware should run on.

## API Routes / Server Actions
- Use Route Handlers (`app/api/route.ts`) for building APIs.
- Use Server Actions for mutations directly from Client or Server Components.
- Implement proper request validation (e.g., using Zod).
- Handle errors consistently and return appropriate HTTP status codes.
- Secure API routes and Server Actions (e.g., authentication, rate limiting).
- Use caching headers for API responses where appropriate.

## Testing
- Use testing libraries like Jest, React Testing Library, or Vitest.
- Write unit tests for components, hooks, and utility functions.
- Implement integration tests for API routes and Server Actions.
- Use end-to-end testing tools like Playwright or Cypress for user flows.
- Mock external dependencies (API calls, database).
- Strive for good test coverage.

## Deployment
- Deploy to platforms optimized for Next.js (e.g., Vercel, Netlify).
- Understand the build output (Serverless Functions, Edge Functions, Static Assets).
- Configure environment variables correctly for production.
- Monitor application logs and performance in production.
- Implement proper CI/CD pipelines.

## Error Handling
- Implement `error.tsx` for route segment error handling.
- Use `notFound.tsx` for handling not-found errors.
- Implement global error handling for unexpected errors.
- Log errors effectively on the server and client.
- Show user-friendly error messages and fallback UI.

## Accessibility
- Use semantic HTML elements.
- Implement proper ARIA attributes.
- Ensure keyboard navigation is functional.
- Test with screen readers.
- Provide sufficient color contrast.
- Add alt text for images.
