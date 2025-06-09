---
description: "Comprehensive React best practices for building modern, scalable, and maintainable user interfaces."
author: "AI Engineer Team"
version: "1.0"
---

# React Best Practices

## Component Structure
- Use functional components over class components
- Keep components small and focused
- Name components with PascalCase (e.g., `MyComponent`).
- Extract reusable logic into custom hooks
- Use composition over inheritance
- Implement proper prop types with TypeScript
- Use defaultProps for optional props where appropriate (or default values in destructuring for functional components).
- Split large components into smaller, focused ones

## Hooks
- Follow the Rules of Hooks
- Use custom hooks for reusable logic
- Keep hooks focused and simple
- Use appropriate dependency arrays in useEffect
- Implement cleanup in useEffect when needed
- Use `useRef` for managing focus, animations, or integrating with third-party DOM libraries.
- Avoid nested hooks

## State Management
- Use useState for local component state
- Implement useReducer for complex state logic
- Use Context API for shared state
- Keep state as close to where it's used as possible
- Avoid prop drilling through proper state management
- Use state management libraries (e.g., Redux, Zustand, Jotai, Recoil) only when necessary for complex global state.

## Performance
- Implement proper memoization (useMemo, useCallback)
- Use React.memo for expensive components
- Avoid unnecessary re-renders
- Implement proper lazy loading
- Use virtualization (e.g., `react-window` or `react-virtualized`) for long lists.
- Use proper key props in lists
- Profile and optimize render performance

## Forms
- Use controlled components for form inputs
- Implement proper form validation
- Handle form submission states properly
- Show appropriate loading and error states
- Use form libraries for complex forms
- Implement proper accessibility for forms
- Consider using libraries like React Hook Form or Formik for complex forms.

## Error Handling
- Implement Error Boundaries
- Handle async errors properly
- Show user-friendly error messages
- Implement proper fallback UI
- Log errors appropriately
- Handle edge cases gracefully

## Testing
- Write unit tests for components
- Implement integration tests for complex flows
- Use testing libraries like React Testing Library, Jest, or Vitest.
- Test user interactions
- Test error scenarios
- Implement proper mock data
- Mock API calls and other external dependencies.

## Accessibility
- Use semantic HTML elements
- Implement proper ARIA attributes
- Ensure keyboard navigation
- Manage focus appropriately, especially in modals and dynamic content.
- Test with screen readers
- Handle focus management
- Provide proper alt text for images

## Code Organization
- Group related components together
- Use proper file naming conventions
- Implement proper directory structure (e.g., feature-based, atomic design, or type-based).
- Keep styles close to components (e.g., CSS Modules, styled-components, Tailwind CSS, or co-located CSS files).
- Use proper imports/exports
- Document complex component logic 

## Routing
- Use a routing library like React Router for client-side navigation in SPAs (if not using a framework like Next.js that handles routing).
- Implement route-based code splitting.
- Handle protected routes and authentication flows.

## Development Workflow
- Use linters (e.g., ESLint with React plugins) and formatters (e.g., Prettier).
- Implement pre-commit hooks (e.g., Husky with lint-staged).
- Use a modern build tool (e.g., Vite, Create React App, or framework-specific tools like Next.js).
- Manage dependencies with `npm`, `yarn`, or `pnpm`.
- Keep dependencies updated and audit for vulnerabilities.
- Utilize React DevTools for debugging and profiling.
